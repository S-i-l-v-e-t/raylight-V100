import torch
import gc

import comfy
from comfy.ldm.minimax.model import pack_audio, patchify_video, rope_rotation_table, time_shift_sigma, unpack_audio, unpatchify_video, PackedLayout, VISUAL_COND_TIMESTEP, AUDIO_COND_TIMESTEP
from comfy.ldm.modules.attention import attention_pytorch
from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group

import raylight.distributed_modules.attention as xfuser_attn
from ..utils import pad_to_world_size

attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)


def _split_packed_sequence(h, rope_freqs, mod_segments):
    world_size = get_sequence_parallel_world_size()
    local_size = h.shape[0] // world_size
    start = get_sequence_parallel_rank() * local_size
    end = start + local_size
    local_segments = []
    for segment_start, segment_end, row in mod_segments:
        segment_start = max(segment_start, start)
        segment_end = min(segment_end, end)
        if segment_start < segment_end:
            local_segments.append((segment_start - start, segment_end - start, row))
    return h[start:end], rope_freqs[:, start:end], local_segments


def _convert_qkv_to_fp16(module):
    """递归转换所有 Attention 层的 QKV 权重为 FP16（原地）"""
    if hasattr(module, 'qkv_proj') and hasattr(module.qkv_proj, 'weight'):
        if module.qkv_proj.weight.dtype == torch.float32:
            with torch.no_grad():
                module.qkv_proj.weight.data = module.qkv_proj.weight.data.to(torch.float16)
                print(f"[MiniMax FP16] Converted QKV weight to FP16 for {module.__class__.__name__}")
    for child in module.children():
        _convert_qkv_to_fp16(child)


_original_model_init = None


def patched_model_init(self, *args, **kwargs):
    global _original_model_init
    _original_model_init(self, *args, **kwargs)
    _convert_qkv_to_fp16(self)
    torch.cuda.empty_cache()
    gc.collect()
    print("[MiniMax FP16] QKV weights converted to FP16 immediately after loading.")


def usp_attn_forward(self, x, rope_freqs=None, transformer_options={}):
    if not hasattr(self, '_weight_checked'):
        print(f"Rank {get_sequence_parallel_rank()}: qkv_proj.weight.dtype = {self.qkv_proj.weight.dtype}")
        setattr(self, '_weight_checked', True)
    seq_len = x.shape[0]
    residual_dtype = x.dtype

    x_fp16 = x.to(torch.float16)
    q, k, v = self.qkv_proj(x_fp16).split(self.heads * self.head_dim, dim=-1)

    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.view(seq_len, self.heads, self.head_dim)

    q = self.q_norm(q.view(seq_len, self.heads, self.head_dim))
    k = self.k_norm(k.view(seq_len, self.heads, self.head_dim))

    if rope_freqs is not None:
        rot = rope_freqs.shape[-3] * 2
        q[..., :rot], k[..., :rot] = comfy.quant_ops.ck.apply_rope_split_half(
            q[..., :rot], k[..., :rot], rope_freqs
        )

    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)

    out = xfuser_optimized_attention(
        q.to(torch.float16), k.to(torch.float16), v, self.heads, skip_reshape=True
    )
    out = out.to(residual_dtype)

    # ====== 每个 block 都用全局 K/V 以 FP32 重算音频行（音频必须 FP32，否则爆音） ======
    audio_global_ranges = transformer_options.get('minimax_audio_global_ranges', [])
    if audio_global_ranges:
        sp_start = transformer_options.get('sp_start', 0)
        sp_end = transformer_options.get('sp_end', seq_len)
        local_audio_ranges = []
        for gs, ge in audio_global_ranges:
            ls = max(gs, sp_start) - sp_start
            le = min(ge, sp_end) - sp_start
            if ls < le:
                local_audio_ranges.append((ls, le))

        # all_gather 是集体操作：所有 SP rank 必须同一组、同一顺序、同一形状地调用，
        # 不能按"本 rank 是否有音频行"跳过；只有重算按本地音频行执行。
        # K 和 V 拼成一次 all_gather，减少集合点数量。
        kv = torch.cat([k.to(torch.float16).contiguous(),
                        v.to(torch.float16).contiguous()], dim=-1)
        kv_global = get_sp_group().all_gather(kv, dim=2).to(torch.float32)
        k_global, v_global = torch.split(kv_global, self.head_dim, dim=-1)
        for ls, le in local_audio_ranges:
            out[:, ls:le] = attention_pytorch(
                q[:, :, ls:le], k_global, v_global, self.heads, mask=None, skip_reshape=True
            )
        del kv, kv_global

    return self.out_proj(out.squeeze(0))


def usp_dit_forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
    video_x, audio_x = x[0], x[1]
    orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, self.patch_size)
    if video_x.shape[0] != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    payload = minimax_payload or {}
    device = video_x.device
    dtype = context.dtype

    latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]
    layout = payload.get("layout")
    if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
        layout = PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                              keyframes=payload.get("keyframes"),
                              refs=payload.get("refs"),
                              frame_count=payload.get("frame_count"))

    shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video))
    shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio))
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))

    vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
    aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
    has_vis_cond = any(k in ("cond", "ref_img") for _, _, k in layout.segments)
    has_aud_cond = any(k == "ref_audio" for _, _, k in layout.segments)
    seg_t = {"text": t_v, "video": t_v, "audio": t_a,
             "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
             "ref_audio": max(t_a, aud_aug)}
    unique_t = sorted({t_v, t_a} | ({seg_t["cond"]} if has_vis_cond else set())
                      | ({seg_t["ref_audio"]} if has_aud_cond else set()))
    t_row = {t: i for i, t in enumerate(unique_t)}
    seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "ref_audio": 2}

    text_tags = payload.get("text_token_tags")
    mod_segments = []
    for a, b, kind in layout.segments:
        row_base = t_row[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            tags = text_tags.view(-1).tolist()
            run_start = 0
            for i in range(1, b - a + 1):
                if i == b - a or tags[i] != tags[run_start]:
                    mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                    run_start = i
        else:
            mod_segments.append((a, b, row_base + seg_tag[kind]))

    img_update = layout.img_update.to(device)
    audio_update = layout.audio_update.to(device)
    video_rows = patchify_video(video_x.to(torch.float32), self.patch_size)
    audio_rows = pack_audio(audio_x.to(torch.float32))
    cond_video_rows = self._cond_video_rows(payload, device)
    cond_audio_rows = self._cond_audio_rows(payload, device)

    all_video_rows = video_rows
    if cond_video_rows is not None:
        all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
        all_video_rows[~img_update] = cond_video_rows
        all_video_rows[img_update] = video_rows
    all_audio_rows = audio_rows
    if cond_audio_rows is not None:
        all_audio_rows = torch.empty(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)
        all_audio_rows[~audio_update] = cond_audio_rows
        all_audio_rows[audio_update] = audio_rows

    video_embed = self.video_patch_proj(all_video_rows).to(dtype)
    audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
    text_states = context[0]
    if text_states.shape[-1] != self.hidden_size:
        text_states = self.token_refiner(self.condition_proj(text_states),
                                         transformer_options=transformer_options)

    h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
    voff = aoff = 0
    for a, b, kind in layout.segments:
        n = b - a
        if kind == "text":
            h[a:b] = text_states
        elif kind in ("cond", "ref_img", "video"):
            h[a:b] = video_embed[voff:voff + n]
            voff += n
        else:
            h[a:b] = audio_embed[aoff:aoff + n]
            aoff += n

    t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
    if self.use_adaln_curves:
        table = comfy.model_management.cast_to(self.adaln_t_table, device=device)
        pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
    else:
        t_emb = self.time_embedder(t_vals).to(dtype)

    rope_freqs = rope_rotation_table(self.rope_freqs(layout.position_ids, device), dtype)

    # SP 切分准备
    h, h_orig_size = pad_to_world_size(h, dim=0)
    rope_freqs, _ = pad_to_world_size(rope_freqs, dim=1)

    world_size = get_sequence_parallel_world_size()
    total_pad_len = h.shape[0]
    local_len = total_pad_len // world_size
    sp_start = get_sequence_parallel_rank() * local_len
    sp_end = sp_start + local_len

    audio_global_ranges = []
    for start, end, kind in layout.segments:
        if kind in ('audio', 'ref_audio'):
            audio_global_ranges.append((start, end))

    transformer_options['sp_start'] = sp_start
    transformer_options['sp_end'] = sp_end
    transformer_options['minimax_audio_global_ranges'] = audio_global_ranges

    h, rope_freqs, mod_segments = _split_packed_sequence(h, rope_freqs, mod_segments)

    # DiT Blocks
    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), device, transformer_options)

    for i, block in enumerate(self.blocks):

        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block)
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                return {"img": block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                     transformer_options=args["transformer_options"])}
            h = blocks_replace[("double_block", i)](
                {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
                 "transformer_options": transformer_options},
                {"original_block": block_wrap})["img"]
        else:
            h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)

    if prefetch_queue is not None:
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)

    # SP 聚合
    h = get_sp_group().all_gather(h.contiguous(), dim=0)
    h = h[:h_orig_size]

    video_seg = next((a, b, t_row[seg_t["video"]]) for a, b, k in layout.segments if k == "video")
    audio_seg = next((a, b, t_row[seg_t["audio"]]) for a, b, k in layout.segments if k == "audio")
    v, a = self.final_layer(h, t_emb, video_seg, audio_seg)

    video_out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)
    video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
    audio_out = unpack_audio(a)

    return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]


# ========== 在模型加载时安装补丁 ==========
if not hasattr(comfy.ldm.minimax.model.MiniMaxH3Model, '_patched_by_raylight'):
    _original_model_init = comfy.ldm.minimax.model.MiniMaxH3Model.__init__
    comfy.ldm.minimax.model.MiniMaxH3Model.__init__ = patched_model_init
    comfy.ldm.minimax.model.MiniMaxH3Model._patched_by_raylight = True
    print("[MiniMax FP16] Patch installed: QKV weights will be converted to FP16 immediately after model load.")
