"""MiniMax-H3 Turbo LoRA apply, worker-side (Raylight).

Ported from custom_nodes/comfyui-minimax-h3-turbo so the turbo LoRA can be
applied on Ray workers, where the diffusion model actually lives. Pure
model-patching logic with no node I/O: a node forwards (path, strength,
low_vram) to the worker and the worker calls apply_turbo_lora().

The generic Raylight LoRA path (comfy.sd.load_lora_for_models) misses the H3
key naming and is merge-only, which folds the tiny delta into the FP16-
converted QKV weights and rounds it away. Bypass adds the delta in activation
space instead, exactly like the standalone node's default.
"""

import math
import os

import torch
import torch.nn.functional as F

import comfy.lora
import comfy.weight_adapter
import comfy.utils
import comfy.patcher_extension

SHIFT_V, SHIFT_A = 12.0, 3.0


def _time_shift_sigma(sigma, fr, to):
    base = sigma / (fr + sigma * (1.0 - fr))
    return to * base / (1.0 + (to - 1.0) * base)


def _unique_t(timestep, shift_v, shift_a, has_vis_cond):
    sv = float((timestep.flatten()[0] / 1000.0).clamp(min=1e-6))
    t_v = 1.0 - sv
    t_a = 1.0 - _time_shift_sigma(sv, shift_v, shift_a)
    s = {t_v, t_a}
    if has_vis_cond:
        s.add(max(t_v, 0.999))
    return sorted(s)


_EGRID = None
_EGRID_PATH = None


def _egrid(egrid_path=None):
    global _EGRID, _EGRID_PATH
    path = egrid_path or os.path.join(os.path.dirname(__file__), "h3_silu_temb_grid.safetensors")
    if _EGRID is None or _EGRID_PATH != path:
        _EGRID = comfy.utils.load_torch_file(path)["silu_t_emb_grid"]   # [1025, 2688]
        _EGRID_PATH = path
    return _EGRID


def _interp_egrid(unique_t, E, device, dtype):
    E = E.to(device)
    n = E.shape[0]
    rows = []
    for t in unique_t:
        pos = min(max(t, 0.0), 1.0) * (n - 1)
        i0 = min(int(math.floor(pos)), n - 2)
        rows.append(torch.lerp(E[i0].float(), E[i0 + 1].float(), pos - i0))
    return torch.stack(rows).to(dtype)                                # [M, 2688]


def _make_adaln_forward(base, a, b, shared):
    """Curve-mode adaln injection as a forward-attribute patch: adds
    B @ A @ silu(t_emb) to the projection before the reference view/chunk.
    Only the .forward attribute is replaced, so adaln_proj.linear.weight
    stays at its natural path and dynamic-VRAM streaming backup/restore
    behaves exactly as unpatched. a/b are plain captured tensors (never
    registered) and are cast to x's device/dtype per call."""

    def forward(t_emb):
        x = base.linear(F.silu(t_emb) if base.apply_silu else t_emb)
        st = shared.get("silu_temb")
        if st is not None:
            av = a.to(x.device, x.dtype)
            bv = b.to(x.device, x.dtype)
            sv = st.to(x.device, x.dtype)
            x = x + (bv @ (av @ sv.T)).T                              # [M, out]
        x = x.view(x.shape[0] * base.modalities, base.expand * base.hidden)
        return x.chunk(base.expand, dim=-1)

    return forward


class _FrugalLoRA(comfy.weight_adapter.LoRAAdapter):
    """LoRA bypass adapter with a memory-frugal additive path.

    The stock bypass allocates ~3x the output activation transiently
    (out, out*scale, base+out); accumulating up(down(x))*scale straight into
    base_out in place keeps one temporary. base_out is the module's fresh
    output, so the in-place add is safe. Numerically identical to stock.
    Linear-only; anything else falls back to the stock path."""

    def bypass_forward(self, org_forward, x, *args, **kwargs):
        base_out = org_forward(x, *args, **kwargs)
        if getattr(self, "is_conv", False):
            return super().bypass_forward(org_forward, x, *args, **kwargs)
        up, down, alpha = self.weights[0], self.weights[1], self.weights[2]
        rank = down.shape[0]
        scale = (alpha / rank if alpha is not None else 1.0) * getattr(self, "multiplier", 1.0)
        down = down.to(dtype=x.dtype)
        up = up.to(dtype=x.dtype)
        return base_out.add_(F.linear(F.linear(x, down), up), alpha=scale)


def _apply_bypass_lora(new_model, lora, modules, strength):
    """Apply the low-rank update at run time (base(x) + lora(x)) via ComfyUI's
    bypass injection, so it is never folded into the weights. The stock
    model_lora_keys_unet does not recognise the H3 lora naming, so build the
    key map directly (module -> diffusion_model.<module>.weight)."""
    key_map = {m: "diffusion_model.{}.weight".format(m) for m in modules}
    loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)
    manager = comfy.weight_adapter.BypassInjectionManager()
    sd_keys = set(new_model.model.state_dict().keys())
    n = 0
    for key, adapter in loaded.items():
        if key not in sd_keys:
            continue
        if isinstance(adapter, comfy.weight_adapter.LoRAAdapter):
            adapter = _FrugalLoRA(adapter.loaded_keys, adapter.weights)
        elif not isinstance(adapter, comfy.weight_adapter.WeightAdapterBase):
            continue
        manager.add_adapter(key, adapter, strength=strength)
        n += 1
    injections = manager.create_injections(new_model.model)
    if manager.get_hook_count() > 0:
        new_model.set_injections("bypass_lora", injections)
    return n


def _apply_merge_lora(new_model, lora, modules, strength, fsdp=False):
    """Low-VRAM path: fold the low-rank update into the weights (add_patches,
    the same call ComfyUI's own load_lora_for_models makes). Cheapest on peak
    VRAM, but on FP16/quantized bases the delta is partly rounded away.

    Under FSDP the adapters must come from raylight's comfy_dist: its
    calculate_weight only handles raylight's own adapter classes, and a stock
    comfy LoRAAdapter there falls through to a len() on the adapter."""
    key_map = {m: "diffusion_model.{}.weight".format(m) for m in modules}
    if fsdp:
        from raylight.comfy_dist import lora as fsdp_lora
        loaded = fsdp_lora.load_lora(lora, key_map, log_missing=False)
    else:
        loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)
    return len(new_model.add_patches(loaded, strength))


def _int8_fused_fc2(dm, modules):
    """MLP fc2 modules whose base weight rides ComfyUI's fused int8 matmul.

    comfy.ops.linear_input_act calls the fused int8 kernel on linear.weight
    directly and never calls the module's forward, so a BypassForwardHook on
    fc2.forward never fires. Those fc2 must go through the merge path instead."""
    fused = []
    for m in modules:
        if not m.endswith(".mlp.fc2"):
            continue
        try:
            w = comfy.utils.get_attr(dm, m + ".weight")
        except Exception:
            continue
        if (getattr(w, "_layout_cls", None) == "TensorWiseINT8Layout"
                and not getattr(getattr(w, "_params", None), "transposed", False)):
            fused.append(m)
    return fused


def _inject_adaln_egrid(new_model, dm, lora, adaln, strength, egrid_path=None):
    """Pruned/curve base only: re-inject the adaln update at run time via a
    shared silu(t_emb) interpolated from the E-grid plus a forward-attribute
    patch on each adaln projection."""
    E = _egrid(egrid_path)
    shared = {"silu_temb": None}
    shift_v = float(getattr(dm, "sigma_shift_video", SHIFT_V))
    shift_a = float(getattr(dm, "sigma_shift_audio", SHIFT_A))

    def wrap(executor, *args, **kwargs):
        ts = args[1] if len(args) > 1 else kwargs.get("timestep")
        ctx = args[2] if len(args) > 2 else kwargs.get("context")
        payload = kwargs.get("minimax_payload") or {}
        has_vc = bool(payload.get("keyframes") or payload.get("refs"))
        us = _unique_t(ts, shift_v, shift_a, has_vc)
        shared["silu_temb"] = _interp_egrid(us, E, ctx.device, ctx.dtype)
        return executor(*args, **kwargs)

    new_model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "h3turbo", wrap)
    for name in adaln:                       # name = "....adaln_proj.linear"
        a = lora[name + ".lora_A.weight"]
        b = lora[name + ".lora_B.weight"] * strength
        key = "diffusion_model." + name.rsplit(".linear", 1)[0]
        new_model.add_object_patch(
            key + ".forward",
            _make_adaln_forward(new_model.get_model_object(key), a, b, shared))


def _add_dbg_wrapper(new_model, dm, tag, mode):
    """Observability: log that the lora is active on the first few forwards.
    In bypass mode qkv_proj.forward_owner reads BypassForwardHook iff live;
    in merge mode a weight_function on the layer indicates the patch is live."""
    st = {"n": 0}

    def wrap(executor, *args, **kwargs):
        if st["n"] < 6:
            st["n"] += 1
            try:
                m0 = dm.blocks[0].attn.qkv_proj
                owner = type(getattr(m0.forward, "__self__", None)).__name__
                has_wf = bool(getattr(m0, "weight_function", None)) or \
                    getattr(m0, "weight_lowvram_function", None) is not None
            except Exception as e:                       # noqa
                owner, has_wf = "err:%s" % e, "?"
            ts = args[1] if len(args) > 1 else kwargs.get("timestep")
            xx = args[0] if args else kwargs.get("x")
            try:
                vr = float(xx[0].float().pow(2).mean().sqrt())
                ar = float(xx[1].float().pow(2).mean().sqrt())
                dt = str(xx[0].dtype)
            except Exception:
                vr = ar = -1.0
                dt = "?"
            tsv = float(ts.flatten()[0]) if ts is not None else -1
            if mode == "merge":
                canary = (f"qkv_proj.forward_owner={owner} weight_patched={has_wf} "
                          f"(merge: delta folded into weights; owner is the base "
                          f"Linear, patch presence => lora ACTIVE)")
            else:
                canary = (f"qkv_proj.forward_owner={owner} "
                          f"(BypassForwardHook => lora ACTIVE; else => BASE ONLY!)")
            print(f"[H3TurboLoRA fwd {tag}/{mode}] call#{st['n']}  {canary}  is_injected="
                  f"{getattr(new_model, 'is_injected', '?')}  timestep={tsv:.2f}  "
                  f"video_rms={vr:.4f} audio_rms={ar:.4f} dtype={dt}", flush=True)
        return executor(*args, **kwargs)

    new_model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "h3turbo_dbg", wrap)


def apply_turbo_lora(model_patcher, lora, strength, low_vram=False, egrid_path=None, fsdp=False):
    """Apply the MiniMax-H3 Turbo LoRA to a ModelPatcher. Returns a new patcher.

    On a pruned/curve base the adaln update can't be a weight patch, so it is
    always re-injected at run time regardless of mode; everything else is the
    "backbone", which takes the bypass or merge path per low_vram. When fsdp is
    set, merge patches go through raylight's own adapter loading so its FSDP
    weight function can apply them.
    """
    dm = model_patcher.model.diffusion_model
    pruned = getattr(dm, "use_adaln_curves", False)
    modules = sorted({k.rsplit(".lora_", 1)[0] for k in lora})
    new_model = model_patcher.clone()
    mode = "merge" if low_vram else "bypass"

    if pruned:
        backbone = [m for m in modules if "adaln_proj" not in m]
        adaln = [m for m in modules if "adaln_proj" in m]
    else:
        backbone, adaln = modules, []

    n_fc2 = 0
    if low_vram:
        n = _apply_merge_lora(new_model, lora, backbone, strength, fsdp=fsdp)
    else:
        fc2_fused = set(_int8_fused_fc2(dm, backbone))
        bypass_mods = [m for m in backbone if m not in fc2_fused]
        n = _apply_bypass_lora(new_model, lora, bypass_mods, strength)
        if fc2_fused:
            n_fc2 = _apply_merge_lora(new_model, lora, sorted(fc2_fused), strength, fsdp=fsdp)
            n += n_fc2
    if pruned and adaln:
        _inject_adaln_egrid(new_model, dm, lora, adaln, strength, egrid_path=egrid_path)

    if low_vram:
        detail = f"{n} weights patched (merged)"
    else:
        injs = new_model.injections.get("bypass_lora", [])
        detail = f"{n - n_fc2} bypass adapters, {len(injs)} injections"
        if n_fc2:
            detail += f", {n_fc2} int8 fc2 via merge"
    extra = f" + {len(adaln)} adaln injected at run time" if adaln else ""
    print(f"[H3TurboLoRA/raylight] {'pruned' if pruned else 'full'} base [{mode}]: "
          f"strength={strength} | {len(backbone)} backbone modules, {detail}{extra}", flush=True)
    _add_dbg_wrapper(new_model, dm, "pruned" if pruned else "full", mode)
    return new_model
