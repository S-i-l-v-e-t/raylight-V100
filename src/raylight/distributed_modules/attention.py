import torch
import torch.distributed

from xfuser.core.long_ctx_attention import (
    xFuserLongContextAttention,
)

from yunchang.kernels import AttnType
from .sageattention_hf_patch import ensure_hf_fp8_cuda_kernel, ensure_hf_sm90_kernel
from .inner_attention import INNER_ATTENTION_KEY, InnerAttentionDispatcher

# FlashAttention for V100 (SM70) via flash-attention-v100 (1Cat cu128 build)
_HAS_FLASH_V100 = False
try:
    from flash_attn_v100.flash_attn_interface import flash_attn_bhmd_func as _flash_v100_bhmd_func
    _HAS_FLASH_V100 = True
except ImportError:
    pass

_ATTN_TYPE = None
_SYNC_ULYSSES = None


def set_attn_type(attn):
    global _ATTN_TYPE
    _ATTN_TYPE = attn


def get_attn_type():
    if _ATTN_TYPE is None:
        raise RuntimeError("_ATTN_TYPE is not initialized")
    else:
        return _ATTN_TYPE


def set_sync_ulysses(is_sync):
    global _SYNC_ULYSSES
    _SYNC_ULYSSES = is_sync


def get_sync_ulysses():
    if _SYNC_ULYSSES is None:
        raise RuntimeError("_SYNC_ULYSSES variable is not initialized")
    else:
        return _SYNC_ULYSSES


def _flash_v100_ring_attn_factory(orig_ring_attn):
    def ring_attn(q, k, v, *args, **kwargs):
        # Dense fp16 flash path on V100 only; defer everything else (ring>1,
        # kv cache, joint tensors, causal, mask-dependent args) to the original.
        use_flash = (
            _HAS_FLASH_V100
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability(0)[0] == 7
            and q.dtype == torch.float16
            and not kwargs.get("causal", False)
            and kwargs.get("window_size", (-1, -1)) == (-1, -1)
            and kwargs.get("alibi_slopes", None) is None
            and kwargs.get("dropout_p", 0.0) == 0.0
            and kwargs.get("attn_layer", None) is None
            and kwargs.get("attn_processor", None) is None
            and kwargs.get("joint_tensor_key", None) is None
            and kwargs.get("joint_tensor_value", None) is None
            and not kwargs.get("return_attn_probs", False)
        )
        if use_flash and torch.distributed.get_world_size(kwargs.get("group", None)) <= 1:
            out = _flash_v100_bhmd_func(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                softmax_scale=kwargs.get("softmax_scale", None),
            ).transpose(1, 2)
            return out
        return orig_ring_attn(q, k, v, *args, **kwargs)
    return ring_attn


def make_xfuser_attention(attn_type, sync_ulysses):
    print(f"Using XFuser {attn_type} attention, Sync Ulysses: {sync_ulysses}")
    attn = AttnType[attn_type]
    if attn_type == "SAGE_FP8_CUDA":
        ensure_hf_fp8_cuda_kernel()
    elif attn_type == "SAGE_FP8_SM90":
        ensure_hf_sm90_kernel

    xfuser_attn = xFuserLongContextAttention(use_sync=sync_ulysses, attn_type=attn)
    inner_dispatcher = InnerAttentionDispatcher(xfuser_attn)
    # NOTE: flash_attn_v100 (1Cat build) measured ~2x slower than torch
    # mem-efficient (TORCH_EFFICIENT) on V100 dense prefill; override disabled.
    # Re-enable with: xfuser_attn.ring_attn_fn = _flash_v100_ring_attn_factory(xfuser_attn.ring_attn_fn)

    def _attention_xfuser_unmask(
            q,
            k,
            v,
            heads,
            join_q=None,
            join_k=None,
            join_v=None,
            mask=None,
            attn_precision=None,
            skip_reshape=False,
            skip_output_reshape=False,
            *args,
            **kwargs):

        if skip_reshape:
            b, _, _, dim_head = q.shape
            if join_q is not None:
                j_b, _, _, j_dim_head = join_q.shape
        else:
            b, _, dim_head = q.shape
            dim_head //= heads
            q, k, v = map(
                lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
                (q, k, v),
            )
            if join_q is not None:
                j_b, _, j_dim_head = join_q.shape
                j_dim_head //= heads
                join_q, join_k, join_v = map(
                    lambda t: t.view(j_b, -1, heads, j_dim_head).transpose(1, 2),
                    (join_q, join_k, join_v),
                )

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
        query = q.transpose(1, 2)
        key = k.transpose(1, 2)
        value = v.transpose(1, 2)

        # For custom inner attentnion, such as SLA, maybe SOL
        transformer_options = kwargs.get("transformer_options", {})
        processor = transformer_options.get(INNER_ATTENTION_KEY)
        with inner_dispatcher.scope(processor, transformer_options):
            if join_q is not None:
                out = xfuser_attn(
                    None, query, key, value,
                    joint_strategy="rear",
                    joint_tensor_query=join_q.transpose(1, 2),
                    joint_tensor_key=join_k.transpose(1, 2),
                    joint_tensor_value=join_v.transpose(1, 2),
                    softmax_scale=kwargs.get("scale", None),
                ).transpose(1, 2)
            else:
                out = xfuser_attn(
                    None,
                    query,
                    key,
                    value,
                    softmax_scale=kwargs.get("scale", None),
                ).transpose(1, 2)
        if not skip_output_reshape:
            out = (
                out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            )
        return out

    return _attention_xfuser_unmask
