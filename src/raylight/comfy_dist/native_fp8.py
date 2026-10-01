"""Eager collective fallback for NATIVE (plain, non-Qtensor) FP8 weights.

Native FP8 checkpoints (QwenImage / Flux fp8_e4m3fn, ...) load their weights as
raw ``torch.float8_e4m3fn`` ``nn.Parameter``\\ s.  Under raylight FSDP these raw
FP8 params are sharded and re-assembled by torch FSDP2, and the full state dict
is initialized with ``set_model_state_dict(..., broadcast_from_rank0=True)``.
NCCL only supports FP8 collectives on sm90+ (Hopper), so on Volta/Ampere every
one of those dies with:

    FP8 reduction support begins with sm90 capable devices.

comfy_kitchen QuantizedTensor FP8 already avoids this by shipping its qdata as
uint8 through pre/post_all_gather hooks (see kitchen_patches/fp8.py).  Plain
FP8 tensors have no Qtensor hooks, so here we reinterpret the bytes as uint8 at
the two eager collectives raylight FSDP actually uses:

* ``torch.distributed.all_gather_into_tensor`` - FSDP2 unshard during forward.
* ``torch.distributed.broadcast`` - full state dict sync from rank 0.

FP8 and uint8 are both 1 byte, so the view is byte-exact and NCCL accepts the
uint8 collective on every architecture.  Only installed when the local device
cannot run FP8 NCCL collectives (compute capability < sm90), and only while an
FSDP sampling pass is active (see patch_enable_comfy_kitchen_fsdp).
"""

from __future__ import annotations

import torch

_FP8_TYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    getattr(torch, "float8_e8m0fnu", None),
)
_FP8_TYPES = tuple(dt for dt in _FP8_TYPES if dt is not None)

_PATCHED = False
_ORIG: dict[str, object] = {}


def _fp8_requires_uint8() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        capability = torch.cuda.get_device_capability()
    except Exception:
        return False
    return capability is None or capability[0] < 9


def _is_fp8(tensor: object) -> bool:
    return isinstance(tensor, torch.Tensor) and tensor.dtype in _FP8_TYPES


def _uint8_view(tensor: torch.Tensor) -> torch.Tensor | None:
    # view is only valid when the FP8 tensor maps 1:1 onto a uint8 tensor;
    # every collective we redirect (FSDP shards, state dict sync) is contiguous.
    if not tensor.is_contiguous():
        return None
    try:
        return tensor.view(torch.uint8)
    except Exception:
        return None


def install_native_fp8_collective_patch() -> None:
    global _PATCHED
    if _PATCHED or not _fp8_requires_uint8():
        return

    import torch.distributed as dist
    from torch.distributed import distributed_c10d

    orig_all_gather_into_tensor = distributed_c10d.all_gather_into_tensor
    orig_broadcast = distributed_c10d.broadcast

    def all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False):
        if _is_fp8(input_tensor) and _is_fp8(output_tensor):
            output_u8 = _uint8_view(output_tensor)
            input_u8 = _uint8_view(input_tensor)
            if output_u8 is not None and input_u8 is not None:
                return orig_all_gather_into_tensor(output_u8, input_u8, group=group, async_op=async_op)
        return orig_all_gather_into_tensor(output_tensor, input_tensor, group=group, async_op=async_op)

    def broadcast(tensor, src=None, group=None, async_op=False, group_src=None):
        if _is_fp8(tensor):
            tensor_u8 = _uint8_view(tensor)
            if tensor_u8 is not None:
                return orig_broadcast(tensor_u8, src=src, group=group, async_op=async_op, group_src=group_src)
        return orig_broadcast(tensor, src=src, group=group, async_op=async_op, group_src=group_src)

    for module in (dist, distributed_c10d):
        setattr(module, "all_gather_into_tensor", all_gather_into_tensor)
        setattr(module, "broadcast", broadcast)

    _ORIG["all_gather_into_tensor"] = orig_all_gather_into_tensor
    _ORIG["broadcast"] = orig_broadcast
    _PATCHED = True


def restore_native_fp8_collective_patch() -> None:
    global _PATCHED
    if not _PATCHED:
        return

    import torch.distributed as dist
    from torch.distributed import distributed_c10d

    for module in (dist, distributed_c10d):
        setattr(module, "all_gather_into_tensor", _ORIG["all_gather_into_tensor"])
        setattr(module, "broadcast", _ORIG["broadcast"])
    _ORIG.clear()
    _PATCHED = False
