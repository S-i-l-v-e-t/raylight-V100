"""MiniMax-H3 Turbo 4-step sampler, worker-side (Raylight).

Ported from custom_nodes/comfyui-minimax-h3-turbo so the SAMPLER object (and
the _turbo_sampler function it holds) lives in a module importable on Ray
workers. A KSAMPLER holding a function from a custom-node module would fail
Ray deserialization on the worker; keeping the function here lets Ray
deserialize it by reference.

Auto-adapts to the ComfyUI version: on recent builds that handle the audio
schedule natively (ModelSamplingAV) it steps as a plain single-schedule flow
sampler; on older builds it steps video and audio on their separate clocks.
"""

import math

import torch
from tqdm.auto import trange

import comfy.samplers
import comfy.model_sampling

SHIFT_V, SHIFT_A = 12.0, 3.0


def _time_shift_sigma(sigma, fr, to):
    base = sigma / (fr + sigma * (1.0 - fr))
    return to * base / (1.0 + (to - 1.0) * base)


def _time_shift_slope(sigma, fr, to):
    base = sigma / (fr + sigma * (1.0 - fr))
    return (to * (1.0 + (fr - 1.0) * base) ** 2) / (fr * (1.0 + (to - 1.0) * base) ** 2)


def _audio_sigma(sv):
    return _time_shift_sigma(sv, SHIFT_V, SHIFT_A)


def _audio_slope(sv):
    return _time_shift_slope(sv, SHIFT_V, SHIFT_A)


def _latent_shapes(model):
    """[video_shape, audio_shape] the sampler is packing over — video latent is
    flattened first, then audio, so we need the split point."""
    guider = getattr(model, "inner_model", model)
    conds = getattr(guider, "conds", None)
    if conds:
        for cond_list in conds.values():
            for c in (cond_list or []):
                mc = c.get("model_conds", {}) if isinstance(c, dict) else {}
                if "latent_shapes" in mc:
                    return mc["latent_shapes"].cond
    return None


def _model_sampling(model):
    """The model's model_sampling instance, reached from the object a KSAMPLER
    hands the sampler function: KSamplerX0Inpaint -> CFGGuider -> predictor."""
    for chain in (("inner_model", "inner_model", "model_sampling"),
                  ("inner_model", "model_sampling"),
                  ("model_sampling",)):
        o = model
        try:
            for a in chain:
                o = getattr(o, a)
        except AttributeError:
            continue
        if o is not None:
            return o
    return None


def _native_av_schedule(model):
    """True when this ComfyUI resolves the MiniMax-H3 audio/video dual flow
    schedule natively via ModelSamplingAV. Recent ComfyUI carries the audio
    latent scaled onto the video schedule, so a plain single-schedule flow step
    is correct; re-applying the audio shift here would double-shift and corrupt
    the audio."""
    ms = _model_sampling(model)
    if ms is None:
        return False
    if getattr(ms, "audio_shift", None) is not None:
        return True
    av = getattr(comfy.model_sampling, "ModelSamplingAV", None)
    return av is not None and isinstance(ms, av)


@torch.no_grad()
def _turbo_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    _rms = lambda t: float(t.float().pow(2).mean().sqrt())

    if _native_av_schedule(model):
        # Recent ComfyUI: ModelSamplingAV already carries the audio stream scaled
        # onto the video schedule, so the pack is an ordinary single-schedule flow
        # latent. Step the whole pack with a plain flow (Euler) update.
        print(f"[H3TURBO sampler] native ModelSamplingAV -> single-schedule Euler  "
              f"sigmas={[round(float(s),4) for s in sigmas]}  x.shape={tuple(x.shape)} "
              f"dtype={x.dtype}", flush=True)
        for i in trange(len(sigmas) - 1, disable=disable):
            sv, sv_n = float(sigmas[i]), float(sigmas[i + 1])
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            d = (x - denoised) / sigmas[i]
            x = x + (sv_n - sv) * d
            print(f"[H3TURBO step {i}] sv={sv:.4f}->{sv_n:.4f}  "
                  f"denoised_rms={_rms(denoised):.4f} x_rms={_rms(x):.4f} d_rms={_rms(d):.4f}",
                  flush=True)
            if callback is not None:
                callback({"i": i, "denoised": denoised, "x": x,
                          "sigma": sigmas[i], "sigma_hat": sigmas[i]})
        return x

    # Legacy ComfyUI without ModelSamplingAV: video and audio ride separate flow
    # schedules (video shift 12, audio shift 3); step each on its own clock.
    shapes = _latent_shapes(model)
    if not shapes or len(shapes) < 2:
        raise RuntimeError(
            "MiniMaxH3TurboSampler expects the MiniMax-H3 video+audio latent "
            "(the EmptyMiniMaxH3LatentAV / MiniMaxH3ImageToVideo output).")
    v_numel = math.prod(shapes[0][1:])           # flat pack is [video | audio]
    a_numel = (x.shape[-1] - v_numel)
    print(f"[H3TURBO sampler] legacy dual-schedule (no native ModelSamplingAV)  "
          f"sigmas={[round(float(s),4) for s in sigmas]}  x.shape={tuple(x.shape)} "
          f"dtype={x.dtype}  v_numel={v_numel} a_numel={a_numel}  shapes={shapes}", flush=True)
    for i in trange(len(sigmas) - 1, disable=disable):
        sv, sv_n = float(sigmas[i]), float(sigmas[i + 1])
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        out = (x - denoised) / sigmas[i]
        xv, ov = x[..., :v_numel], out[..., :v_numel]
        xa, oa = x[..., v_numel:], out[..., v_numel:]
        xv = xv + (sv_n - sv) * ov               # video on its own sigma
        sl = _audio_slope(max(sv, 1e-6))
        xa = xa + (_audio_sigma(sv_n) - _audio_sigma(sv)) * (oa / sl)  # audio clock
        x = torch.cat([xv, xa], dim=-1)
        print(f"[H3TURBO step {i}] sv={sv:.4f}->{sv_n:.4f}  denoised_rms={_rms(denoised):.4f}  "
              f"video: x_rms={_rms(xv):.4f} v_rms={_rms(ov):.4f}  "
              f"audio: x_rms={_rms(xa):.4f} v_rms={_rms(oa):.4f} slope={sl:.4f}", flush=True)
        if callback is not None:
            callback({"i": i, "denoised": denoised, "x": x,
                      "sigma": sigmas[i], "sigma_hat": sigmas[i]})
    return x


def make_turbo_sampler():
    """KSAMPLER the node hands out; the function lives in this importable
    module so Ray can deserialize it on workers."""
    return comfy.samplers.KSAMPLER(_turbo_sampler)
