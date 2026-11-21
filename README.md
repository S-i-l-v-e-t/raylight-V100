# Raylight (V100 fork)

Raylight runs the sampler on Ray workers and splits a single generation across GPUs with
XDiT-XFuser (sequence and CFG parallelism) and FSDP. This fork is based on upstream 1.11.0 and
adds what Tesla V100 (SM70 / Volta) needs to run it. The node set and the way you build a
workflow are the same as upstream; the design notes are in `docs/`.

It is meant to be used with the matching ComfyUI fork,
[ComfyUI-V100](https://github.com/S-i-l-v-e-t/ComfyUI-V100), which brings
[softbf16](https://github.com/S-i-l-v-e-t/Soft-BF16-V100): that is what makes bf16 models and
bf16 attention work on this card.

Verified on 2× Tesla V100 32GB, torch 2.9.1+cu128, CUDA 12.8.

## Branches

Only use the files on the `v100` branch, to avoid running into unknown issues.

## Installing

Clone into `custom_nodes` under the directory name `raylight`:

```bash
cd ComfyUI/custom_nodes
git clone -b v100 https://github.com/S-i-l-v-e-t/raylight-V100 raylight
```

Install the dependencies with the same Python environment ComfyUI uses (here a uv-managed venv,
which has no pip):

```bash
uv pip install --python ../.venv/bin/python -r raylight/requirements.txt
```

Upstream also suggests `nvidia-nccl-cu12==2.28.9` when running FSDP with fp8 models on Nvidia
GPUs, and FlashAttention 2 or another attention backend. Both are optional; the fork runs
without them.

Restart ComfyUI afterwards.

## Using it

A distributed graph replaces the plain loaders and sampler. The nodes that matter:

| Node | What it does |
| --- | --- |
| Ray Init Actor (Advanced) | Starts the Ray workers; go here for worker count and VRAM budget |
| Load Diffusion Model (Ray) | Loads the diffusion model onto the workers |
| Load VAE (Ray) | Same for the VAE |
| XFuser KSampler (Advanced) | The sampler with parallelism enabled |
| Distributed VAE (Ray) | Decodes across workers |
| Kill Ray | Tears the workers down |

After updating raylight or changing its code, use Kill Ray and restart ComfyUI. The workers run
a packaged copy of the module, so reloading nodes alone does not pick up the change.

## On V100

bf16 models run as they are; the compute goes through softbf16 instead of being converted to
fp16. fp8 and int8 quantized weights work with FSDP as well.

`Load Diffusion Model (Ray)` also offers `fp8e4m3fn_fp32` as the weight dtype. It dequantizes
the fp8 weights per layer and runs the GEMMs in fp32, which is meant for checking output against
a fp16 run, not for daily use: it is roughly twice as slow per step.

MiniMax H3 and MiniMax H3 Image both run under sequence parallelism. The image model denoises a
single latent frame with no audio stream.

What has been run end to end on this hardware: MiniMax H3 I2V with FSDP, and Flux Kontext with
sequence parallelism.

## Documentation

- `docs/1-intro.md` — architecture
- `docs/2-fsdp.md` — FSDP
- `docs/3-usp.md` — sequence parallelism
- `docs/4-cfg.md` — CFG parallelism
- `docs/5-expansion.md` — expansion modules
- Upstream project: https://github.com/komikndr/raylight

## License

Apache-2.0, see `LICENSE`.
