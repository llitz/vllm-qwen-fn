# vllm-kk: Qwen3.8-Flash-Next on the Karmic Kraken channel

Runtime bind-mount bundle over
`ghcr.io/local-inference-lab/vllm@sha256:55e477ad62ae15a77c9b869e8fb8e2d958f6edcc4d95e306adfa89dee9ed19df`
(floating tag `karmic-kraken-beta`). No image build: the pinned image plus
these mounts is the deployment. Model weights and the image are external.

Component pins inside this image (from `/opt/venv/share/lil-runtime/manifest.json`):
vllm `integration/karmic-kraken-beta @ 57a80980bbf4b40398de7ed851b23e55a3a4c50e`,
b12x `integration/karmic-kraken-beta @ eea3ced11fc14625b683d3c575cf2d702ad3706a`,
flashinfer `2206a14e`, LMCache `688bee14`, InstantTensor `95d4729b`,
NCCL `93fe05d9` (2.31.2+lil), foundation NGC pytorch 26.08 (CUDA 13.4.1).

## Layout

```
docker-compose.yml              single vllm-kk service
files/vllm-kk/middleware/       request middleware (frequency penalty, strict tool calling)
files/vllm-kk/plugins/          qwen_clean reasoning plugin; stats_logger stat plugin
files/vllm-kk/patches/vllm/     lineage-patched modules (see below)
files/templates/qwen/           chat_template.jinja (froggeric's Qwen3.8 template)
files/models/                   model weights (external; HF cache layout)
```

## Launch

`docker-compose.yml` runs the engine directly (`lil-runtime-bootstrap` +
`vllm.entrypoints.cli.main serve`), bypassing the image's lil launcher: the
launcher blocks `--enable-cumem-allocator` / `--kv-transfer-config` /
`--kv-offloading-*` as native args, and this deployment needs them. All
profile/hardware environment the launcher would set (`qwen38-flash-next` +
`rtx-pro-6000-pcie`, captured via `--print-config` of the pinned digest) is
declared explicitly in the compose file.

Required external inputs:
- `LLAMAKEY` in the environment or `.env` (API key).
- Model weights at `./files/models/vllm` as an HF cache layout:
  `models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4[-qad]` (resolved
  offline via `HF_HUB_OFFLINE=1`). Adjust `HF_HUB_CACHE` to your mount.

The chat template is bundled at `files/templates/qwen/chat_template.jinja`
(md5 `5c928b42a335c295c31789e97713c778`, froggeric's Qwen3.8 template) and
mounted at `/app/chat_template.jinja`.

## Lineage patches (bind-mounted over site-packages)

Source: `local-inference-lab/vllm` fork, branch `kk-serve-20260920`
(base `57a80980bb` = this image's vllm pin).

| File under `files/vllm-kk/patches/vllm/` | Patch | Why |
|---|---|---|
| `v1/core/sched/scheduler.py` | 008 | MTP + structured output: grammar-invalid tail floored below accepted count crashed the scheduler assert |
| `v1/simple_kv_offload/manager.py` | 004 | Simple CPU offload must skip non-prefix-cacheable QSA key-ring groups |
| `config/speculative.py` | 002 | MTP draft `ModelConfig` inherits `enable_cumem_allocator` from the target |
| `entrypoints/openai/chat_completion/protocol.py` | 001 | opt-in `clamp_max_tokens` (per-request output-budget clamping) |
| `entrypoints/openai/completion/protocol.py` | 001 | same, completion side |
| `distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` | 003 | upstream PR #46960 rebased head; inert while `SimpleCPUOffloadConnector` is active, kept for the complex connector |
| `models/qwen3_8_flash_next/config.py` | 005 | composite config mirrors text `max_position_embeddings`; rope validation ignores `mrope_section`/`mrope_interleaved` — REQUIRED, startup crashes without it on this checkpoint |

006/007 (PLE disk prefetch) and 009/010 (GDN shared staging) are NOT ported:
this branch removed the per-layer GDN staging upstream (`e4f9b816f5`) and has
no PLE hosted-prefetch surface. PLE runs mapped-host
(`VLLM_PLE_CPU_OFFLOAD=1`).

## KV sizing

`--kv-cache-memory-bytes 15490411776` (14.4 GiB) + `--max-model-len 524288`;
no `--gpu-memory-utilization`. Verify at startup: the log line
`Setting attention block size to N tokens` must show N dividing the retention
interval (3008 on RTX PRO 6000 + b12x; other backends measure differently,
e.g. 1600 on GB10 + flashinfer — set
`--prefix-cache-retention-interval` to the observed N).

## b12x JIT cache

`/cache` is a bind mount; the b12x compile cache keys embed the physical
GPU UUID, so caches are NOT portable between machines — expect one full
compile pass per host (several minutes), then warm restarts.

## Removal guide

- Drop a patch: remove its volume line from `docker-compose.yml`.
- 001 is optional (client opt-in). 002/004 required with the KV-offload +
  cumem block. 003 inert unless `VLLM_USE_SIMPLE_KV_OFFLOAD=0`. 005 required.
- Drop KV offload: remove `--kv-offloading-size`, `--kv-transfer-config`,
  `VLLM_USE_SIMPLE_KV_OFFLOAD`, then the 004/003 mounts.
