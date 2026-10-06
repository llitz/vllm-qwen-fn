# karmic-kraken-beta-20261005-4102660abd8a98b1 — vllm-kk serving bundle

Self-contained bundle for serving **Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD**
(FP4-CSF QAD checkpoint) on karmic-kraken-beta: the pinned base image plus a
3-diff patch chain baked at build time, the compose service definition, the
FP4-CSF serving dir, and the runtime files (middleware, plugins, chat
template) the service bind-mounts.

Branch name = ghcr tag of the pinned base image. The shipped vllm version is
`0.1.dev22052+g90d709c9b` (see `files/vllm-kk/patches/kk/version.txt`).

## Contents

| path | role |
|---|---|
| `docker-compose.yml` | `vllm-kk` service: build + run definition |
| `Dockerfile.vllm-kk` | applies the chain over the base, verifies per-file sha256, stamps version |
| `files/vllm-kk/patches/kk/` | chain inputs: 3 bundle dirs (`port.diff` + reference files + READMEs) + `manifest.sha256`, `version.txt`, `head.txt`, `chain.txt` |
| `files/vllm-kk/csf/` | FP4-CSF serving dir (metadata + rewritten `config.json`; weights stay in the HF snapshot via `checkpoint_root`) |
| `files/vllm-kk/middleware/` | `vllm_defaults.py`, `vllm_middleware_strict_tool_calling.py` (mounted to `/app/middleware`) |
| `files/vllm-kk/plugins/` | `qwen_clean_reasoning.py`, `kk_profile_env.py`, `stats_logger/` (mounted to `/app/plugins`) |
| `files/templates/qwen/chat_template.jinja` | served chat template; copy into your model dir — the compose reads it at `/models/vllm/chat_template.jinja` |

External, not in this bundle: the model snapshot
(`./files/models/vllm`, offline HF cache for
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD` at revision
`ac1c8d8edfb88a41df837e8ca431e0ec237c139d`) and `./files/vllm-kk/cache`
(JIT caches; created on first run).

## Model: FP4-CSF QAD checkpoint

The checkpoint stores routed-expert scales losslessly compressed; B12X reads
them compressed and they stay compressed in GPU memory (more room for KV).
vLLM cannot open the raw CSF repo layout (`metadata/` + `tensors/` +
`manifest.json`) directly, so the image's launcher synthesizes a serving dir
(metadata files + a `config.json` whose `quantization_config` names the CSF
reader and the checkpoint root). This bundle ships that serving dir under
`files/vllm-kk/csf/`, mounted at `/models/csf`; `checkpoint_root` assumes the
snapshot sits at `/models/vllm/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD/snapshots/ac1c8d8.../`
(i.e. `./files/models/vllm/...` with the default compose mount). The dir
name embeds `sha256(checkpoint_root)[:12]` — if you change the mount layout,
regenerate it with the image's own
`runtime.launcher.csf_serving_config` (same steps as
`prepare_csf_checkpoint`).

`--quantization` and `--load-format` must both be `nvfp4_csf`; MoE backend
must be `b12x` without expert parallelism (image contract).

## Context: YaRN to 524,288

The CSF checkpoint ships `rope_type: default` at 262,144. The compose
extends it to 524,288 with `--hf-overrides`: the yarn block (factor 2.0,
`original_max_position_embeddings: 262144`) plus `max_position_embeddings:
524288` — vLLM's YaRN convention is that the config carries the SCALED
limit, and the MTP draft derives its own `max_model_len` (hence the b12x
QSA binding limit) from the draft config. Omitting the scaled limit fails
CUDA-graph capture with `QSA sequence length 524288 exceeds the configured
limit 262144`.

## KV sizing

`--kv-cache-memory-bytes 17000000000` (15.83 GiB) measured **1,106,972 KV
tokens / 2.11x concurrency @ 524,288** on a 96 GB RTX PRO 6000 with this
base + checkpoint. 17.3G boots but dies on the first real forward (CUDA
OOM — profiling does not cover eager activation spikes); 15.87G measures
1,033,444 tokens. Re-measure the `GPU KV cache size` boot line after any
base bump or checkpoint swap.

## Provenance

- Base: `ghcr.io/local-inference-lab/vllm@sha256:fd74019fc217a541060093d18427c36330df6ac5700651b55593dafdbb936920`
  (tag `karmic-kraken-beta-20261005-4102660abd8a98b1`; the tag suffix is the
  lil assembly sha256, not a commit), vllm source `f1c2508f10` =
  integration/karmic-kraken-beta tip 2026-10-05, b12x `52640cb15d`.
- Chain: 3 commits `008 -> 001 -> 005`, head
  `90d709c9b883e47a0c66000879191055868fffef` (full shas + subjects in
  `chain.txt`).
- Proof: applying the 3 `port.diff`s onto the base `vllm/` tree reproduces
  the chain-head tree byte-exact; the build re-verifies every patched
  package file against `manifest.sha256` (4 files) inside the image.
- Dropped vs the previous bundle (`karmic-kraken-beta-20260923-23a6d60a4b49c403`):
  the 003 pack (stale-KV tracking, CoW store-range alignment, align-Mamba
  external-hit serving) merged upstream as local-inference-lab/vllm PRs
  #869 + #870 with identical patch-ids, and the aligned-retention gate fix
  was superseded by upstream `212f46bfa8` (accepts
  `--recurrent-checkpoint-policy aligned` for connectors declaring
  `supports_aligned_hybrid_transfer` at DCP==1 — `OffloadingConnector`
  does). 002/004 had already been absorbed upstream at the previous tip.

## Launch notes

- The image's lil launcher (`lil-entrypoint --profile qwen38-flash-next`)
  refuses the OffloadingConnector flags by policy, so this bundle runs
  `vllm serve` directly but still consumes the launcher's resolved
  environment via `plugins/kk_profile_env.py` (`eval` in the command). The
  JIT caches land under `/cache/jit/<runtime-lock-sha>/<profile-id>/` and
  self-invalidate per base build.
- First boot on a fresh cache compiles the b12x kernel set (GDN prefill
  autotune ~7 min); subsequent boots hit the on-disk caches.
- CPU tier: `--kv-offloading-size 35.6` (GiB) needs `/dev/shm` headroom
  (`shm_size: 36g` above).

## Startup checks

1. `Setting attention block size to 3008 tokens` (must divide the retention
   interval).
2. `GPU KV cache size: ... tokens` (compare against the numbers above).
3. `Creating v1 connector with name: OffloadingConnector` + mmap of ~38 GB.
4. `Application startup complete` and `GET /health` 200.
5. Spec decode on b12x: `Using 'B12X' NvFp4 MoE backend ...` and
   `Using b12x CuTeDSL GDN prefill kernel`.
