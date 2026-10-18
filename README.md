# karmic-kraken-beta-20261013.1 — vllm-kk serving bundle

Self-contained bundle for serving **Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD**
(QAD checkpoint) on karmic-kraken-beta: the pinned base image plus a
6-diff patch chain baked at build time, the compose service definition, and
the runtime files (middleware, plugins, chat template) the service
bind-mounts.

Branch name = ghcr tag of the pinned base image plus the bundle revision
suffix (upstream tags no longer carry a hash suffix). The shipped vllm
version is `0.1.dev21832+g850215bbb` (see `files/vllm-kk/patches/kk/version.txt`).

## Contents

| path | role |
|---|---|
| `docker-compose.yml` | `vllm-kk` service: build + run definition |
| `Dockerfile.vllm-kk` | applies the chain over the base, verifies per-file sha256, stamps version |
| `files/vllm-kk/patches/kk/` | chain inputs: 6 bundle dirs (`port.diff` + reference files + READMEs) + `manifest.sha256`, `version.txt`, `head.txt`, `chain.txt` |
| `files/vllm-kk/middleware/` | `vllm_defaults.py`, `vllm_middleware_strict_tool_calling.py` (mounted to `/app/middleware`) |
| `files/vllm-kk/plugins/` | `qwen_clean_reasoning.py`, `kk_profile_env.py`, `stats_logger/` (mounted to `/app/plugins`) |
| `files/templates/qwen/chat_template.jinja` | served chat template; copy into your model dir — the compose reads it at `/models/vllm/chat_template.jinja` |

External, not in this bundle: the model snapshot
(`./files/models/vllm`, offline HF cache for
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD` at revision
`dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`) and `./files/vllm-kk/cache`
(JIT caches; created on first run).

## Model: QAD checkpoint (HF/ModelOpt layout)

The checkpoint is a standard HF/ModelOpt export: 21 safetensors shards
under `tensors/`, and a `config.json` whose `quantization_config` names
`quant_method: modelopt` (`MIXED_PRECISION`). Serve the snapshot dir
directly from the HF cache mount — the custom CSF loader and the
synthesized serving dir the previous bundle shipped are gone upstream.
The routed-expert scales stay losslessly compressed in GPU memory (B12X
reads them compressed; more room for KV).

The compose passes `--quantization modelopt_mixed --load-format
instanttensor` — the stock pair the image's launcher emits for this
revision (auto-detect reaches the same compressed-scale path;
instanttensor additionally speeds up weight load). MoE backend must be
`b12x` without expert parallelism (image contract).

## Context: YaRN to 524,288

The checkpoint ships `rope_type: default` at 262,144. The compose
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

- Base: `ghcr.io/local-inference-lab/vllm@sha256:31d8ac21adb95359796d45455ba3d77a92e4b5d9455f091c1bd178e24f53370a`
  (tag `karmic-kraken-beta-20261013`), vllm source `2e7b458480` =
  integration/karmic-kraken-beta tip 2026-10-13; flashinfer 0.7.1
  `b4c35e6ec1` (B12X ships inside `flashinfer/experimental/b12x`, no
  separate wheel); lmcache_cumem `8ef537d6`.
- Chain: 6 commits `008 -> 001 -> 005 -> 009 -> 010 -> 011`, head
  `850215bbbcee1328c910911d90dee431247e2a29` (full shas + subjects in
  `chain.txt`). 009-011 are the prefill-lane livelock fix chain: live
  only when served with `--max-parallel-prefills` > 1; the default
  lanes=1 path is untouched.
- Proof: applying the 6 `port.diff`s onto the base `vllm/` tree reproduces
  the chain-head tree byte-exact; the build re-verifies every patched
  package file against `manifest.sha256` (5 files) inside the image.
- Dropped vs the 2026-10-05 release: the FP4-CSF
  serving dir (`files/vllm-kk/csf/`) and the custom CSF quantization
  loader — both removed upstream at this base; the checkpoint now serves
  directly from its snapshot with the stock `modelopt_mixed` /
  `instanttensor` pair. The 008/001/005 chain is the same 3 diffs,
  re-ported onto the new tip (clean cherry-picks); vs the 20261013
  release (3 diffs) this bundle adds the 009-011 prefill-lane livelock
  fix chain (new fix commits on the chain tip, not ports; per-patch
  lineage in the bundle-dir READMEs). The 003 pack (stale-KV tracking,
  CoW store-range alignment,
  align-Mamba external-hit serving) had merged upstream as
  local-inference-lab/vllm PRs #869 + #870 at the previous base, and the
  aligned-retention gate fix was superseded by upstream `212f46bfa8`
  (accepts `--recurrent-checkpoint-policy aligned` for connectors
  declaring `supports_aligned_hybrid_transfer` at DCP==1 —
  `OffloadingConnector` does); 002/004 had been absorbed earlier still.

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
