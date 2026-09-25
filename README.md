# karmic-kraken-beta-20260923-23a6d60a4b49c403 — vllm-kk serving bundle

Self-contained bundle for serving **Qwen3.8-Flash-Next-NVFP4** on
karmic-kraken-beta: the pinned base image plus a 7-diff patch chain baked at
build time, the compose service definition, and the runtime files
(middleware, plugins, chat template) the service bind-mounts.

Branch name = ghcr tag of the pinned base image. The shipped vllm version is
`0.1.dev21739+g5942b493c` (see `files/vllm-kk/patches/kk/version.txt`).

## Contents

| path | role |
|---|---|
| `docker-compose.yml` | `vllm-kk` service: build + run definition |
| `Dockerfile.vllm-kk` | applies the chain over the base, verifies per-file sha256, stamps version |
| `files/vllm-kk/patches/kk/` | chain inputs: 7 bundle dirs (`port.diff` + reference files + READMEs) + `manifest.sha256`, `version.txt`, `head.txt`, `chain.txt` |
| `files/vllm-kk/middleware/` | `vllm_defaults.py`, `vllm_middleware_strict_tool_calling.py` (mounted to `/app/middleware`) |
| `files/vllm-kk/plugins/` | `qwen_clean_reasoning.py`, `stats_logger/` (mounted to `/app/plugins`) |
| `files/templates/qwen/chat_template.jinja` | served chat template (`qwen3.8-froggeric-v22.5`, locally tuned: proper `high` thinking level, default effort `xhigh`); copy into your model dir — the compose reads it at `/models/vllm/chat_template.jinja` |

External, not in this bundle: the model snapshot (`./files/models/vllm`,
offline HF cache for `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`) and
`./files/vllm-kk/cache` (JIT caches; created on first run).

## Provenance

- Base: `ghcr.io/local-inference-lab/vllm@sha256:445c58f040c7804e2bf795c3fe6250c1d13a48cd6ea6855e526f280906f8302e`
  (tag `karmic-kraken-beta-20260923-23a6d60a4b49c403`), vllm source
  `ccfef884dd` = integration/karmic-kraken-beta tip 2026-09-23.
- Chain: branch `kk-tip-20260923`, head `5942b493c3eb44ad3bb30e6d8b2b1e4dd4d9b09a`,
  7 commits in order `008 -> 001 -> 003_01_PR46960 -> 005 -> 003_02 ->
  003_03 -> 003_04` (full shas + subjects in `chain.txt`).
- Proof: applying the 7 `port.diff`s onto the base `vllm/` tree reproduces
  the chain-head tree byte-exact; the build re-verifies every patched
  package file against `manifest.sha256` (18 files) inside the image.
- Upstream pairing: 003_01/003_02/003_03 + 003_04 correspond to
  local-inference-lab/vllm PR #867 (content accepted into the beta line;
  003_04 later extended upstream into a
  `supports_aligned_hybrid_transfer()` connector capability hook).
- 001 (per-request `max_tokens` clamp) and 005 (YaRN/M-RoPE config
  validation for qwen3_8_flash_next) are local-only patches, not submitted
  upstream.

## Patch chain

| bundle | what |
|---|---|
| `008-spec-invalid-tail-metrics` | keep spec-decode acceptance metrics internally valid when grammar validation marks a trailing suffix invalid |
| `001-clamp_max_tokens` | per-request body field clamping output reservation to context headroom (local-only) |
| `003_01_PR46960-stale-kv-alloc` | offloading scheduler must not allocate from stale KV state (upstream PR 46960 port) |
| `005-qwen-rope-config-validation` | accept M-RoPE metadata keys in rope validation; mirror text-config context limit onto the composite (local-only) |
| `003_02-offload-cow-zip-align` | CoW zip alignment for offloaded hybrid state blocks |
| `003_03-mamba-offload-serve-fix` | serve external prefix hits for align-mode Mamba hybrids (boundary captures, padded QSA page refs) |
| `003_04-aligned-policy-gate-fix` | exempt explicit `--recurrent-checkpoint-policy aligned` from the atomic request-boundary connector requirement |

## Build & run

From the compose project dir (this bundle's layout is the build context):

```
docker compose build vllm-kk     # or: podman compose build vllm-kk
docker compose up -d vllm-kk
```

Requirements:
- NVIDIA GPU, single card, full memory (Blackwell-class; RTX PRO 6000
  validated). `devices: nvidia.com/gpu=all`.
- `LLAMAKEY` env var (API key) — passed through, never stored here.
- Model snapshot at `./files/models/vllm` (offline; `HF_HUB_OFFLINE=1`).
- Chat template copied to `./files/models/vllm/chat_template.jinja`.
- On podman rootless: `group_add` + `x-podman.gidmaps` for the nvidia
  device groups, and host networking or a compose-created network with
  explicit `ports:` (see your local podman config).

Build-time verification (all fail the build loudly):
1. base guard — pinned file hash of `offloading/scheduler.py` must match;
2. each `port.diff` applies individually (failure attributes to one diff);
3. `sha256sum -c manifest.sha256` over the patched site-packages tree;
4. import smoke of the patched scheduler/connector modules.

Post-start verification:
- log shows `v0.1.dev21739+g5942b493c`;
- `Setting attention block size to 3008 tokens` (must divide the retention
  interval);
- no atomic-checkpoint gate raise (aligned policy + 003_04);
- `GET :8000/health` -> 200; cold start with a cold JIT cache can exceed
  10 minutes (b12x gdn_prefill autotune) — the healthcheck
  `start_period: 10m` accounts for it.

## Notes

- `--recurrent-checkpoint-policy aligned` is mandatory with this image:
  the default (`auto`) still trips PR #864's atomic-connector gate on Qwen
  model types.
- `--max-num-seqs 16 --max-cudagraph-capture-size 64` bounds CUDA graph
  capture cost on the workstation Blackwell; `--kv-cache-memory-bytes
  15490411776` pins the KV pool against the offload sizing.
- Known non-fatal log noise: triton `matmul_ogs` import errors (mxfp4
  fallback, base-image issue).
