# vllm-fn: Qwen3.8-Flash-Next serving setup

Derived image (`Dockerfile.vllm-fn`, sha256-pinned apply) plus runtime
bind-mounts over `localinferencelab/vllm:jovian-judgement-community-20260914-r38`.
`docker-compose.yml` is the single vllm-fn service. Model weights and the base
image are external. Patches 001-005 + 008 are **baked** by the Dockerfile
(remove = edit Dockerfile `RUN` list + `COPY` inputs, rebuild); 006/007 are
**bind-mounts** (remove = drop volume lines + env vars).

## Patches

- **001-clamp_max_tokens** (baked) — optional, not required to serve. Adds
  per-request `clamp_max_tokens`: clamps output length to context headroom
  instead of rejecting over-long requests. Default off; clients must pass the
  argument. No compose variables. Remove: drop the two 001 entries from the
  Dockerfile sha256 list + copy list.
- **002-draft-cumem-allocator** (baked) — required while both
  `--enable-cumem-allocator` and `--speculative-config '{"method":"mtp",...}'`
  are in the command: MTP draft config inherits `enable_cumem_allocator` from
  the target model. Remove only if you drop `--enable-cumem-allocator`.
- **003-stale-kv-alloc** (baked) — **inert here; beyond optional, remove
  freely.** Targets the newer `OffloadingConnector` (unfinished). This stack
  must use SimpleCPUOffload (see `VLLM_USE_SIMPLE_KV_OFFLOAD=1` +
  `kv_connector":"SimpleCPUOffloadConnector"` below), so the patched file is
  never loaded. Remove: drop the 003 sha256/copy entries from the Dockerfile.
- **004-qsa-ring-offload-skip** (baked) — required together with KV offload:
  makes `SimpleCPUOffloadScheduler` skip non-prefix-cacheable QSA key-ring
  groups. Only needed if you keep the KV-offload block below.
- **005-qwen-rope-config-validation** (baked) — required: rope validation
  ignores `mrope_section` / `mrope_interleaved`; also enables the yarn-rotary
  `config.json` route for long context.
- **006-ple-resident-scales** + **007-ple-prefetch** (bind-mounts) — optional;
  only for disk streaming of the PLE table, still under testing. Active only
  with `VLLM_PLE_TABLE_MEMORY=disk` (this repo ships `ram`); gated by
  `B12X_PLE_RESIDENT_SCALES` and `VLLM_PLE_DISK_PREFETCH` — both inert in
  `ram` mode, shipped as `0`. Remove: the 5 volume lines under the `Fix A` /
  `Fix B` comments + the two env vars + patch dirs.
- **008-spec-invalid-tail-metrics** (baked) — required while
  `--speculative-config` (MTP) is active and structured output is used:
  prevents a fatal metrics assertion when grammar validation invalidates a
  trailing draft suffix.

## Optional: CPU KV offload

Caches prefix KV in pinned host RAM (`--kv-offloading-size 35.6`) for
cross-request reuse on long contexts. Everything to remove:

- env `VLLM_USE_SIMPLE_KV_OFFLOAD=1`
- command `--kv-offloading-size 35.6` and
  `--kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector",...}'`
- patches 003 + 004 (their only consumer)
- `shm_size: 36g` is **not** part of this: CPU KV tensors are pinned via
  `cudaHostRegister`, not `/dev/shm` (runtime usage measured ~4 KB). It can be
  shrunk or dropped regardless of KV offload.

Without KV offload, long-context prefix reuse re-runs prefill on GPU instead
of re-fetching from host RAM.

## Optional middleware and plugins

`files/vllm/middleware` (`vllm_defaults.FrequencyPenaltyMiddleware`,
`vllm_middleware_strict_tool_calling.StrictToolCallingMiddleware`) and
`files/vllm/plugins` (`qwen_clean_reasoning.py`, `stats_logger`) are optional.
Remove: their two volume lines, the two `--middleware` args,
`--reasoning-parser qwen3_clean` + `--reasoning-parser-plugin
/app/plugins/qwen_clean_reasoning.py`, and the `/app/middleware:/app/plugins`
entries in the `PYTHONPATH` env.
