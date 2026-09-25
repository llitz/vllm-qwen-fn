# 003_04-aligned-policy-gate-fix (kk)

PR #864 (lil, tip `ccfef884dd`) requires Qwen model types to use a connector
that advertises `supports_request_boundary_checkpoints` whenever any KV
connector is configured. The gate keys only on model type + connector
presence, but the request-boundary rail it protects cannot engage under an
explicit `--recurrent-checkpoint-policy aligned`
(`use_request_boundary_checkpoints` requires "auto"/"request_boundaries").
Result: tip cannot boot aligned-retention offload at all — the exact mode
the 003 pack serves — on any Qwen3.8/Qwen4 config, including lil's own.

Fix: the gate skips its requirement only when
`cache_config.recurrent_checkpoint_policy == "aligned"`. "auto",
"request_boundaries", and configs lacking a cache_config keep the
requirement. When the rail can engage, the atomic-adapter contract still
applies unchanged.

- kk commit: `5942b493c3` on branch `kk-tip-20260923` (chain head, base
  `ccfef884dd`).
- Apply: `git apply --exclude='tests/*' port.diff` (format-patch of the
  commit). Files: `vllm.py` = full
  `vllm/config/vllm.py` at this commit; `test_config.py` = full
  `tests/v1/kv_connector/unit/test_config.py` at this commit (adds the
  policy-parameterized exemption matrix; the existing
  `test_qwen_qsa_requires_atomic_checkpoint_connector` passes unchanged).
- Live path: only reached when a KV connector is configured on a Qwen
  model type. Without the fix, `VllmConfig` validation raises
  "Qwen QSA KV transfer requires an atomic request-boundary checkpoint
  connector" at startup.
- Deployment pairing: compose adds `--recurrent-checkpoint-policy aligned`
  (the validated 57a-base behavior; request-boundary rail was absent
  entirely there).
- Upstream status: report as a #864 bug (gate too wide: blocks
  aligned-retention configs that never use the rail).
