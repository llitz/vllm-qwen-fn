# 005-qwen-rope-config-validation (kk-tip-20261013)

REQUIRED for the Qwen3.8-Flash-Next-NVFP4 checkpoint: config parsing crashes
without it. The checkpoint carries `max_position_embeddings` and the M-RoPE
metadata (`mrope_section`, `mrope_interleaved`) only inside `text_config`;
transformers v5 rope standardization reads `self.max_position_embeddings`
during `__post_init__` (before undeclared kwargs are assigned) and rejects
the M-RoPE keys as unrecognized.

Fix: mirror the text config's context limit onto the composite config by
setting the attribute on `self` before `super().__init__`, and pass
`ignore_keys_at_rope_validation` via kwargs. On kk the qwen3_8 config is a
subclass shell over qwen4_exp, so kwargs-only placement is too late for the
v5 dataclass post-init path.

- Lineage: tip `1d15136507` -> kk `d2a8d6c0a0` (kk-20260923 chain, 3-way
  ported) -> prior kk `90d709c9b8` (kk-2026-10-05 chain) -> this base commit
  `b7bd843725` (clean cherry-pick onto the 20261013 tip; chain head).
- Base: vllm `2e7b458480` (karmic-kraken-beta tip, force-pushed 2026-10-14).
- Not upstream-equivalent: `qwen3_8_flash_next/config.py` at the new tip is
  still the bare subclass shell — no `max_position_embeddings` mirror, no
  `ignore_keys_at_rope_validation`.
- Drift absorbed: the `qwen4_exp/` parent had ~10 upstream commits since the
  2026-10-05 tip (PLE shards, CSF-QAD, FP4-CSF dtype, host-table sharing).
  Checked the parent `__init__` signatures: both `Qwen4ExpTextConfig` (line
  89) and `Qwen4ExpConfig` (line 261) still accept `**kwargs: Any` and forward
  to super, so the patch's kwargs passthrough
  (`ignore_keys_at_rope_validation`, `max_position_embeddings`) and the
  pre-super attribute set remain valid. No re-anchoring needed; the patch
  touches only the qwen3_8 shell. Verified: patch applies, `py_compile`
  clean, no conflict markers, full-tree chain proof byte-exact.
- Apply: `git apply port.diff` (format-patch; `git am` also works).
- Files: `config.py` = full patched
  `vllm/models/qwen3_8_flash_next/config.py` (sha256
  `8884e922d721041397f822cc2363e69ea92e869321dcabf6191f16b1e654c653`,
  == HEAD blob; byte-identical to the previous (2026-10-05) bundle — the shell
  content did not change across the base bump);
  `test_qwen3_8_flash_next_rope_config.py` =
  `tests/models/test_qwen3_8_flash_next_rope_config.py`.
