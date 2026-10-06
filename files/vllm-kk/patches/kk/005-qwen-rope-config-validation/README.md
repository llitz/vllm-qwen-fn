# 005-qwen-rope-config-validation (kk)

REQUIRED for Qwen3.8-Flash-Next checkpoints whose `max_position_embeddings`
and M-RoPE metadata (`mrope_section`, `mrope_interleaved`) live only inside
`text_config`: transformers v5 rope standardization reads
`self.max_position_embeddings` during the composite config's `__post_init__`
(before undeclared kwargs are assigned) and rejects the M-RoPE keys as
unrecognized, crashing config parsing with
`'Qwen3_8FlashNextConfig' object has no attribute 'max_position_embeddings'`.

Fix: mirror the text config's context limit onto the composite config by
setting the attribute on `self` before `super().__init__`, and pass
`ignore_keys_at_rope_validation` via kwargs. On this base the qwen3_8 config
is a subclass shell over qwen4_exp, so kwargs-only placement is too late for
the v5 dataclass post-init path.

Note: the FP4-CSF QAD checkpoint in this bundle declares
`model_type: qwen4_exp` directly (not `qwen3_8_flash_next`), so this patch's
class is not instantiated for that model — it stays in the chain for
checkpoints that do use the qwen3_8 shell.

- Base: vllm `f1c2508f10` (karmic-kraken-beta tip 2026-10-05).
- Chain commit: `90d709c9b883e47a0c66000879191055868fffef` (cherry-pick of
  the kk-20260923 chain commit `d2a8d6c0a0`; absorbed one-line upstream
  drift — `supports_full_tp_dcp_with_kv_gather` moved to the qwen4_exp
  parent and is inherited).
- Apply: `git apply port.diff` (format-patch; `git am` also works).
- Files: `config.py` = full patched
  `vllm/models/qwen3_8_flash_next/config.py` (sha256
  `8884e922d721041397f822cc2363e69ea92e869321dcabf6191f16b1e654c653`,
  == chain-head blob); `test_qwen3_8_flash_next_rope_config.py` =
  `tests/models/test_qwen3_8_flash_next_rope_config.py`.
