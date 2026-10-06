# 001-clamp_max_tokens (kk)

Opt-in per-request output-token budget clamp: a request-level field caps
generated tokens for clients that need a hard output budget independent of
`max_tokens` semantics. Convenience patch — it never gates correctness
paths.

- Base: vllm `f1c2508f10` (karmic-kraken-beta tip 2026-10-05).
- Chain commit: `d6c33e6528e05d235a7a731bda516ecd025115e7` (clean
  cherry-pick of the kk-20260923 chain commit `07d86daa71`).
- Apply: `git apply port.diff` (format-patch; `git am` also works). The
  Dockerfile applies it with `--exclude='tests/*'`.
- Files: `chat_protocol.py` = patched
  `vllm/entrypoints/openai/chat_completion/protocol.py`;
  `completion_protocol.py` = patched
  `vllm/entrypoints/openai/completion/protocol.py`;
  `test_clamp_max_tokens.py` = `tests/entrypoints/openai/test_clamp_max_tokens.py`.
