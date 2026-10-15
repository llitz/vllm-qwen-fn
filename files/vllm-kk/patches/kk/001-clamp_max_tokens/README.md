# 001-clamp_max_tokens (kk-tip-20261013)

Opt-in per-request output-token budget clamp: a `max_tokens_budget`-style
request field caps generated tokens for clients that need a hard output
budget independent of `max_tokens` semantics. Diagnostic/convenience patch —
NOT a corruption workaround (see workspace PLAN.md standing order; this patch
never gates correctness paths).

- Lineage: tip `cf8193372e` -> kk `07d86daa71` (kk-20260923 chain) ->
  prior kk `d6c33e6528` (kk-2026-10-05 chain) -> this base commit
  `839afda71d` (clean cherry-pick onto the 20261013 tip).
- Base: vllm `2e7b458480` (karmic-kraken-beta tip, force-pushed 2026-10-14).
- Drift: none — zero commits touched `chat_completion/protocol.py` or
  `completion/protocol.py` between the 2026-10-05 and 2026-10-13 tips, and
  `git grep clamp_max_tokens` at the new tip returns no hits in `vllm/`, so
  the patch is still required and not upstream-equivalent.
- Apply: `git apply port.diff` (format-patch; `git am` also works).
- Files: `chat_protocol.py` = patched
  `vllm/entrypoints/openai/chat_completion/protocol.py`;
  `completion_protocol.py` = patched
  `vllm/entrypoints/openai/completion/protocol.py`;
  `test_clamp_max_tokens.py` = `tests/entrypoints/openai/test_clamp_max_tokens.py`.
