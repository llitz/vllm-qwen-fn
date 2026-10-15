# 008-spec-invalid-tail-metrics (kk-tip-20261013)

Speculative-decode metrics crash fix: with structured output active, grammar
validation can conservatively mark a trailing draft suffix invalid
(`num_invalid_spec_tokens`) while the sampler still accepts tokens within the
target-verified window. The old `num_draft_tokens = min(num_draft_tokens,
num_valid_draft_tokens)` floor then pushed the verified count below the
accepted count and `assert num_accepted <= num_draft_tokens` fired, killing
the scheduler.

Fix: compute `num_accepted` first, assert it against the hard verified-window
bound, then floor the grammar-adjusted count at `num_accepted` via
`max(num_accepted, min(...))`.

- Lineage: tip `9e6808c614` -> kk `ae9cd6b7fb` (kk-20260923 chain) ->
  prior kk `89c5b30b52` (kk-2026-10-05 chain) -> this base commit
  `cd10c0784c` (clean cherry-pick onto the 20261013 tip).
- Base: vllm `2e7b458480` (karmic-kraken-beta tip, force-pushed 2026-10-14).
- Drift: scheduler.py had 13 upstream commits since the 2026-10-05 tip (CSF
  scheduling, prefill-lane, KV-import fixes). The `min()` grammar floor was
  still applied *before* the `assert num_accepted <= num_draft_tokens` at the
  new tip (no `max(num_accepted, ...)` anywhere), so the patch is still
  required and not upstream-equivalent. The cherry-pick auto-merged the block
  into the shifted context (region moved from ~2998 to ~2965); no manual
  conflict resolution. Verified post-apply: floor
  `max(num_accepted, min(num_draft_tokens, num_valid_draft_tokens))` present
  at line 2988, assert still guards the verified-window invariant.
- Apply: `git apply port.diff` (format-patch; `git am` also works).
- Files: `scheduler.py` = full patched
  `vllm/v1/core/sched/scheduler.py`; `test_scheduler_full.py` = full patched
  `tests/v1/core/test_scheduler.py` (regression case
  `test_spec_decode_metrics_preserve_accepted_grammar_tail`).
- Live path: yes — serves MTP3 + structured output.
