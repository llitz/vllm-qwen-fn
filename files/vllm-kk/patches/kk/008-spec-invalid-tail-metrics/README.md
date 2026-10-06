# 008-spec-invalid-tail-metrics (kk)

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

- Base: vllm `f1c2508f10` (karmic-kraken-beta tip 2026-10-05).
- Chain commit: `89c5b30b5277af8a9a4d6c6caa5d5a4dbcc22135` (clean
  cherry-pick of the kk-20260923 chain commit `ae9cd6b7fb`).
- Apply: `git apply port.diff` (format-patch; `git am` also works). The
  Dockerfile applies it with `--exclude='tests/*'`.
- Files: `scheduler.py` = full patched
  `vllm/v1/core/sched/scheduler.py`; `test_scheduler_full.py` = full patched
  `tests/v1/core/test_scheduler.py` (regression case
  `test_spec_decode_metrics_preserve_accepted_grammar_tail`).
- Live path: yes — serves MTP3 + structured output.
