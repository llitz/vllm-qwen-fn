# 008-spec-invalid-tail-metrics

R38 fix for a fatal speculative-decoding metrics assertion when structured
output invalidates a trailing draft suffix.

R38 commit `5b0fe021de` changed speculative metrics to count only drafts
considered valid by scheduler-side grammar validation and added
`num_accepted <= num_draft_tokens` assertions. With asynchronous speculative
decoding, target verification already runs against worker-resident draft IDs.
The scheduler later replaces a grammar-invalid suffix with `-1` placeholders
for bitmask construction and metrics. If the grammar terminates inside that
window, later positions can be unconstrained and may already have been accepted
by the target. The sampled acceptance count can therefore exceed the
post-validation metric count even though it never exceeds the scheduled or
actually verified window.

The fix keeps the hard invariant that accepted tokens cannot exceed the target-
verified draft count. It then floors the grammar-adjusted metrics denominator at
the accepted count. Rejection rollback continues to use the full scheduled
window, so request state behavior is unchanged.

Files:

- `scheduler.py` — full R38 patched file (baked into the derived image over
  `vllm/v1/core/sched/scheduler.py`)
- `spec_invalid_tail_metrics.diff` — the R38 hunks, `-p1` layout

Upstream status: not reported; regression introduced by R38 commit
`5b0fe021de`.
