# 011-lane-share-align-quantum (kk-tip-20261013)

Prefill-lane scheduler livelock fix, third defect class: a
mamba-align x lane-fanout deadlock at `max_parallel_prefills > 1` in
which no chunk can ever advance.

Mechanism: in mamba "align" cache mode reusable SSM states materialize
at block boundaries, so `_mamba_block_aligned_split` floors a prefill
chunk that ends before the next block boundary to zero tokens. The
interleave token-budget fanout split gives every lane-selected chunk a
share `ceil(step_budget / fanout)`; with more lane candidates than
lanes (e.g. 3 chunks + 2 doomed async-reload candidates sharing
4 lanes) every share falls below the align quantum. Each chunk is
then zeroed at the
(`mark_unavailable`) before the allocator is reached: the no-progress
preemption fallback added by 010 lives inside the allocator-failure
path and is unreachable, while the doomed async candidates fail the
full-sequence admission gate every step and re-win their lanes on the
per-step rebuild. Self-sustaining empty steps.

Fix (scheduler-only, ~3 lines at the chunk share cap; lanes=1 path
untouched):

- `vllm/v1/core/sched/scheduler.py`: when a lane-selected prefill
  chunk's fanout share falls below the recurrent-state align quantum
  (`cache_config.block_size`, the same value the aligned split floors
  to) and the global step budget can cover a full quantum, drop the
  share cap and let the ordinary token-budget clamp apportion the
  capacity. Work-conserving: a lane held by a candidate that cannot
  run (doomed async reload, sub-quantum sibling) no longer taxes a
  runnable chunk; the first chunk takes whole quanta, later chunks
  get the remainder. Invariant commented at the call site.
- Why lanes=1 is immune / stays bit-identical: the fanout share cap
  exists only inside the `prefill_interleave_step is not None`
  branch, and that step object is None for `max_parallel_prefills <= 1`
  (single lane => fanout is always 1 => share == full step budget).
  The relaxation additionally fires only when
  `need_mamba_block_aligned_split` is set, so non-align deployments
  keep fractional lane shares unchanged.
- Known limitation: the `has_pending_push_work()` quiescence predicate
  is wrong-sided for tiering managers (it can report quiescent while
  tier work is pending); moot for the CPU-offload deployment, where
  the tier manager reports no pending work by construction.

- Lineage: new fix commit on branch `kk-tip-20261013`; not a port.
- Base: chain head `88a9218411` (010-sched-second-pass-preempt);
  fix commit `850215bbbc`.
- Apply: `git apply port.diff` (format-patch; `git am` also works).
- Files: `scheduler.py` = full patched
  `vllm/v1/core/sched/scheduler.py`; `test_prefill_lane_livelock.py` =
  full `tests/v1/core/test_prefill_lane_livelock.py`.
- Verification: in-image pytest (`--noconftest`, overlaid files):
  the new case
  (`test_lane_share_below_align_quantum_cannot_starve_chunk`) passes
  with the fix and fails on the parent `88a9218411` with the
  starvation message; full livelock suite 5 passed; existing bundle
  suites (`test_scheduler.py` incl. the 008 case,
  `test_clamp_max_tokens.py`, `test_qwen3_8_flash_next_rope_config.py`)
  = 215 passed (whole run).
- Live path: only when served with `--max-parallel-prefills` > 1.
  Production serve otherwise uses the default 1 (flag absent),
  where the prefill-interleave step object is
  None and this patch is unreachable.
