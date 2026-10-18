# 010-sched-second-pass-preempt (kk-tip-20261013)

Prefill-lane scheduler livelock fix, second defect class: the engine
spins on empty steps (GPU idle, no forward dispatched) at
`max_parallel_prefills > 1` when the lane-monopoly cohort is requests
scheduled once and then preempted.

Mechanism: 009's lane aging is a no-op for this cohort by
construction: `_lane_rank` consults the attempt counter only when
`last_scheduled_step` is None, and `_preempt_request` never clears
`last_scheduled_step`. A request preempted after being scheduled keeps
its rank forever and is invisible to aging. The pair therefore wins
every lane at `begin_step`; the running chunks are skipped in the
preempt-capable first running pass and receive lanes only for the
post-waiting interleave pass, which is `allow_preemption=False`. Their
`allocate_slots` fails at free == 0, the pass breaks, and the
empty-step spin is self-sustaining: no forward, no deferred-free
drain, no recovery. Preemption is the only block-freeer and is gated
behind lane reachability.

Fix (scheduler-only; lanes=1 path untouched, no new env knobs):

- `vllm/v1/core/sched/scheduler.py`: the post-waiting interleave pass
  becomes preempt-capable when the step produced nothing
  (`allow_preemption=not num_scheduled_tokens`; in-tree precedents:
  the boundary-logits release and the decode-refill fallback use the
  same expression). Preemption is allowed only while the connector is
  quiescent: preempting a request with in-flight connector jobs
  routes the store/load flush into an unbounded worker-side `wait()`
  (a hard hang). `has_pending_push_work()` is False exactly when the
  scheduler-side connector holds no transfer jobs and the tier
  manager reports no pending work. The invariant is documented at the
  call site.
- `vllm/v1/core/sched/prefill_interleave.py`: `_lane_rank` ages the
  scheduled cohort: a candidate whose `last_scheduled_step` is more
  than `LANE_AGING_STEPS` (32) steps behind the current step ranks
  stale (`current_step`), rotating behind every candidate scheduled
  more recently, regardless of the rank being set. The ordering is
  deterministic and bounded: ties among stale candidates break on
  arrival time/request id in the caller's sort; any request that
  progresses refreshes its
  rank via `record_scheduled`. This is the recovery path while the
  connector is non-quiescent and the second-pass preemption is
  guarded.

The never-admissible-cohort regression case holds its connector
non-quiescent so the guarded second pass cannot preempt: aging remains
that test's isolated recovery mechanism.

- Lineage: new fix commit on branch `kk-tip-20261013`; not a port.
- Base: chain head `cbf82b123e` (009-prefill-lane-release-aging);
  fix commit `88a9218411`.
- Apply: `git apply port.diff` (format-patch; `git am` also works).
- Files: `scheduler.py` = full patched
  `vllm/v1/core/sched/scheduler.py`; `prefill_interleave.py` = full
  patched `vllm/v1/core/sched/prefill_interleave.py`;
  `test_prefill_lane_livelock.py` = full
  `tests/v1/core/test_prefill_lane_livelock.py`.
- Verification: in-image pytest (`--noconftest`, overlaid files):
  the two new cases
  (`test_preempted_after_scheduled_cohort_cannot_starve_runnable_chunks`,
  `test_busy_connector_guards_preemption_and_aging_recovers`) pass
  with the fix and fail on the parent `cbf82b123e` with the
  starvation messages; existing bundle suites pass.
- Live path: only when served with `--max-parallel-prefills` > 1.
  Production serve uses the default 1 (flag absent), where the
  prefill-interleave step object is None and this patch is
  unreachable.
