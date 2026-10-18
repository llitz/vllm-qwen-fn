# 009-prefill-lane-release-aging (kk-tip-20261013)

Prefill-lane scheduler livelock fix at `max_parallel_prefills > 1`:
the engine spins on empty steps (GPU idle, no forward dispatched)
while never-scheduled waiting requests monopolize the prefill lanes.
Observed with OffloadingConnector + chunked prefill at lanes>1.

Mechanism: prefill-lane monopoly by never-scheduled rank-(-1) waiting
requests. Two defects:

1. Intra-step leak: a lane-selected async-load candidate whose
   `allocate_slots` returns None takes the non-interleaved FCFS branch
   of the waiting loop and breaks without `mark_unavailable`,
   stranding the unprocessed lane-holders; `_replenish` refills
   released lanes from the same rank-(-1) cohort, so the running
   chunks never hold a lane in any pass.
2. Cross-step monopoly: `PrefillInterleaveStep` is rebuilt every
   `schedule()` call, so intra-step parking cannot change the
   cross-step invariant: the cohort re-wins all lanes at rank -1 in
   `begin_step`, chunks are skipped in the preempt-capable first
   running pass, and the only post-waiting pass is
   `allow_preemption=False`. With free == 0 nothing forwards, deferred
   frees never drain, and the cycle is self-sustaining.

Fix (scheduler-only; the lanes=1 path is untouched - the
prefill-interleave step object is None for `max_parallel_prefills <= 1`):

- `vllm/v1/core/sched/scheduler.py`: a lane-selected request that
  fails allocation in the FCFS branch is parked and its lane released
  (`remove_request` + `mark_unavailable` + `step_skipped_waiting`,
  mirroring the interleaved path), so lanes rotate to the running
  chunks within the step and the loop keeps processing the remaining
  lane-holders. FCFS stop semantics are unchanged for non-lane
  requests (the ready-imports escape is preserved).
- `vllm/v1/core/sched/prefill_interleave.py`:
  `PrefillInterleaveController` tracks per-request lane-candidate
  attempts; a never-scheduled request rotates out of the lane front
  after `LANE_AGING_STEPS` (32) consecutive candidate steps - ranking
  behind recently-scheduled requests for an equal window - so chunks
  reach the preempt-capable first pass, preemption frees blocks, and
  the spin breaks within a bounded number of steps. The rotation is
  deterministic and bounded, with no new env knobs; scheduling resets
  the cycle
  (`record_scheduled`/`forget`).

- Lineage: new fix commit on branch `kk-tip-20261013`; not a port.
- Base: chain head `b7bd843725` (005-qwen-rope-config-validation);
  fix commit `cbf82b123e`.
- Apply: `git apply port.diff` (format-patch; `git am` also works).
- Files: `scheduler.py` = full patched
  `vllm/v1/core/sched/scheduler.py`; `prefill_interleave.py` = full
  patched `vllm/v1/core/sched/prefill_interleave.py`;
  `test_prefill_lane_livelock.py` = full
  `tests/v1/core/test_prefill_lane_livelock.py`.
- Verification: in-image pytest (`--noconftest`): both regression cases
  (`test_async_admission_failure_releases_lane_for_other_candidates`,
  `test_never_admissible_cohort_cannot_starve_runnable_chunk_forever`)
  pass with the patch and fail on the unpatched chain head; existing
  bundle suites pass.
- Live path: only when served with `--max-parallel-prefills` > 1.
  Production serve uses the default 1 (flag absent), where the
  prefill-interleave step object is None and this patch is
  unreachable.
