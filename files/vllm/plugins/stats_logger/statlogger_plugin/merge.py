"""Pure-Python accumulation logic for the prefill-last stat-logger plugin.

This module has NO vllm (or prometheus_client) imports so it can be
unit-tested on a host where vLLM is not installed.

Domain context: vLLM's ``vllm:prompt_tokens`` counter credits an entire
prefill at request completion, so a single long prompt produces a 1-second
spike in a 1-second /metrics poller (e.g. 95k prompt tokens/s). The
accumulator below groups the finished requests of the "last prefill batch"
and exposes the summed token count, the summed per-request prefill
durations ("work-seconds"), and the average COMPUTED (non-cached)
prompt tokens/s: (tokens - cached_tokens) / window_seconds.
A 1-second poller can then reconcile: spike tokens == gauge tokens;
the cache-inclusive rate is derivable as tokens / window_seconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

# One finished request: (num_prompt_tokens, prefill_time, num_cached_tokens)
Entry = Tuple[int, float, int]

__all__ = [
    "PrefillBatchAccumulator",
    "PrefillBatchSnapshot",
    "compute_avg_pps",
]


@dataclass(frozen=True)
class PrefillBatchSnapshot:
    """Immutable state of the current ("last") prefill batch."""

    tokens: int
    """Sum of num_prompt_tokens over the requests in this batch."""

    window_seconds: float
    """Sum of per-request prefill_time. This is summed *work-seconds*
    (how long each prefill took on the GPU), NOT wall-clock time: with
    concurrent or data-parallel prefills the work-seconds can exceed the
    wall-clock span of the batch."""

    cached_tokens: int
    """Sum of num_cached_tokens (local + external prefix-cache hits) over
    the requests in this batch."""

    avg_pps: float
    """Average COMPUTED (non-cached) prefill prompt tokens/s:
    (tokens - cached_tokens) / window_seconds (0.0 when
    window_seconds <= 0). GPU work only; cached/skipped tokens are
    excluded."""

    batch_size: int
    """Number of finished requests folded into this batch."""


def compute_avg_pps(
    tokens: float, window_seconds: float, cached_tokens: float = 0
) -> float:
    """(tokens - cached_tokens) / window_seconds, divide-by-zero guarded.

    The computed (non-cached) prompt-token rate: GPU work only, cached
    tokens excluded. Returns 0.0 for any non-numeric or non-finite
    input, a non-positive window, or a negative computed rate (cached
    can exceed tokens across merged entries).
    """
    try:
        tokens = float(tokens)
        window_seconds = float(window_seconds)
        cached_tokens = float(cached_tokens)
    except (TypeError, ValueError):
        return 0.0
    if not (
        math.isfinite(tokens)
        and math.isfinite(window_seconds)
        and math.isfinite(cached_tokens)
    ):
        return 0.0
    if window_seconds <= 0.0:
        return 0.0
    return max(0.0, (tokens - cached_tokens) / window_seconds)


class PrefillBatchAccumulator:
    """Accumulates finished-prefill entries into a "last prefill batch".

    Merge rules (see ``observe``):
      * Entries with ``num_prompt_tokens <= 0`` or ``prefill_time <= 0``
        are skipped (they carry no usable prefill work).
      * If the last batch update happened within ``merge_seconds`` of
        ``now`` (monotonic), the new valid entries are folded into the
        current batch (summing tokens, window seconds, cached tokens).
        Otherwise a fresh batch starts.
      * ``merge_seconds`` is clamped to >= 0. With 0, effectively every
        ``observe`` call starts a fresh batch (unless the monotonic clock
        reports the identical tick).
    """

    def __init__(self, merge_seconds: float = 0.0) -> None:
        try:
            merge_seconds = float(merge_seconds)
        except (TypeError, ValueError):
            merge_seconds = 0.0
        if not math.isfinite(merge_seconds):  # NaN or +/-inf
            merge_seconds = 0.0
        self.merge_seconds: float = max(0.0, merge_seconds)
        self._tokens: int = 0
        self._window_seconds: float = 0.0
        self._cached_tokens: int = 0
        self._batch_size: int = 0
        self._last_update: Optional[float] = None
        self._last_accepted: int = 0

    @property
    def has_batch(self) -> bool:
        """True once at least one valid entry has been observed."""
        return self._last_update is not None

    @property
    def last_accepted(self) -> int:
        """Number of entries ACCEPTED by the most recent ``observe`` call.

        Entries skipped for ``num_prompt_tokens <= 0`` or
        ``prefill_time <= 0`` (or malformed) do NOT count. This is 0 when
        the last ``observe`` call accepted nothing (and therefore did not
        update the batch)."""
        return self._last_accepted

    @property
    def current(self) -> Optional[PrefillBatchSnapshot]:
        """Snapshot of the current batch, or None before the first entry."""
        if not self.has_batch:
            return None
        return self._snapshot()

    def observe(
        self, entries: Iterable[Any], now: float
    ) -> Optional[PrefillBatchSnapshot]:
        """Fold ``entries`` (at monotonic time ``now``) into the batch.

        ``entries`` is an iterable of (num_prompt_tokens, prefill_time,
        num_cached_tokens) tuples. Invalid entries are skipped silently.

        Returns the updated batch snapshot, or None if no valid entry was
        seen (batch state is unchanged).
        """
        tokens = 0
        window = 0.0
        cached = 0
        count = 0
        self._last_accepted = 0
        for entry in entries:
            try:
                num_prompt_tokens = int(entry[0])
                prefill_time = float(entry[1])
                num_cached_tokens = int(entry[2])
            except (TypeError, ValueError, IndexError):
                continue
            if num_prompt_tokens <= 0 or not prefill_time > 0.0:
                continue
            tokens += num_prompt_tokens
            window += prefill_time
            cached += max(0, num_cached_tokens)
            count += 1
        if count == 0:
            return None

        try:
            now = float(now)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(now):
            return None

        last_update = self._last_update
        if (
            last_update is not None
            and 0.0 <= (now - last_update) <= self.merge_seconds
        ):
            # Fold into the current batch.
            self._tokens += tokens
            self._window_seconds += window
            self._cached_tokens += cached
            self._batch_size += count
        else:
            # First batch, or the merge window elapsed: start fresh.
            self._tokens = tokens
            self._window_seconds = window
            self._cached_tokens = cached
            self._batch_size = count
        self._last_update = now
        self._last_accepted = count
        return self._snapshot()

    def reset(self) -> None:
        """Drop all accumulated state."""
        self._tokens = 0
        self._window_seconds = 0.0
        self._cached_tokens = 0
        self._batch_size = 0
        self._last_update = None
        self._last_accepted = 0

    def _snapshot(self) -> PrefillBatchSnapshot:
        return PrefillBatchSnapshot(
            tokens=self._tokens,
            window_seconds=self._window_seconds,
            cached_tokens=self._cached_tokens,
            avg_pps=compute_avg_pps(
                self._tokens, self._window_seconds, self._cached_tokens
            ),
            batch_size=self._batch_size,
        )
