"""prefill-last stat-logger plugin for vLLM.

Problem
-------
vLLM's ``vllm:prompt_tokens`` counter credits an entire prefill at request
completion. A single long prompt therefore produces a 1-second spike in a
1-second /metrics poller (e.g. ~95k prompt tokens/s) that does not
represent sustained throughput.

What this plugin does
---------------------
It registers four gauges (no labels, system totals) describing the *last
prefill batch* -- the finished requests whose completions landed within a
merge window (default 0.0 s: per-request batches, same-tick completions
still sum; set ``VLLM_PREFILL_METRICS_MERGE_SECONDS`` > 0 for aggregate
batches):

  * ``vllm_plugin_prefill_last_tokens``          total prompt tokens
  * ``vllm_plugin_prefill_last_window_seconds``  summed per-request
    prefill_time ("work-seconds", not wall-clock)
  * ``vllm_plugin_prefill_last_avg_pps``         (tokens - cached) /
    window_seconds -- the COMPUTED (non-cached) rate, GPU work only
  * ``vllm_plugin_prefill_last_cached_tokens``   summed cached prompt tokens

plus one counter, ``vllm_plugin_prefill_completions_total``, that counts
the finished-request entries ACCEPTED into the batch (i.e. the number of
completions that updated the last-batch gauges). A 1-second poller reads the
counter delta each second: delta > 0 means the four gauges were just updated
and are safe to read, and the delta equals the number of completions merged
into that gauge update.

A 1-second poller can then reconcile the spike: spike tokens == gauge
tokens; avg_pps is the pure GPU-compute rate (cached/skipped tokens
excluded), and the cache-inclusive rate is derivable as
tokens / window_seconds.

Metric naming
-------------
The ``vllm_plugin_`` prefix (instead of the built-in ``vllm:`` prefix)
avoids colliding with vLLM's own metrics and keeps these gauges safe from
``vllm.v1.metrics.prometheus.unregister_vllm_metrics()`` (imported by
vllm.v1.metrics.loggers), which only sweeps ``vllm:``-prefixed names when
PrometheusStatLogger re-initialises.

Single-instance guarantee
-------------------------
``PrefillLastStatLogger`` subclasses ``AggregateStatLoggerBase``. In this
vLLM tree, ``StatLoggerManager`` instantiates aggregate factory classes
EXACTLY ONCE for all engines, while plain ``StatLoggerBase`` subclasses
are wrapped in a per-engine adapter (one instance per DP engine) -- with
N engines that would register the same-named gauges N times and raise
``prometheus_client``'s duplicate-metric error. The aggregate base still
receives ``iteration_stats`` in ``record()`` for every engine, so the
gauges remain system-wide totals. Gauge/counter registration is
additionally guarded so a (should-not-happen) duplicate registration
reuses the existing collector and logs clearly instead of crashing
startup.

Safety
------
``record()`` wraps its entire body in try/except and never propagates;
errors are logged once per distinct exception type (rate-limited). The
plugin only sets gauges and increments a counter, starts no threads, and
touches no request path: it cannot affect serving. If the vLLM stats
interface changes, the plugin degrades to a logged no-op.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Set, Type

from prometheus_client import Counter, Gauge

from .merge import PrefillBatchAccumulator

try:
    from vllm.v1.metrics.loggers import AggregateStatLoggerBase
except ImportError as _exc:
    # Only reached if this module is imported outside a vLLM deployment
    # (or with stubs absent in tests). vLLM's plugin loader catches import
    # failures of plugins and logs them, so this cannot break startup.
    raise ImportError(
        "statlogger_plugin requires "
        "vllm.v1.metrics.loggers.AggregateStatLoggerBase; it must be "
        "loaded inside a vLLM deployment (see ai/README.md)."
    ) from _exc

logger: logging.Logger
try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # pragma: no cover - stdlib fallback
    logger = logging.getLogger(__name__)

__all__ = [
    "PrefillLastStatLogger",
    "parse_merge_seconds",
    "resolve_registry",
    "get_or_create_gauge",
    "get_or_create_counter",
    "METRIC_TOKENS",
    "METRIC_WINDOW_SECONDS",
    "METRIC_AVG_PPS",
    "METRIC_CACHED_TOKENS",
    "METRIC_COMPLETIONS",
]

DEFAULT_MERGE_SECONDS: float = 0.0
MERGE_SECONDS_ENV: str = "VLLM_PREFILL_METRICS_MERGE_SECONDS"

METRIC_TOKENS = "vllm_plugin_prefill_last_tokens"
METRIC_WINDOW_SECONDS = "vllm_plugin_prefill_last_window_seconds"
METRIC_AVG_PPS = "vllm_plugin_prefill_last_avg_pps"
METRIC_CACHED_TOKENS = "vllm_plugin_prefill_last_cached_tokens"
METRIC_COMPLETIONS = "vllm_plugin_prefill_completions_total"

_METRIC_DOCS: Dict[str, str] = {
    METRIC_TOKENS: (
        "Total prompt tokens in the most recent prefill batch (finished "
        "requests within the merge window)."
    ),
    METRIC_WINDOW_SECONDS: (
        "Sum of per-request prefill_time in seconds (work-seconds, not "
        "wall-clock) for the current prefill batch."
    ),
    METRIC_AVG_PPS: (
        "Average COMPUTED (non-cached) prefill prompt tokens/s for the "
        "current batch: (tokens - cached_tokens) / window_seconds. "
        "GPU work only; cached/skipped tokens are excluded."
    ),
    METRIC_CACHED_TOKENS: (
        "Sum of cached prompt tokens (local + external prefix cache) in "
        "the current prefill batch."
    ),
    METRIC_COMPLETIONS: (
        "Total prefill completions that updated the last-batch gauges "
        "since process start."
    ),
}

# Creation order (stable, matches _METRIC_DOCS insertion order).
_METRIC_NAMES: List[str] = [
    METRIC_TOKENS,
    METRIC_WINDOW_SECONDS,
    METRIC_AVG_PPS,
    METRIC_CACHED_TOKENS,
]


def parse_merge_seconds(env: Optional[Mapping[str, str]] = None) -> float:
    """Read the merge window from ``VLLM_PREFILL_METRICS_MERGE_SECONDS``.

    Default 0.0 s (per-request batches; same-tick completions still sum);
    invalid or non-finite (NaN/inf) values fall back to the default;
    negative values are clamped to 0.
    """
    source: Mapping[str, str] = os.environ if env is None else env
    raw = source.get(MERGE_SECONDS_ENV)
    if raw is None:
        return DEFAULT_MERGE_SECONDS
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "prefill-last plugin: invalid %s=%r; using default %s s.",
            MERGE_SECONDS_ENV,
            raw,
            DEFAULT_MERGE_SECONDS,
        )
        return DEFAULT_MERGE_SECONDS
    if not math.isfinite(value):  # NaN or +/-inf
        logger.warning(
            "prefill-last plugin: %s is non-finite; using default %s s.",
            MERGE_SECONDS_ENV,
            DEFAULT_MERGE_SECONDS,
        )
        return DEFAULT_MERGE_SECONDS
    if value < 0.0:
        logger.warning(
            "prefill-last plugin: %s=%r is negative; clamped to 0.",
            MERGE_SECONDS_ENV,
            raw,
        )
        return 0.0
    return value


def resolve_registry() -> Any:
    """Return the Prometheus registry vLLM exposes.

    Prefers ``vllm.v1.metrics.prometheus.get_prometheus_registry`` (the
    module vllm.v1.metrics.loggers already imports from, so it exists in
    this tree); falls back to ``prometheus_client.REGISTRY``; returns
    None (and logs) if neither is available.
    """
    try:
        from vllm.v1.metrics.prometheus import get_prometheus_registry

        registry = get_prometheus_registry()
        if registry is not None:
            return registry
    except Exception:
        logger.warning(
            "prefill-last plugin: could not import "
            "vllm.v1.metrics.prometheus.get_prometheus_registry; "
            "falling back to prometheus_client.REGISTRY.",
            exc_info=True,
        )
    try:
        import prometheus_client
    except ImportError:
        logger.error(
            "prefill-last plugin: prometheus_client is unavailable; "
            "prefill-last gauges will be disabled."
        )
        return None
    return prometheus_client.REGISTRY


def get_or_create_gauge(registry: Any, name: str, documentation: str) -> Gauge:
    """Create a Gauge in ``registry``, idempotency-safe.

    Duplicate registration (should not happen: the aggregate base gives a
    single instance) is handled by reusing the existing collector and
    logging a clear warning instead of raising.
    """
    try:
        return Gauge(
            name=name,
            documentation=documentation,
            registry=registry,
            multiprocess_mode="mostrecent",
        )
    except ValueError:
        # Best-effort lookup of the existing collector. prometheus_client
        # registries keep name -> collector in ``_names_to_collectors``;
        # any registry exposing the same map (incl. test fakes) works.
        existing = getattr(registry, "_names_to_collectors", {}).get(name)
        if existing is not None:
            logger.warning(
                "prefill-last plugin: gauge %r was already registered in "
                "the Prometheus registry (a single plugin instance is "
                "expected); reusing the existing collector.",
                name,
            )
            return existing
        logger.error(
            "prefill-last plugin: gauge %r reported as a duplicate but no "
            "existing collector was found; cannot register safely.",
            name,
        )
        raise


def get_or_create_counter(
    registry: Any, name: str, documentation: str
) -> Counter:
    """Create a Counter in ``registry``, idempotency-safe.

    Mirrors ``get_or_create_gauge``: a (should-not-happen) duplicate
    registration reuses the existing collector and logs a clear warning
    instead of raising.
    """
    try:
        return Counter(
            name=name,
            documentation=documentation,
            registry=registry,
        )
    except ValueError:
        existing = getattr(registry, "_names_to_collectors", {}).get(name)
        if existing is not None:
            logger.warning(
                "prefill-last plugin: counter %r was already registered "
                "in the Prometheus registry (a single plugin instance is "
                "expected); reusing the existing collector.",
                name,
            )
            return existing
        logger.error(
            "prefill-last plugin: counter %r reported as a duplicate but "
            "no existing collector was found; cannot register safely.",
            name,
        )
        raise


class PrefillLastStatLogger(AggregateStatLoggerBase):
    """Stat-logger plugin exposing the "last prefill batch" as gauges.

    See the module docstring for the single-instance rationale
    (AggregateStatLoggerBase) and safety properties.
    """

    def __init__(self, vllm_config: Any, engine_indexes: Optional[List[int]]):
        self.vllm_config = vllm_config
        self.engine_indexes: List[int] = (
            list(engine_indexes) if engine_indexes else [0]
        )
        self.merge_seconds: float = parse_merge_seconds()
        self._accumulator = PrefillBatchAccumulator(self.merge_seconds)
        self._registry = resolve_registry()
        self._gauges: Dict[str, Gauge] = {}
        self._completions_counter: Optional[Counter] = None
        if self._registry is None:
            logger.error(
                "prefill-last plugin: no Prometheus registry available; "
                "gauges disabled (plugin degrades to a no-op)."
            )
        else:
            for metric_name in _METRIC_NAMES:
                self._gauges[metric_name] = get_or_create_gauge(
                    self._registry, metric_name, _METRIC_DOCS[metric_name]
                )
            self._completions_counter = get_or_create_counter(
                self._registry,
                METRIC_COMPLETIONS,
                _METRIC_DOCS[METRIC_COMPLETIONS],
            )
        self._logged_error_types: Set[Type[BaseException]] = set()
        logger.debug(
            "prefill-last plugin loaded: engines=%s merge_seconds=%.3f",
            self.engine_indexes,
            self.merge_seconds,
        )

    def record(
        self,
        scheduler_stats: Any,
        iteration_stats: Any,
        mm_cache_stats: Any = None,
        engine_idx: int = 0,
    ) -> None:
        """Update the gauges from finished requests. Never raises."""
        try:
            if iteration_stats is None:
                return
            finished = getattr(iteration_stats, "finished_requests", None)
            if finished is None:
                return
            entries = []
            for req in finished:
                # getattr-with-defaults so future vllm changes degrade
                # (skipped entry) instead of crashing the engine loop.
                entries.append(
                    (
                        getattr(req, "num_prompt_tokens", 0) or 0,
                        getattr(req, "prefill_time", 0.0) or 0.0,
                        getattr(req, "num_cached_tokens", 0) or 0,
                    )
                )
            snapshot = self._accumulator.observe(entries, time.monotonic())
            if snapshot is None or not self._gauges:
                return
            # Batch (re)established or extended: publish it.
            self._gauges[METRIC_TOKENS].set(float(snapshot.tokens))
            self._gauges[METRIC_WINDOW_SECONDS].set(
                float(snapshot.window_seconds)
            )
            self._gauges[METRIC_AVG_PPS].set(float(snapshot.avg_pps))
            self._gauges[METRIC_CACHED_TOKENS].set(
                float(snapshot.cached_tokens)
            )
            # Count the completions that were ACCEPTED into this batch
            # update. observe() returns a snapshot only when at least one
            # entry was accepted, so last_accepted >= 1 here; entries
            # skipped for tokens<=0 / window<=0 are not counted.
            if self._completions_counter is not None:
                self._completions_counter.inc(
                    self._accumulator.last_accepted
                )
        except Exception as exc:
            self._log_error_once(exc)

    def log(self) -> None:
        # Gauges are always current; nothing to flush at log interval.
        return

    def log_engine_initialized(self) -> None:
        logger.debug("prefill-last plugin loaded (engine initialized).")

    def _log_error_once(self, exc: BaseException) -> None:
        """Log an exception from record() once per distinct error type."""
        etype = type(exc)
        if etype in self._logged_error_types:
            return
        self._logged_error_types.add(etype)
        try:
            logger.error(
                "prefill-last plugin: record() failed with %s: %s "
                "(further errors of this type are suppressed; the plugin "
                "is degrading to a logged no-op and does not affect "
                "serving).",
                etype.__name__,
                exc,
            )
        except Exception:
            pass
