"""Bounded io_uring row staging shared by PLE and Engram.

The native reader owns bounded request planning and I/O. This module owns
storage, source registration and the host/GPU transaction boundary; hashing
and numerical decoding belong to each embedding operation.
"""

from __future__ import annotations

import math
import operator
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext, suppress
from typing import TYPE_CHECKING, Callable, Iterator

import torch
if TYPE_CHECKING:
    from cuda.bindings import runtime as cudart


def _check_cuda(error: cudart.cudaError_t, operation: str) -> None:
    from cuda.bindings import runtime as cudart

    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed: {error}")


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result = []
    for extent in reversed(shape):
        result.append(stride)
        stride *= int(extent)
    return tuple(reversed(result))


def _tensor_from_pointer(
    pointer: int,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    nbytes: int,
) -> torch.Tensor:
    constructor = getattr(torch._C, "_construct_storage_from_data_pointer", None)
    if constructor is None:
        raise RuntimeError(
            "mapped-host storage requires torch._C._construct_storage_from_data_pointer"
        )
    storage = constructor(int(pointer), device, int(nbytes))
    return torch.empty(0, dtype=dtype, device=device).set_(
        storage, 0, shape, _contiguous_strides(shape)
    )


class MappedHostAllocation:
    """Own a mapped page-locked allocation and its CPU/CUDA tensor aliases."""

    def __init__(
        self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> None:
        from cuda.bindings import runtime as cudart

        self._host_pointer = 0
        self._closed = True
        if device.type != "cuda" or device.index is None:
            raise ValueError(
                f"mapped-host storage requires an indexed CUDA device, got {device}"
            )
        nbytes = math.prod(shape) * dtype.itemsize
        if nbytes <= 0:
            raise ValueError(
                f"mapped-host allocation size must be positive, got {nbytes}"
            )
        self.device = device
        self.nbytes = nbytes
        with torch.cuda.device(device):
            error, pointer = cudart.cudaHostAlloc(
                nbytes, cudart.cudaHostAllocMapped | cudart.cudaHostAllocWriteCombined
            )
            _check_cuda(error, "cudaHostAlloc")
            self._host_pointer = int(pointer)
            self._closed = False
            try:
                error, device_pointer = cudart.cudaHostGetDevicePointer(pointer, 0)
                _check_cuda(error, "cudaHostGetDevicePointer")
                self.host_view = _tensor_from_pointer(
                    self._host_pointer,
                    shape=shape,
                    dtype=dtype,
                    device=torch.device("cpu"),
                    nbytes=nbytes,
                )
                self.device_view = _tensor_from_pointer(
                    int(device_pointer),
                    shape=shape,
                    dtype=dtype,
                    device=device,
                    nbytes=nbytes,
                )
            except Exception:
                cudart.cudaFreeHost(pointer)
                self._host_pointer = 0
                self._closed = True
                raise

    def close(self) -> None:
        from cuda.bindings import runtime as cudart

        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        _check_cuda(cudart.cudaFreeHost(self._host_pointer)[0], "cudaFreeHost")
        self._host_pointer = 0
        self._closed = True

    def __del__(self) -> None:
        if self._closed or self._host_pointer == 0 or sys is None or sys.is_finalizing():
            return
        with suppress(Exception):
            self.close()


class DiskRowCache:
    """Read immutable row planes into a reusable, batch-sized mapped cache.

    Source rows have one weight byte plane and an optional independent scale
    byte plane. Neither file data nor full table allocations are retained.
    Hold ``transaction`` across ID production, ``read_rows`` and GPU decoding.
    Downstream graphs consume the decoder's fixed device output, not disk I/O.
    """

    def __init__(
        self,
        *,
        device: torch.device | str,
        max_lookups: int,
        table_rows: int,
        shard_start: int,
        shard_end: int,
        shard_rows: int,
        weight_row_bytes: int,
        scale_row_bytes: int = 0,
        queue_depth: int = 64,
    ) -> None:
        from b12x.loader._native import load

        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("disk row staging requires CUDA")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.max_lookups = operator.index(max_lookups)
        self.table_rows = operator.index(table_rows)
        self.shard_start = operator.index(shard_start)
        self.shard_end = operator.index(shard_end)
        self.shard_rows = operator.index(shard_rows)
        self.weight_row_bytes = operator.index(weight_row_bytes)
        self.scale_row_bytes = operator.index(scale_row_bytes)
        queue_depth = operator.index(queue_depth)
        if not 0 < self.shard_rows <= (1 << 63) - 1:
            raise ValueError("shard_rows must be a positive signed int64")
        self._native = load()
        self._reader = self._native.ple_reader(
            self.shard_rows,
            self.table_rows,
            self.shard_start,
            self.shard_end,
            self.weight_row_bytes,
            self.scale_row_bytes,
            self.max_lookups,
            queue_depth,
        )
        self.shard_count = (self.table_rows + self.shard_rows - 1) // self.shard_rows
        self.ids_host = torch.empty(
            (self.max_lookups,), dtype=torch.int64, device="cpu", pin_memory=True
        )
        self._ids_buffer = memoryview(self.ids_host.numpy())
        self._weight_allocation = MappedHostAllocation(
            (self.max_lookups, self.weight_row_bytes), torch.uint8, device
        )
        self.weight = self._weight_allocation.device_view
        self.weight_host = self._weight_allocation.host_view
        self._weight_buffer = memoryview(self.weight_host.numpy())
        self._scale_allocation = None
        self.scale = None
        self.scale_host = None
        self._scale_buffer = None
        if self.scale_row_bytes:
            self._scale_allocation = MappedHostAllocation(
                (self.max_lookups, self.scale_row_bytes), torch.uint8, device
            )
            self.scale = self._scale_allocation.device_view
            self.scale_host = self._scale_allocation.host_view
            self._scale_buffer = memoryview(self.scale_host.numpy())
        self._sources: set[tuple[bool, int]] = set()
        self._frozen = False
        self._lock = threading.RLock()
        self._transaction_thread: int | None = None
        self._transaction_stream: torch.cuda.Stream | None = None
        self._ids_ready = torch.cuda.Event()
        self._cache_done = torch.cuda.Event()
        self._cache_used = False

    def add_shard(
        self, shard_index: int, path: str, offset: int, *, scale: bool = False
    ) -> None:
        with self._lock:
            if self._frozen:
                raise RuntimeError("cannot change disk shards after binding")
            shard_index = operator.index(shard_index)
            offset = operator.index(offset)
            if not 0 <= shard_index < self.shard_count:
                raise ValueError("checkpoint shard index is out of range")
            if offset < 0:
                raise ValueError("checkpoint file offset must be nonnegative")
            if scale and not self.scale_row_bytes:
                raise ValueError("disk table has no row scale plane")
            key = (scale, shard_index)
            if key in self._sources:
                raise ValueError("checkpoint shard is already registered")
            start = shard_index * self.shard_rows
            end = min(start + self.shard_rows, self.table_rows)
            if end <= self.shard_start or start >= self.shard_end:
                return
            self._native.ple_reader_add(
                self._reader, shard_index, os.fspath(path), offset, scale
            )
            self._sources.add(key)

    def require_complete(self) -> None:
        with self._lock:
            first = self.shard_start // self.shard_rows
            last = (self.shard_end + self.shard_rows - 1) // self.shard_rows
            for shard in range(first, last):
                if (False, shard) not in self._sources:
                    raise ValueError(f"missing disk weight shard {shard}")
                if self.scale_row_bytes and (True, shard) not in self._sources:
                    raise ValueError(f"missing disk scale shard {shard}")

    def freeze(self) -> None:
        with self._lock:
            self.require_complete()
            self._frozen = True

    @contextmanager
    def transaction(self) -> Iterator[DiskRowCache]:
        if torch.compiler.is_compiling():
            raise RuntimeError("disk table preparation cannot run under torch.compile")
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "disk table preparation must run outside CUDA graph capture"
                )
            with self._lock:
                if self._transaction_thread is not None:
                    raise RuntimeError("disk row transactions cannot be nested")
                stream = torch.cuda.current_stream(self.device)
                if self._cache_used:
                    stream.wait_event(self._cache_done)
                self._transaction_thread = threading.get_ident()
                self._transaction_stream = stream
                try:
                    yield self
                finally:
                    # Record on the (possibly re-anchored) decoder stream,
                    # not the enter-time stream: DiskPrefetch.complete()
                    # re-anchors _transaction_stream to the stream the
                    # decoder launched on, so the next begin()'s entry wait
                    # orders staging/hash behind the previous lookup. On the
                    # sync path these are the same stream.
                    self._cache_done.record(self._transaction_stream)
                    self._cache_used = True
                    self._transaction_stream = None
                    self._transaction_thread = None

    def read_rows(self, ids: torch.Tensor, count: int) -> None:
        self._stage_ids(ids, count)
        self._read_staged(count)

    def _stage_ids(self, ids: torch.Tensor, count: int) -> None:
        if self._transaction_thread != threading.get_ident():
            raise RuntimeError("read_rows requires an active disk row transaction")
        count = operator.index(count)
        if not 0 <= count <= self.max_lookups:
            raise ValueError("disk lookup count exceeds batch capacity")
        if (
            ids.device != self.device
            or ids.dtype != torch.int64
            or not ids.is_contiguous()
        ):
            raise ValueError(
                "disk row IDs must be contiguous CUDA int64 on the cache device"
            )
        if ids.numel() < count:
            raise ValueError("disk row IDs do not cover the requested count")
        self.ids_host[:count].copy_(ids.view(-1)[:count], non_blocking=True)
        self._ids_ready.record(self._transaction_stream)

    def _read_staged(self, count: int) -> None:
        # Completes ID production and all prior cache readers before host
        # writes. Runs on the thread that memcpy-scatters the rows into
        # write-combined staging; the decoder reads that staging directly.
        with torch.cuda.device(self.device):
            self._ids_ready.synchronize()
            self._native.ple_reader_run(
                self._reader,
                self._ids_buffer,
                self._weight_buffer,
                self._scale_buffer,
                count,
            )

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            result = dict(self._native.ple_reader_stats(self._reader))
            result["ids_host_bytes"] = (
                self.ids_host.numel() * self.ids_host.element_size()
            )
            result["weight_cache_bytes"] = self._weight_allocation.nbytes
            result["scale_cache_bytes"] = (
                self._scale_allocation.nbytes if self._scale_allocation else 0
            )
            result["cache_bytes"] = (
                result["weight_cache_bytes"] + result["scale_cache_bytes"]
            )
            result["owned_staging_bytes"] = (
                result["staging_bytes"]
                + result["ids_host_bytes"]
                + result["cache_bytes"]
            )
            return result


class DiskPrefetch:
    """One in-flight read, retaining its owner's transaction until completion.

    begin/complete/abort must run on the same host thread outside capture.
    The caller must not mutate the ID tensor before completion. Only the I/O
    worker runs concurrently; it never launches the downstream GPU decoder.
    """

    def __init__(self, cache: DiskRowCache) -> None:
        self.cache = cache
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="b12x-disk")
        self._pending = None
        self._closed = False
        self._done = torch.cuda.Event()
        self._error: BaseException | None = None

    @property
    def pending(self) -> bool:
        return self._pending is not None

    def begin(
        self,
        ids: torch.Tensor,
        count: int,
        *,
        pre_stage: Callable[[], None] | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Stage one batch and start its read on the worker thread.

        ``pre_stage`` runs inside the transaction before ID staging (e.g. the
        PLE hash kernel). ``stream`` selects the preparing stream; ID
        production must be stream-ordered on it. The host returns while the
        worker still blocks on ID production and performs the io_uring read.
        The worker records the completion event on the preparing stream after
        the read, so a consumer stream can wait_event() it without a host
        join. Surface worker failures via error() before the next begin().
        """
        if self._error is not None:
            error, self._error = self._error, None
            raise error
        if self._closed or self._pending is not None:
            raise RuntimeError("disk prefetch is closed or has unconsumed rows")
        preparing = (
            stream
            if stream is not None
            else torch.cuda.current_stream(self.cache.device)
        )
        with torch.cuda.stream(preparing):
            transaction = self.cache.transaction()
            transaction.__enter__()
            try:
                if pre_stage is not None:
                    pre_stage()
                self.cache._stage_ids(ids, count)
                future = self._executor.submit(self._worker, count, preparing)
                self._pending = (future, transaction, threading.get_ident(), ids, count)
            except BaseException:
                transaction.__exit__(*sys.exc_info())
                raise

    def _worker(self, count: int, preparing: torch.cuda.Stream) -> None:
        try:
            with torch.cuda.device(self.cache.device):
                self.cache._read_staged(count)
                # Event record is a CUDA API call on the preparing stream:
                # it completes only after the io_uring scatter writes into
                # write-combined staging are flushed, so a consumer stream
                # waiting on the event sees full rows.
                self._done.record(preparing)
        except BaseException as error:  # surface on the next host touchpoint
            self._error = error
            with suppress(Exception):
                # Never leave a consumer waiting on a dead read.
                with torch.cuda.device(self.cache.device):
                    self._done.record(preparing)

    def error(self) -> BaseException | None:
        error, self._error = self._error, None
        return error

    @property
    def done_event(self) -> torch.cuda.Event:
        """Event recorded on the preparing stream after the in-flight read."""
        return self._done

    def join(self) -> None:
        """Host-side join: wait for the worker and raise any worker error."""
        if self._pending is None:
            raise RuntimeError("disk prefetch has nothing to join")
        self._pending[0].result()
        error = self.error()
        if error is not None:
            raise error

    @contextmanager
    def complete(self) -> Iterator[None]:
        """Close the transaction after the decoder launch on the current stream.

        Callers that joined via done_event (GPU-side) use this around the
        decoder launch: completion is re-anchored on the current (decoder)
        stream so the next begin()'s staging reuse waits for the decoder.
        Callers that need host certainty call join() first.
        """
        if self._pending is None:
            raise RuntimeError("disk prefetch has nothing to complete")
        future, transaction, thread, _, _ = self._pending
        if thread != threading.get_ident():
            raise RuntimeError("disk prefetch must be completed on its preparing thread")
        with torch.cuda.device(self.cache.device):
            if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "disk prefetch completion must run outside compile/capture"
                )
            try:
                # Re-anchor completion on the decoder stream; it may differ
                # from the preparing stream used by begin().
                self.cache._transaction_stream = torch.cuda.current_stream(
                    self.cache.device
                )
                yield
            finally:
                self._pending = None
                transaction.__exit__(*sys.exc_info())

    def abort(self) -> None:
        if self._pending is None:
            return
        future, transaction, thread, _, _ = self._pending
        if thread != threading.get_ident():
            raise RuntimeError("disk prefetch must be aborted on its preparing thread")
        try:
            future.result()
        finally:
            self._pending = None
            transaction.__exit__(*sys.exc_info())

    def close(self) -> None:
        if self._pending is not None and self._pending[2] != threading.get_ident():
            raise RuntimeError("disk prefetch must be closed on its preparing thread")
        try:
            self.abort()
        finally:
            self._closed = True
            self._executor.shutdown(wait=True)
