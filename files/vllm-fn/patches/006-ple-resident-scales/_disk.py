"""PLE hashing/decoding over shared batch-bounded io_uring row storage."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable

import torch
from .._shared.disk_table import DiskPrefetch, DiskRowCache, MappedHostAllocation

if TYPE_CHECKING:
    from ._contracts import Binding, Plan


class DiskTable:
    """Own PLE file sources and decode a compact batch cache outside graphs."""

    def __init__(
        self,
        plan: Plan,
        shard_rows: int,
        *,
        queue_depth: int = 64,
        resident_scales: bool | None = None,
    ) -> None:
        from ._contracts import Plan

        if not isinstance(plan, Plan):
            raise TypeError("plan must be Plan")
        if plan.caps.table_memory != "io_uring":
            raise ValueError("DiskTable requires table_memory='io_uring'")
        if resident_scales is None:
            resident_scales = os.environ.get("B12X_PLE_RESIDENT_SCALES", "0") == "1"
        if resident_scales and (
            plan.caps.quant_mode != "nvfp4_group16" or plan.scale_shape is None
        ):
            raise ValueError("resident scales require nvfp4_group16 scale planes")
        self.plan = plan
        self.resident_scales = bool(resident_scales)
        self._scale_owner: MappedHostAllocation | None = None
        self._scale_sources: set[int] = set()
        self._cache = DiskRowCache(
            device=plan.caps.device,
            max_lookups=plan.caps.max_tokens * plan.head_count,
            table_rows=plan.padded_vocab_size,
            shard_start=plan.shard_start,
            shard_end=plan.shard_end,
            shard_rows=shard_rows,
            weight_row_bytes=plan.weight_shape[1] * plan.weight_dtype.itemsize,
            scale_row_bytes=(
                0
                if self.resident_scales
                else (
                    plan.head_dim // 16
                    if plan.caps.quant_mode == "nvfp4_group16"
                    else 0
                )
            ),
            queue_depth=queue_depth,
        )
        self._prefetch = DiskPrefetch(self._cache)
        # Decoders read the write-combined mapped staging directly (measured
        # 0.97 ms for the batch); the io_uring scatter fills the host view.
        self.weight = self._cache.weight.view(plan.weight_dtype)
        self.weight_host = self._cache.weight_host.view(plan.weight_dtype)
        if self.resident_scales:
            # Retain the checkpoint's original e4m3 group scales in mapped
            # host RAM: removes the per-row scale-plane reads (half the io_uring
            # requests) at the cost of shard_rows*10 bytes of pinned memory.
            self._scale_owner = MappedHostAllocation(
                plan.scale_shape, torch.uint8, plan.caps.device
            )
            self._scale_owner.host_view.zero_()
            self.weight_scale = self._scale_owner.device_view.view(
                torch.float8_e4m3fn
            )
            self.weight_scale_host = self._scale_owner.host_view.view(
                torch.float8_e4m3fn
            )
        else:
            self.weight_scale = (
                self._cache.scale.view(torch.float8_e4m3fn)
                if self._cache.scale is not None
                else None
            )
            self.weight_scale_host = (
                self._cache.scale_host.view(torch.float8_e4m3fn)
                if self._cache.scale_host is not None
                else None
            )

    def add_shard(
        self, shard_index: int, path: str, offset: int, *, scale: bool = False
    ) -> None:
        if not (scale and self.resident_scales):
            self._cache.add_shard(shard_index, path, offset, scale=scale)
            return
        import os as _os

        shard_index = int(shard_index)
        offset = int(offset)
        cache = self._cache
        with cache._lock:
            if cache._frozen:
                raise RuntimeError("cannot change disk shards after binding")
            if not 0 <= shard_index < cache.shard_count or offset < 0:
                raise ValueError("invalid scale shard index or offset")
            if shard_index in self._scale_sources:
                raise ValueError("checkpoint scale shard is already registered")
            start = shard_index * cache.shard_rows
            end = min(start + cache.shard_rows, cache.table_rows)
            first = max(start, cache.shard_start)
            last = min(end, cache.shard_end)
            if first >= last:
                return
            view = self._scale_owner.host_view[
                first - self.plan.shard_start : last - self.plan.shard_start
            ]
            data = memoryview(view.numpy()).cast("B")
            row_bytes = self.plan.head_dim // 16
            with open(_os.fspath(path), "rb", buffering=0) as source:
                if offset + (end - start) * row_bytes > _os.fstat(
                    source.fileno()
                ).st_size:
                    raise ValueError("scale plane exceeds checkpoint file bounds")
                source.seek(offset + (first - start) * row_bytes)
                done = 0
                while done < len(data):
                    count = source.readinto(data[done : done + (16 << 20)])
                    if not count:
                        raise ValueError("short resident PLE scale read")
                    done += count
            self._scale_sources.add(shard_index)

    def _require_complete(self) -> None:
        self._cache.require_complete()
        if self.resident_scales:
            cache = self._cache
            first = cache.shard_start // cache.shard_rows
            last = (cache.shard_end + cache.shard_rows - 1) // cache.shard_rows
            for shard in range(first, last):
                if shard not in self._scale_sources:
                    raise ValueError(f"missing resident scale shard {shard}")

    def _freeze(self) -> None:
        self._require_complete()
        self._cache.freeze()

    def stats(self) -> dict[str, int | float]:
        result = self._cache.stats()
        result["resident_scale_bytes"] = (
            self._scale_owner.nbytes if self._scale_owner is not None else 0
        )
        return result

    def _hash(self, binding: Binding, token_count: int) -> None:
        from ._kernels import _launch_hash

        plan = self.plan
        caps = plan.caps
        _launch_hash(
            binding.token_ids,
            binding.query_start_loc,
            binding.committed_history,
            binding.num_seqs,
            binding.num_tokens,
            plan.multipliers,
            plan.prime_sizes,
            plan.table_offsets,
            binding._ids,
            binding._hash_binding.request_ids,
            binding.error_code,
            caps.eos_token_id,
            caps.vocab_size,
            caps.max_order,
            caps.heads_per_order,
            caps.max_seqs,
            caps.max_tokens,
            token_count,
        )

    def _lookup(self, binding: Binding, token_count: int) -> None:
        from ._kernels import (
            _launch_bf16_lookup,
            _launch_fp8_lookup,
            _launch_nvfp4_lookup,
        )

        if token_count == 0:
            return
        plan = self.plan
        caps = plan.caps
        args = (
            binding._ids,
            binding.num_tokens,
            binding.out[:token_count],
            caps.max_tokens,
            plan.head_count,
            plan.head_dim,
            caps.embedding_dim,
            plan.table_vocab_size,
            plan.shard_start,
            plan.shard_end,
        )
        if caps.quant_mode == "bf16":
            _launch_bf16_lookup(self.weight, *args, compact_rows=True)
        elif caps.quant_mode == "fp8_e4m3_per_tensor":
            _launch_fp8_lookup(
                self.weight, binding.weight_scale, *args, compact_rows=True
            )
        else:
            _launch_nvfp4_lookup(
                self.weight,
                self.weight_scale,
                binding.weight_scale_2,
                *args,
                compact_rows=True,
                resident_scales=self.resident_scales,
            )

    def begin_prefetch(
        self,
        binding: Binding,
        *,
        token_count: int,
        stream: torch.cuda.Stream,
        stage: Callable[[], None] | None = None,
    ) -> None:
        """Hash on ``stream`` and start the io_uring read on the worker thread.

        The host returns immediately; the worker blocks until hashed IDs land.
        Consume with consume_prefetch() on the main thread: its host-side join
        is the write-combined visibility point before the lookup launch.
        ``stage`` runs inside the transaction before the hash (e.g. copying
        host-prepared PLE inputs into the binding), so it inherits the
        transaction's wait for the previous step's decoder.
        """

        def pre_stage() -> None:
            if stage is not None:
                stage()
            self._hash(binding, token_count)

        self._prefetch.begin(
            binding._ids,
            token_count * self.plan.head_count,
            pre_stage=pre_stage,
            stream=stream,
        )

    def consume_prefetch(self, binding: Binding, *, token_count: int) -> None:
        """Join the in-flight read, then decode on the current stream.

        The join must stay host-side: io_uring completion has no CUDA-visible
        ordering, and the worker's memcpy-scatter into write-combined staging
        is only guaranteed visible to the decoder once this host thread has
        joined and enqueued the lookup (same guarantee as the sync path).
        """
        self._prefetch.join()
        error = self._prefetch.error()
        if error is not None:
            self._prefetch.abort()
            raise error
        with self._prefetch.complete():
            self._lookup(binding, token_count)

    def abort_prefetch(self) -> None:
        """Drain an unconsumed prefetch (invalidated batch) before reuse."""
        self._prefetch.abort()

    def _run(self, binding: Binding, *, token_count: int) -> None:
        from ._kernels import (
            _launch_bf16_lookup,
            _launch_fp8_lookup,
            _launch_hash,
            _launch_nvfp4_lookup,
        )

        plan = self.plan
        caps = plan.caps
        with self._cache.transaction():
            _launch_hash(
                binding.token_ids,
                binding.query_start_loc,
                binding.committed_history,
                binding.num_seqs,
                binding.num_tokens,
                plan.multipliers,
                plan.prime_sizes,
                plan.table_offsets,
                binding._ids,
                binding._hash_binding.request_ids,
                binding.error_code,
                caps.eos_token_id,
                caps.vocab_size,
                caps.max_order,
                caps.heads_per_order,
                caps.max_seqs,
                caps.max_tokens,
                token_count,
            )
            self._cache.read_rows(binding._ids, token_count * plan.head_count)
            if token_count == 0:
                return
            args = (
                binding._ids,
                binding.num_tokens,
                binding.out[:token_count],
                caps.max_tokens,
                plan.head_count,
                plan.head_dim,
                caps.embedding_dim,
                plan.table_vocab_size,
                plan.shard_start,
                plan.shard_end,
            )
            if caps.quant_mode == "bf16":
                _launch_bf16_lookup(self.weight, *args, compact_rows=True)
            elif caps.quant_mode == "fp8_e4m3_per_tensor":
                _launch_fp8_lookup(
                    self.weight, binding.weight_scale, *args, compact_rows=True
                )
            else:
                _launch_nvfp4_lookup(
                    self.weight,
                    self.weight_scale,
                    binding.weight_scale_2,
                    *args,
                    compact_rows=True,
                    resident_scales=self.resident_scales,
                )
