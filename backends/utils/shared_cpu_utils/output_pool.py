"""``empty``: ``torch.empty`` for large CPU outputs, from a pool of reused mappings.

A block returns to the pool when the last reference to the tensor's storage dies and
is reused for the same size, so a per-call output is not new, page-faulting memory
on every call. What the pool does not take gets ``torch.empty``.
``AIBENCH_OUTPUT_POOL`` (bytes, read at import; 0 turns it off) limits the free
blocks kept.
"""

# From helion_mlir_backend/_compiler/output_pool.py of
# https://github.com/libxsmm/helion-mlir-backend at 01ba1c8.

from __future__ import annotations

from collections import OrderedDict
from collections import deque
import contextlib
import ctypes
import math
import mmap
import os
from pathlib import Path
import threading
import warnings

import torch

OUTPUT_POOL_ENV = "AIBENCH_OUTPUT_POOL"
# Below this torch allocates: a single buffer this small stays on glibc's heap.
MIN_BYTES = 4 * 1024 * 1024
_PAGE = 4096
_HUGE_PAGE = 2 * 1024 * 1024
DEFAULT_MAX_BYTES = 2 << 30
_MAX_FREE_BLOCKS = 64
_CPU = torch.device("cpu")
# What torch.frombuffer and resize_ were checked on.
_DTYPES = frozenset(
    {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    }
)


class _Block:
    """A mapping of ours, ``length`` bytes at ``address``."""

    __slots__ = ("address", "length", "lease", "mapping")

    def __init__(self, length: int) -> None:
        huge = length >= _HUGE_PAGE
        padding = _HUGE_PAGE if huge else 0
        self.mapping = mmap.mmap(
            -1, length + padding, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
        )
        base = ctypes.addressof(ctypes.c_char.from_buffer(self.mapping))
        self.address = -(-base // _HUGE_PAGE) * _HUGE_PAGE if huge else base
        self.length = length
        if huge:  # Huge pages also where THP is madvise-only. Only a hint.
            with contextlib.suppress(OSError, AttributeError):
                self.mapping.madvise(mmap.MADV_HUGEPAGE, self.address - base, length)
        # The class of this block's throwaway arrays: its __del__ knows the block.
        self.lease = type(
            "_Lease",
            (ctypes.c_char * length,),
            {"__slots__": (), "__del__": _return_block, "block": self},
        )

    def close(self) -> None:
        self.lease = None  # breaks the block <-> class cycle
        self.mapping.close()


def _return_block(lease: ctypes.Array) -> None:
    # Runs wherever the storage dies, possibly inside the garbage collector during
    # _take on this very thread: only an atomic append, then a try at the lock.
    _returned.append(lease.block)
    if _lock.acquire(blocking=False):
        try:
            _settle_locked()
        finally:
            _lock.release()


# Returned blocks wait here for the lock's holder.
_returned: deque[_Block] = deque()
_lock = threading.Lock()
# Oldest first; by address. _by_length: the same blocks' addresses, per length.
_free: OrderedDict[int, _Block] = OrderedDict()
_by_length: dict[int, list[int]] = {}
_free_bytes = 0
_max_bytes = 0


def _settle_locked() -> None:
    """File returned blocks; evict what is over the limits."""
    global _free_bytes
    while _returned:
        block = _returned.popleft()
        if block.length > _max_bytes:
            block.close()
            continue
        _free[block.address] = block
        _by_length.setdefault(block.length, []).append(block.address)
        _free_bytes += block.length
    while _free and (_free_bytes > _max_bytes or len(_free) > _MAX_FREE_BLOCKS):
        address, block = _free.popitem(last=False)
        _by_length[block.length].remove(address)
        _free_bytes -= block.length
        block.close()


def _release_locked() -> None:
    global _free_bytes
    for block in _free.values():
        block.close()
    _free.clear()
    _by_length.clear()
    _free_bytes = 0


def _take(length: int) -> _Block:
    global _free_bytes
    if _lock.acquire(blocking=False):
        try:
            _settle_locked()
            addresses = _by_length.get(length)
            if addresses:
                block = _free.pop(addresses.pop())
                _free_bytes -= length
                return block
        finally:
            _lock.release()
    # A miss, also when the lock is taken: it may be this thread's, under us.
    try:
        return _Block(length)
    except OSError:  # out of memory or address space: what we keep goes first
        if _lock.acquire(blocking=False):
            try:
                _settle_locked()
                _release_locked()
            finally:
                _lock.release()
        return _Block(length)


def _allocate(sizes: tuple[int, ...], count: int, dtype: torch.dtype) -> torch.Tensor:
    size = dtype.itemsize * count
    unit = _HUGE_PAGE if size >= _HUGE_PAGE else _PAGE
    block = _take(-(-size // unit) * unit)
    lease = block.lease.from_address(block.address)
    # frombuffer keeps the lease until the storage dies. The shape is set in place,
    # as torch.empty's: not a view of a flat tensor.
    return torch.frombuffer(lease, dtype=dtype, count=count).resize_(sizes)


def _plain_dispatch() -> bool:
    """No torch-function or dispatch mode, no transform and no trace (a traced
    allocation becomes a constant: two calls would share its memory): then
    ``torch.empty`` would dispatch as ``frombuffer`` does."""
    return not (
        torch._C._is_torch_function_mode_enabled()
        or torch._C._len_torch_dispatch_stack()
        or torch._C._functorch.peek_interpreter_stack() is not None
        or torch.jit.is_tracing()
    )


def _takes(count: int, dtype: torch.dtype) -> bool:
    # Deterministic mode fills empty tensors: left to torch.
    return (
        dtype in _DTYPES
        and MIN_BYTES <= dtype.itemsize * count <= _max_bytes
        and not (
            torch.are_deterministic_algorithms_enabled()
            and torch.utils.deterministic.fill_uninitialized_memory
        )
    )


def empty(
    *size: int | tuple[int, ...] | list[int],
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    **kwargs: object,
) -> torch.Tensor:
    """``torch.empty(*size, dtype=dtype, device=device, **kwargs)``, from the pool
    if it takes the tensor: a plain CPU one of ``MIN_BYTES`` or more, no other
    arguments. Its storage cannot grow (``resize_`` beyond it raises)."""
    if _max_bytes < MIN_BYTES or kwargs:
        return torch.empty(*size, dtype=dtype, device=device, **kwargs)
    sizes = size[0] if len(size) == 1 and isinstance(size[0], (tuple, list)) else size
    if all(type(n) is int and n >= 0 for n in sizes):
        kind = torch.get_default_dtype() if dtype is None else dtype
        count = math.prod(sizes)
        if (
            _takes(count, kind)
            and (torch.get_default_device() if device is None else torch.device(device))
            == _CPU
            and _plain_dispatch()
        ):
            try:
                return _allocate(tuple(sizes), count, kind)
            except OSError:
                pass  # torch's own attempt raises its error
    return torch.empty(*size, dtype=dtype, device=device, **kwargs)


def _after_fork_in_child() -> None:
    # The child's copies of the free blocks are its own and nobody will reuse them
    # soon: it starts empty. Blocks of tensors it inherited return as usual.
    _returned.clear()
    _release_locked()
    _lock.release()


# A fork holds the lock over the copy, so a child sees the pool whole and unlocked.
os.register_at_fork(
    before=_lock.acquire,
    after_in_parent=_lock.release,
    after_in_child=_after_fork_in_child,
)


def _cgroup_limit() -> int | None:
    """The tightest memory limit of this process's cgroups (v2 and v1), if any."""
    limits = []
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        _, controllers, path = line.split(":", 2)
        if controllers == "":
            root, name = "/sys/fs/cgroup", "memory.max"
        elif "memory" in controllers.split(","):
            root, name = "/sys/fs/cgroup/memory", "memory.limit_in_bytes"
        else:
            continue
        # A process may see only part of the hierarchy: read what exists.
        parts = path.strip("/").split("/") if path.strip("/") else []
        for depth in range(len(parts), -1, -1):
            # No such file, or "max".
            with contextlib.suppress(OSError, ValueError):
                limits.append(int(Path(root, *parts[:depth], name).read_text()))
    return min(limits) if limits else None


def _default_max_bytes() -> int:
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    limit = _cgroup_limit()
    memory = ram if limit is None else min(ram, limit)
    default = min(DEFAULT_MAX_BYTES, memory // 8)
    value = os.environ.get(OUTPUT_POOL_ENV, "").strip()
    if not value:
        return default
    try:
        chosen = int(value)
        if chosen < 0:
            raise ValueError
    except ValueError:
        warnings.warn(
            f"{OUTPUT_POOL_ENV}={value!r} is not a number of bytes: using {default}",
            stacklevel=2,
        )
        return default
    return chosen


_max_bytes = _default_max_bytes()
