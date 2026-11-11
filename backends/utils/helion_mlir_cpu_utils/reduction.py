"""Matrix-vector product (gemv) on Helion's MLIR backend.

A memory-bound row reduction: each tile of rows multiplies its K-wide slices
by the vector's and sums them, in f32. No matrix unit: a gemv reads every
element of A once, so its speed is A's bandwidth.
"""

from __future__ import annotations

from collections import OrderedDict
import os
from typing import Callable

import helion
import helion.language as hl
from helion_mlir_backend import output_pool
import torch
from torch import Tensor

from .kernels import per_config
from .matmul import identity_epilogue


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[32, 32]))
def _matvec_kernel(a: Tensor, x: Tensor) -> Tensor:
    """``a @ x`` of ``[M, K]`` and ``[K]``. Products accumulate per K lane, summed
    once at the end: a sum per K tile vectorizes across rows, reading A by
    columns."""
    m, k = a.shape
    block_k = hl.register_block_size(k)
    out = output_pool.empty([m], dtype=a.dtype, device=a.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_k], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=block_k):
            acc = acc + a[tile_m, tile_k].float() * x[tile_k].float()[None, :]
        out[tile_m] = acc.sum(-1).to(a.dtype)
    return out


def supports_matvec(a: Tensor, b: Tensor) -> bool:
    """Whether :func:`matvec` can handle this pair of operands."""
    if a.dtype != b.dtype or a.dtype not in (torch.float32, torch.bfloat16):
        return False
    if a.device.type != "cpu" or b.device.type != "cpu":
        return False
    if a.dim() != 2 or b.dim() != 2 or b.shape[1] != 1:
        return False
    return a.shape[1] == b.shape[0]


def matvec(a: Tensor, b: Tensor) -> Tensor:
    """``A @ b`` for ``A: [M, K]``, ``b: [K, 1]`` -> ``[M, 1]``.

    Raises ``ValueError`` if unsupported (see :func:`supports_matvec`) instead
    of silently falling back to eager PyTorch -- callers that need a fallback
    must check :func:`supports_matvec` themselves and choose one explicitly.
    """
    if not supports_matvec(a, b):
        raise ValueError(
            f"matvec() does not support these operands: a.shape={tuple(a.shape)} "
            f"a.dtype={a.dtype} a.device={a.device}, b.shape={tuple(b.shape)} "
            f"b.dtype={b.dtype} b.device={b.device}. b must be K-by-1, dtype must be "
            "float32 or bfloat16, both on cpu. Check supports_matvec() before "
            "calling, or use torch.matmul directly for unsupported cases."
        )
    m, k = a.shape
    return _matvec_kernel(a, b.reshape(k)).reshape(m, 1)


def _row_tile(m: int, threads: int) -> int:
    """Rows per tile of a row reduction: every thread a tile, whole tiles only.
    A partial row tile reads its rows through a padded view of unknown strides,
    element by element, and building such a bf16 vector of more than 32 lanes
    aborts LLVM's x86 code generation (33x65 on 8 threads)."""
    rows = max(1, min(32, -(-m // threads)))
    return rows if m % rows == 0 else 1


@per_config(backend="mlir")
def _row_sum_kernel(x: Tensor, epilogue: Callable[[Tensor], Tensor]) -> Tensor:
    """``epilogue`` of the f32 row sums of ``[M, N]``, as ``[M, 1]``."""
    m, n = x.shape
    hl.specialize(n)
    block_n = hl.register_block_size(n)
    out = output_pool.empty((m, 1), dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + x[tile_m, tile_n].to(torch.float32)
        out[tile_m, :] = epilogue(acc.sum(-1, keepdim=True)).to(x.dtype)
    return out


# Least recently used bound kernels, evicted past _ROW_SUM_BOUND_SIZE.
_ROW_SUM_BOUND: OrderedDict[tuple, object] = OrderedDict()
_ROW_SUM_BOUND_SIZE = 64


def row_sum(
    x: Tensor, epilogue: Callable[[Tensor], Tensor] = identity_epilogue
) -> Tensor:
    """``epilogue(x.sum(1, keepdim=True))`` of a 2-D ``x``, summed in f32; rows
    split evenly over the threads."""
    if x.dim() != 2:
        raise ValueError(f"row_sum() expects a 2-D tensor, got {tuple(x.shape)}")
    m, n = map(int, x.shape)
    threads = int(os.environ.get("OMP_NUM_THREADS", torch.get_num_threads()))
    # The column block is registered first.
    block_sizes = (min(n, 512), _row_tile(m, threads))
    x = x.contiguous()
    # The kernel bound once per shape, dtype and epilogue: binding per call
    # costs more host time than a small kernel itself.
    key = (block_sizes, m, n, x.dtype, x.device, epilogue)
    if (bound := _ROW_SUM_BOUND.get(key)) is None:
        bound = _row_sum_kernel(block_sizes).bind((x, epilogue))
        _ROW_SUM_BOUND[key] = bound
        if len(_ROW_SUM_BOUND) > _ROW_SUM_BOUND_SIZE:
            _ROW_SUM_BOUND.popitem(last=False)
    else:
        _ROW_SUM_BOUND.move_to_end(key)
    return bound(x, epilogue)


@per_config(backend="mlir")
def _column_sum_kernel(x: Tensor) -> Tensor:
    """The f32 sums of the columns of ``[M, N]``, as ``[N]``."""
    m, n = x.shape
    hl.specialize(m)
    block_m = hl.register_block_size(m)
    out = output_pool.empty((n,), dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        acc = hl.zeros([block_m, tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m, block_size=block_m):
            acc = acc + x[tile_m, tile_n].to(torch.float32)
        out[tile_n] = acc.sum(0)
    return out


_COLUMN_SUM_BOUND: OrderedDict[tuple, object] = OrderedDict()


def column_sum(x: Tensor) -> Tensor:
    """``x.sum(0)`` of a 2-D ``x`` in f32; columns split evenly over the threads.
    One row needs no sum, and no kernel: a partial column tile of one bf16 row aborts
    LLVM's x86 code generation (1x577 on one thread)."""
    if x.dim() != 2:
        raise ValueError(f"column_sum() expects a 2-D tensor, got {tuple(x.shape)}")
    m, n = map(int, x.shape)
    if m == 1:
        # New memory, as the sum is, and +0.0 where the row has -0.0, as a sum has.
        return x[0].to(torch.float32) + 0.0
    threads = int(os.environ.get("OMP_NUM_THREADS", torch.get_num_threads()))
    block_sizes = (min(m, 32), max(1, min(512, -(-n // threads))))
    x = x.contiguous()
    key = (block_sizes, m, n, x.dtype, x.device)
    if (bound := _COLUMN_SUM_BOUND.get(key)) is None:
        bound = _column_sum_kernel(block_sizes).bind((x,))
        _COLUMN_SUM_BOUND[key] = bound
        if len(_COLUMN_SUM_BOUND) > _ROW_SUM_BOUND_SIZE:
            _COLUMN_SUM_BOUND.popitem(last=False)
    else:
        _COLUMN_SUM_BOUND.move_to_end(key)
    return bound(x)


@per_config(backend="mlir")
def _row_dot_kernel(
    x: Tensor, v: Tensor, epilogue: Callable[[Tensor], Tensor]
) -> Tensor:
    """``epilogue`` of the f32 dot products of the rows of ``[M, N]`` with the
    f32 ``[N]``, as ``[M, 1]``."""
    m, n = x.shape
    hl.specialize(n)
    hl.specialize(v.size(0))
    block_n = hl.register_block_size(n)
    out = output_pool.empty((m, 1), dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + x[tile_m, tile_n].to(torch.float32) * v[tile_n][None, :]
        out[tile_m, :] = epilogue(acc.sum(-1, keepdim=True)).to(x.dtype)
    return out


_ROW_DOT_BOUND: OrderedDict[tuple, object] = OrderedDict()


def row_dot(
    x: Tensor, v: Tensor, epilogue: Callable[[Tensor], Tensor] = identity_epilogue
) -> Tensor:
    """``epilogue(x @ v)`` of a 2-D ``x`` and an f32 ``[N]``, in f32 and rounded
    once, as ``[M, 1]``; rows split evenly over the threads. Reads each element
    of ``x`` once, so its speed is ``x``'s bandwidth."""
    if x.dim() != 2 or v.dim() != 1 or x.shape[1] != v.shape[0]:
        raise ValueError(
            f"row_dot() expects [M, N] and [N], got {tuple(x.shape)}, {tuple(v.shape)}"
        )
    if v.dtype != torch.float32:
        raise ValueError(f"row_dot() expects an f32 vector, got {v.dtype}")
    m, n = map(int, x.shape)
    threads = int(os.environ.get("OMP_NUM_THREADS", torch.get_num_threads()))
    block_sizes = (min(n, 512), _row_tile(m, threads))
    x = x.contiguous()
    key = (block_sizes, m, n, x.dtype, x.device, epilogue)
    if (bound := _ROW_DOT_BOUND.get(key)) is None:
        bound = _row_dot_kernel(block_sizes).bind((x, v, epilogue))
        _ROW_DOT_BOUND[key] = bound
        if len(_ROW_DOT_BOUND) > _ROW_SUM_BOUND_SIZE:
            _ROW_DOT_BOUND.popitem(last=False)
    else:
        _ROW_DOT_BOUND.move_to_end(key)
    return bound(x, v, epilogue)
