"""Row-wise normalizations on Helion's MLIR backend: softmax and GroupNorm.

Each tile holds whole rows (softmax) or whole groups (GroupNorm), so the
statistics are reduced inside the tile, in f32, and the result is written in
one pass.
"""

from __future__ import annotations

import os
from typing import Callable

import helion.language as hl
import helion_mlir_backend  # noqa: F401
from helion_mlir_backend import output_pool
import torch
from torch import Tensor

from .kernels import per_config

# f32 elements of a tile's rows: its intermediates stay in L2.
_TILE_ELEMENTS = 1 << 16
# In place, a store chunk goes through a copy of a whole row tile's chunk, lowered
# per element, which is slow to compile. More tiles than threads are fine.
_IN_PLACE_TILE_ELEMENTS = 1 << 12


@per_config(backend="mlir")
def _softmax_kernel(
    x: Tensor, whole: hl.constexpr, in_place: hl.constexpr = hl.constexpr(False)
) -> Tensor:
    """Softmax over the last dim of ``[M, N]`` in passes over column chunks:
    row maxima, sums of exponentials, then the normalized output. Whole-row
    ops instead get the row reduction fused into, and recomputed by, every
    chunk of the store.

    The ``whole`` columns of whole chunks reduce lane by lane across chunks,
    one reduction per row at the end. A partial last chunk reduces over its
    own columns, so its padding is masked out (``_mask_to``): lane-wise, the
    padded lanes' zeros would enter the maximum and add ``exp(-max)`` per
    lane to the sum.

    ``in_place`` writes into ``x``: a tile's row statistics are reduced before
    its store pass, which reads each chunk before writing it."""
    m, n = x.shape
    hl.specialize(n)
    block_n = hl.register_block_size(n)
    if in_place:
        out = x
    else:
        out = output_pool.empty_like(x)
    for tile_m in hl.tile(m):
        top = hl.full([tile_m, block_n], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(whole, block_size=block_n):
            top = torch.maximum(top, x[tile_m, tile_n].to(torch.float32))
        row_top = torch.amax(top, dim=-1, keepdim=True)
        if whole < n:
            for tile_n in hl.tile(whole, n, block_size=block_n):
                tail = torch.amax(x[tile_m, tile_n].to(torch.float32), -1, True)
                row_top = torch.maximum(row_top, tail)
        total = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(whole, block_size=block_n):
            total = total + torch.exp(x[tile_m, tile_n].to(torch.float32) - row_top)
        row_total = total.sum(-1, keepdim=True)
        if whole < n:
            for tile_n in hl.tile(whole, n, block_size=block_n):
                e = torch.exp(x[tile_m, tile_n].to(torch.float32) - row_top)
                row_total = row_total + e.sum(-1, keepdim=True)
        inv = 1.0 / row_total
        for tile_n in hl.tile(n, block_size=block_n):
            e = torch.exp(x[tile_m, tile_n].to(torch.float32) - row_top)
            out[tile_m, tile_n] = (e * inv).to(x.dtype)
    return out


@per_config(backend="mlir")
def _softmax_one_kernel(x: Tensor) -> Tensor:
    """Softmax of one-column rows ``[M, 1]``: ``x - x + 1``, so 1, and NaN
    where ``x`` is NaN or infinite, as ``exp(x - max) / sum`` gives. The
    chunked kernel does not compile for one column."""
    m, n = x.shape
    hl.specialize(n)
    out = output_pool.empty_like(x)
    for tile_m in hl.tile(m):
        v = x[tile_m, :].to(torch.float32)
        out[tile_m, :] = (v - v + 1.0).to(x.dtype)
    return out


def _second_operand(y: Tensor, extra: Tensor) -> Tensor:
    return y


@per_config(backend="mlir")
def _group_norm_kernel(
    x3: Tensor,
    weight: Tensor,
    bias: Tensor,
    extra: Tensor,
    eps: hl.constexpr,
    epilogue: Callable[[Tensor, Tensor], Tensor],
    in_place: hl.constexpr = hl.constexpr(False),
) -> Tensor:
    """``epilogue(GroupNorm(x), extra)`` of ``x3`` ``[M, G, C/G]``; ``weight``,
    ``bias`` and ``extra`` are per-channel ``[G, C/G]``. ``in_place`` writes
    into ``x3``: a tile reads its groups before writing them."""
    m, groups, group_size = x3.shape
    hl.specialize(group_size)
    if in_place:
        out = x3
    else:
        out = output_pool.empty_like(x3)
    for tile_m, tile_g in hl.tile([m, groups]):
        y = x3[tile_m, tile_g, :].to(torch.float32)
        mean = y.mean(-1, keepdim=True)
        centered = y - mean
        var = (centered * centered).mean(-1, keepdim=True)
        y = centered * torch.rsqrt(var + eps) * weight[tile_g, :].to(
            torch.float32
        ) + bias[tile_g, :].to(torch.float32)
        y = epilogue(y, extra[tile_g, :].to(torch.float32))
        out[tile_m, tile_g, :] = y.to(x3.dtype)
    return out


def _threads() -> int:
    return int(os.environ.get("OMP_NUM_THREADS", torch.get_num_threads()))


def _row_tile(
    rows: int, width: int, other_tiles: int = 1, elements: int = _TILE_ELEMENTS
) -> int:
    """Rows per tile: within ``elements`` and leaving every thread a tile."""
    per_thread = -(-rows * other_tiles // _threads())
    return max(1, min(elements // width, per_thread))


def softmax(x: Tensor, in_place: bool = False) -> Tensor:
    """Softmax over the last dim of a 2-D ``x``. ``in_place`` may overwrite a
    contiguous ``x`` (the caller's fresh intermediate) instead of allocating
    an output; use the result, not ``x``."""
    if x.dim() != 2:
        raise ValueError(f"softmax() expects a 2-D tensor, got {tuple(x.shape)}")
    m, n = map(int, x.shape)
    if n == 1:
        return _softmax_one_kernel((_row_tile(m, 1),))(x.contiguous())
    block_n = min(n, 128)
    whole = hl.constexpr(n // block_n * block_n)
    # The column block is registered first.
    elements = _IN_PLACE_TILE_ELEMENTS if in_place else _TILE_ELEMENTS
    tile_m = _row_tile(m, block_n, elements=elements)
    kernel = _softmax_kernel((block_n, tile_m))
    return kernel(x.contiguous(), whole, hl.constexpr(in_place))


def group_norm(
    x: Tensor,
    num_groups: int,
    weight: Tensor,
    bias: Tensor,
    eps: float,
    epilogue: Callable[[Tensor, Tensor], Tensor] = _second_operand,
    extra: Tensor | None = None,
    in_place: bool = False,
) -> Tensor:
    """``epilogue(GroupNorm(x), extra)`` of ``x`` ``[M, C]``; ``extra`` is an
    optional per-channel ``[C]`` operand of ``epilogue`` (default: identity).
    ``in_place`` may overwrite a contiguous ``x`` (the caller's fresh
    intermediate) instead of allocating an output; use the result, not ``x``."""
    if x.dim() != 2 or x.shape[1] % num_groups:
        raise ValueError(
            f"group_norm() expects [M, C] with C divisible by {num_groups} groups, "
            f"got {tuple(x.shape)}"
        )
    m, c = map(int, x.shape)
    group_size = c // num_groups
    groups_per_tile = max(1, min(num_groups, _TILE_ELEMENTS // 32 // group_size))
    while num_groups % groups_per_tile:
        groups_per_tile -= 1
    tile_m = _row_tile(
        m,
        groups_per_tile * group_size,
        num_groups // groups_per_tile,
        _IN_PLACE_TILE_ELEMENTS if in_place else _TILE_ELEMENTS,
    )
    kernel = _group_norm_kernel((tile_m, groups_per_tile))
    channels = (num_groups, group_size)
    return kernel(
        x.contiguous().view(m, num_groups, group_size),
        weight.view(channels),
        bias.view(channels),
        (weight if extra is None else extra).view(channels),
        hl.constexpr(eps),
        epilogue,
        hl.constexpr(in_place),
    ).view(m, c)
