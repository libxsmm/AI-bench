# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
# SPDX-License-Identifier: Apache-2.0

import functools
import subprocess

import torch

from .sfc_matmul import _BLOCK_SIZE_K
from .sfc_matmul import _BLOCK_SIZE_M
from .sfc_matmul import _BLOCK_SIZE_N
from .sfc_matmul import _thread_lru_cache
from .sfc_matmul import sfc_matmul


@functools.lru_cache(maxsize=1)
def _l2_cache_bytes():
    try:
        size = int(subprocess.check_output(["getconf", "LEVEL2_CACHE_SIZE"], timeout=1))
        if size > 0:
            return size
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return 1024 * 1024


def _group_m(blocks_k_per_pass, element_size):
    """Fit one B panel and grouped A panels in three quarters of L2."""
    pass_k = blocks_k_per_pass * _BLOCK_SIZE_K
    a_bytes = max(1, pass_k * _BLOCK_SIZE_M * element_size)
    b_bytes = pass_k * _BLOCK_SIZE_N * element_size
    accum_bytes = _BLOCK_SIZE_M * _BLOCK_SIZE_N * 4
    return max(
        1, min(4, (_l2_cache_bytes() * 3 // 4 - b_bytes - accum_bytes) // a_bytes)
    )


def _region_grid(blocks_m, blocks_n, programs):
    if programs >= blocks_m * blocks_n:
        return None
    shapes = [
        (regions_m, programs // regions_m)
        for regions_m in range(1, programs + 1)
        if programs % regions_m == 0
        and blocks_m % regions_m == 0
        and blocks_n % (programs // regions_m) == 0
    ]
    if not shapes:
        return None
    # Square regions minimize the A and B panels each program reads.
    return min(
        shapes, key=lambda shape: abs(blocks_m // shape[0] - blocks_n // shape[1])
    )


@_thread_lru_cache()
def _make_region_tensor(blocks_m, blocks_n, programs, group_m):
    """Tiles of equal per-program output regions, swept by groups of row blocks."""
    grid = _region_grid(blocks_m, blocks_n, programs)
    if grid is None:
        return None
    regions_m, regions_n = grid
    rows, columns = blocks_m // regions_m, blocks_n // regions_n
    tiles = []
    for program in range(programs):
        row_begin = program // regions_n * rows
        column_begin = program % regions_n * columns
        groups = list(range(row_begin, row_begin + rows, group_m))
        # Programs sharing A rows start at different groups.
        shift = program % regions_n % len(groups)
        for group in groups[shift:] + groups[:shift]:
            for block_n in range(column_begin, column_begin + columns):
                for block_m in range(group, min(group + group_m, row_begin + rows)):
                    tiles.append((block_m, block_n))
    return torch.tensor(tiles, dtype=torch.int32, device="cpu")


def _tile_map(blocks_m, blocks_n, programs, blocks_k_per_pass, element_size):
    group_m = _group_m(blocks_k_per_pass, element_size)
    return _make_region_tensor(blocks_m, blocks_n, programs, group_m)


def region_matmul(a, b, **kwargs):
    """sfc_matmul giving each program one equal output region; SFC order if none exists."""
    return sfc_matmul(a, b, _tile_map=_tile_map, **kwargs)
