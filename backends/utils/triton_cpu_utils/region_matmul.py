import functools
import subprocess

import torch

from .sfc_matmul import sfc_matmul
from .thread_lru_cache import thread_lru_cache


@functools.lru_cache(maxsize=1)
def _l2_cache_bytes():
    try:
        size = int(subprocess.check_output(["getconf", "LEVEL2_CACHE_SIZE"], timeout=1))
        if size > 0:
            return size
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return 1024 * 1024


def _group_m(block_sizes, blocks_k_per_pass, element_size):
    """Fit one B panel and grouped A panels in three quarters of L2."""
    block_size_m, block_size_n, block_size_k = block_sizes
    pass_k = blocks_k_per_pass * block_size_k
    a_bytes = max(1, pass_k * block_size_m * element_size)
    b_bytes = pass_k * block_size_n * element_size
    accum_bytes = block_size_m * block_size_n * 4
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
    return min(
        shapes,
        key=lambda shape: abs(blocks_m // shape[0] - blocks_n // shape[1]),
        default=None,
    )


@thread_lru_cache()
def _make_region_tensor(blocks_m, blocks_n, programs, group_m, *, grid=None):
    """Tiles of equal per-program output regions, swept by groups of row blocks."""
    if grid is None:
        grid = _region_grid(blocks_m, blocks_n, programs)
    if grid is None:
        return None
    regions_m, regions_n = grid
    if (
        regions_m * regions_n != programs
        or blocks_m % regions_m
        or blocks_n % regions_n
    ):
        return None

    rows, columns = blocks_m // regions_m, blocks_n // regions_n
    tiles = []
    for program in range(programs):
        row_begin = program // regions_n * rows
        column_begin = program % regions_n * columns
        groups = list(range(row_begin, row_begin + rows, group_m))
        shift = program % regions_n % len(groups)
        for group in groups[shift:] + groups[:shift]:
            for block_n in range(column_begin, column_begin + columns):
                for block_m in range(group, min(group + group_m, row_begin + rows)):
                    tiles.append((block_m, block_n))
    return torch.tensor(tiles, dtype=torch.int32, device="cpu")


def _tile_map(
    blocks_m,
    blocks_n,
    programs,
    blocks_k_per_pass,
    element_size,
    block_sizes,
    *,
    grid=None,
):
    group_m = _group_m(block_sizes, blocks_k_per_pass, element_size)
    return _make_region_tensor(blocks_m, blocks_n, programs, group_m, grid=grid)


_WIDE_REGION_MAPS = {
    grid: functools.partial(_tile_map, grid=grid) for grid in ((32, 2), (64, 1))
}


def region_matmul(a, b, *, region_grid=None, **kwargs):
    """Use equal output regions, or SFC order when no equal regions exist.

    A wide grid pins full K and one program per worker on eligible launches.
    Other launches retain the default region schedule and tuning.
    """
    if region_grid is not None and region_grid not in _WIDE_REGION_MAPS:
        raise ValueError(f"Unsupported region grid: {region_grid}")
    if (
        region_grid is not None
        and kwargs.get("b_is_prepacked")
        and kwargs.get("blocking_factor_k") in (None, 1)
        and kwargs.get("progs_per_thread") in (None, 1)
        and a.shape == (16384, 4096)
        and b.shape == (4096, 4096)
        and a.dtype == b.dtype == torch.bfloat16
        and torch.get_num_threads() == 64
        and all(
            kwargs.get(name, 32) == 32
            for name in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K")
        )
        and _group_m((32, 32, 32), 4096 // 32, a.element_size()) == 4
    ):
        kwargs.update(blocking_factor_k=1, progs_per_thread=1)
        return sfc_matmul(a, b, _tile_map=_WIDE_REGION_MAPS[region_grid], **kwargs)
    return sfc_matmul(a, b, _tile_map=_tile_map, **kwargs)
