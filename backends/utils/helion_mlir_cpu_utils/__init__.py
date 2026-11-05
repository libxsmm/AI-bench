"""Helper utilities for AI-bench Helion MLIR CPU kernels."""

from __future__ import annotations

from .elementwise import leaky_relu
from .elementwise import relu
from .elementwise import scalar_mul
from .kernels import per_config
from .linear import CACHE_PREPACKED_WEIGHTS_ENV
from .linear import AffineLinearCache
from .linear import LinearCache
from .linear import cache_prepacked_weights
from .linear import linear
from .linear import linear_affine
from .matmul import bmm
from .matmul import identity_epilogue
from .matmul import matmul
from .matmul import matmul_prepacked_b
from .matmul import matmul_prepacked_b_affine
from .matmul import pack_a_blocked
from .matmul import pack_a_blocked_t
from .matmul import pack_b_blocked
from .matmul import pack_b_blocked_t
from .matmul import pack_b_blocked_vnni
from .matmul import pack_b_blocked_vnni_t
from .matmul import pack_b_vnni_t
from .matmul import supports
from .matmul import tall_output_tiles
from .normalization import group_norm
from .normalization import softmax
from .reduction import column_sum
from .reduction import matvec
from .reduction import row_dot
from .reduction import row_sum
from .reduction import supports_matvec

__all__ = [
    "CACHE_PREPACKED_WEIGHTS_ENV",
    "AffineLinearCache",
    "LinearCache",
    "bmm",
    "cache_prepacked_weights",
    "column_sum",
    "group_norm",
    "identity_epilogue",
    "leaky_relu",
    "linear",
    "linear_affine",
    "matmul",
    "matmul_prepacked_b",
    "matmul_prepacked_b_affine",
    "matvec",
    "pack_a_blocked",
    "pack_a_blocked_t",
    "pack_b_blocked",
    "pack_b_blocked_t",
    "pack_b_blocked_vnni",
    "pack_b_blocked_vnni_t",
    "pack_b_vnni_t",
    "per_config",
    "relu",
    "row_dot",
    "row_sum",
    "scalar_mul",
    "softmax",
    "supports",
    "supports_matvec",
    "tall_output_tiles",
]
