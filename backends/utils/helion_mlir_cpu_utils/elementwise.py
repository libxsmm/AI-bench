"""Elementwise Helion MLIR CPU kernels (ReLU, LeakyReLU, scalar multiply)."""

from __future__ import annotations

import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
from helion_mlir_backend import output_pool
import torch
from torch import Tensor


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[8, 1024]))
def _relu_kernel(x: Tensor) -> Tensor:
    out = output_pool.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = torch.relu(x[tile])
    return out


# `negative_slope`/`s` use hl.constexpr so the MLIR backend sees a compile-time
# constant instead of a dynamic SymFloat (which it can't lower).
@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[8, 1024]))
def _leaky_relu_kernel(x: Tensor, negative_slope: hl.constexpr) -> Tensor:
    out = output_pool.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = torch.nn.functional.leaky_relu(
            x[tile], negative_slope=negative_slope
        )
    return out


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[8, 1024]))
def _scalar_mul_kernel(a: Tensor, s: hl.constexpr) -> Tensor:
    out = output_pool.empty_like(a)
    for tile in hl.tile(a.shape):
        out[tile] = a[tile] * s
    return out


def relu(x: Tensor) -> Tensor:
    """Elementwise ``max(x, 0)``."""
    return _relu_kernel(x)


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    """Elementwise LeakyReLU: ``max(x, x * negative_slope)``."""
    return _leaky_relu_kernel(x, hl.constexpr(negative_slope))


def scalar_mul(a: Tensor, s: float | Tensor) -> Tensor:
    """Elementwise ``a * s``."""
    if isinstance(s, Tensor):
        s = float(s)
    return _scalar_mul_kernel(a, hl.constexpr(s))
