from types import SimpleNamespace

import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import cache_prepacked_weights
from helion_mlir_cpu_utils import linear
from helion_mlir_cpu_utils.matmul import _omp_threads


def _gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))


def _scaled_row_max(y: torch.Tensor, scale: hl.constexpr) -> torch.Tensor:
    """``max(y * scale)`` per row of ``[M, N]``; whole rows per tile, so no
    masked lanes enter the max."""
    m, n = y.shape
    hl.specialize(n)
    out = torch.empty((m,), dtype=y.dtype, device=y.device)
    for tile_m in hl.tile(m):
        out[tile_m] = torch.amax(y[tile_m, :].to(torch.float32) * scale, dim=-1).to(
            y.dtype
        )
    return out


_ROW_MAX_KERNELS: dict[int, helion.Kernel] = {}
# Bound kernels per shape, dtype and scale: binding per call costs about
# as much host time as the kernel.
_ROW_MAX_BOUND: dict[tuple, object] = {}


def _row_max(y: torch.Tensor, scale: float) -> torch.Tensor:
    y = y.contiguous()
    m, n = map(int, y.shape)
    threads = _omp_threads()
    # Rows per tile: every thread a tile, at most 64 Ki elements (L2).
    rows = max(1, min((1 << 16) // n, -(-m // threads)))
    key = (rows, m, n, y.dtype, y.device, scale)
    if (bound := _ROW_MAX_BOUND.get(key)) is None:
        if rows not in _ROW_MAX_KERNELS:
            _ROW_MAX_KERNELS[rows] = helion.kernel(
                _scaled_row_max,
                backend="mlir",
                config=helion.Config(block_sizes=[rows]),
            )
        if len(_ROW_MAX_BOUND) >= 64:
            _ROW_MAX_BOUND.clear()
        bound = _ROW_MAX_KERNELS[rows].bind((y, hl.constexpr(scale)))
        _ROW_MAX_BOUND[key] = bound
    return bound(y, hl.constexpr(scale))


def _parameter_key(p):
    return None if p is None else (p.data_ptr(), p._version, tuple(p.shape), p.dtype)


class Model(nn.Module):
    """``max(scale * GELU(AvgPool_k(x Wᵀ + b)))`` per row.

    The average pool is linear, so it is folded into the weights:
    ``AvgPool_k(x Wᵀ + b) = x W'ᵀ + b'`` with ``W'``, ``b'`` the means of
    each window of ``k`` output rows of ``W`` and ``b`` (the incomplete last
    window is dropped, as AvgPool1d does). The GEMM shrinks by ``k``
    (bench-cpu 1024x8192x8192, k = 16: 1024x512x8192). ``W'`` is computed in
    f32 and rounded once to the input dtype, like a prepacked weight it is
    rebuilt when the parameters change. GELU runs in the GEMM epilogue, the
    scaled row max in a second kernel.

    Fast-math semantics (allowed by the user): pooling the weights first
    reorders the arithmetic. Finite and NaN inputs give the reference's
    results (within rounding; NaN stays NaN), but an infinite input with
    weights of both signs in one pooling window gives +-inf where the
    reference pools +inf and -inf into NaN. There is no guard pass.

    The pooled weights are cached by the parameters' data pointers and
    ``_version`` (in-place ops and ``load_state_dict`` are seen); a change
    through ``.data`` (``p.data.mul_(2)``) changes neither and leaves them
    stale, as for :func:`linear`'s packed weights.

    Config: the pooled GEMM keeps the linear helper's VNNI panels, 64
    columns at 1024 rows on 64 threads (128 on 32, 256 on 56): 8 panels x 8
    row tiles of 128 = 64 tiles for 64 cores, f32 accumulators of 32 KiB, A
    rows (2 MiB per tile) from the LLC. On EMR 64 and 128 columns tie, and 256
    gives too few tiles."""

    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.avg_pool = nn.AvgPool1d(kernel_size=pool_kernel_size)
        self.scale_factor = scale_factor
        self._pooled = None
        self._linear_cache = None

    def _pooled_layer(self, x):
        weight, bias = self.matmul.weight, self.matmul.bias
        k = int(self.avg_pool.kernel_size[0])
        key = (_parameter_key(weight), _parameter_key(bias), k, x.dtype, x.device)
        if self._pooled is not None and self._pooled[0] == key:
            return self._pooled[1]
        n = int(weight.shape[0])
        windows = n // k
        if windows < 1:
            raise ValueError(f"pool window {k} is wider than {n} features")
        with torch.no_grad():
            w = weight.detach()[: windows * k].to(device=x.device, dtype=torch.float32)
            pooled_w = w.view(windows, k, -1).mean(1).to(x.dtype)
            pooled_b = None
            if bias is not None:
                b = bias.detach()[: windows * k].to(
                    device=x.device, dtype=torch.float32
                )
                pooled_b = b.view(windows, k).mean(1).to(x.dtype)
        layer = SimpleNamespace(weight=pooled_w, bias=pooled_b, out_features=windows)
        if cache_prepacked_weights():
            self._pooled = (key, layer)
        return layer

    def forward(self, x):
        pool = self.avg_pool
        if (pool.stride[0], pool.padding[0], pool.ceil_mode) != (
            pool.kernel_size[0],
            0,
            False,
        ):
            raise ValueError("expected AvgPool1d windows that do not overlap or pad")
        layer = self._pooled_layer(x)
        y, self._linear_cache = linear(x, layer, _gelu, self._linear_cache)
        return _row_max(y, float(self.scale_factor))
