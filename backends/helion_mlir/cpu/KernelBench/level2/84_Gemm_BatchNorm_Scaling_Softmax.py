import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
from helion_mlir_backend import output_pool
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear
from helion_mlir_cpu_utils import softmax


def _affine(y, running_mean, running_var, weight, bias, scale, eps, tile_n):
    """``scale * BatchNorm(y)`` with inference statistics, columns ``tile_n``."""
    inv_std = torch.rsqrt(running_var[tile_n].to(torch.float32) + eps)
    shift = bias[tile_n].to(torch.float32) - running_mean[tile_n].to(torch.float32) * (
        inv_std * weight[tile_n].to(torch.float32)
    )
    row = y.to(torch.float32) * (inv_std * weight[tile_n].to(torch.float32)) + shift
    return row * scale


def _batch_norm_scale_softmax(
    y: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
    eps: hl.constexpr,
    whole: hl.constexpr,
    in_place: hl.constexpr = hl.constexpr(False),
) -> torch.Tensor:
    """Row softmax of ``scale * BatchNorm(y)`` (inference statistics) of
    ``[M, N]``; ``scale`` is ``[1]``. Passes over column chunks: row maxima,
    sums of exponentials, then the normalized output. The ``whole`` columns
    of whole chunks reduce lane by lane; a partial last chunk reduces over
    its columns, which masks its padding (lane-wise, a padded lane's
    ``_affine(0)`` would enter the maximum and the sum)."""
    m, n = y.shape
    hl.specialize(n)
    block_n = hl.register_block_size(n)
    # In place, a tile's row statistics come before its store pass, which reads
    # each chunk before writing it.
    if in_place:
        out = y
    else:
        out = output_pool.empty_like(y)
    params = (running_mean, running_var, weight, bias)
    for tile_m in hl.tile(m):
        factor = scale[:].to(torch.float32)
        top = hl.full([tile_m, block_n], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(whole, block_size=block_n):
            row = _affine(y[tile_m, tile_n], *params, factor, eps, tile_n)
            top = torch.maximum(top, row)
        row_top = torch.amax(top, dim=-1, keepdim=True)
        if whole < n:
            for tile_n in hl.tile(whole, n, block_size=block_n):
                row = _affine(y[tile_m, tile_n], *params, factor, eps, tile_n)
                row_top = torch.maximum(row_top, torch.amax(row, -1, True))
        total = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(whole, block_size=block_n):
            row = _affine(y[tile_m, tile_n], *params, factor, eps, tile_n)
            total = total + torch.exp(row - row_top)
        row_total = total.sum(-1, keepdim=True)
        if whole < n:
            for tile_n in hl.tile(whole, n, block_size=block_n):
                row = _affine(y[tile_m, tile_n], *params, factor, eps, tile_n)
                row_total = row_total + torch.exp(row - row_top).sum(-1, keepdim=True)
        inv = 1.0 / row_total
        for tile_n in hl.tile(n, block_size=block_n):
            row = _affine(y[tile_m, tile_n], *params, factor, eps, tile_n)
            out[tile_m, tile_n] = (torch.exp(row - row_top) * inv).to(y.dtype)
    return out


_KERNELS: dict[tuple[int, int], helion.Kernel] = {}


def _kernel(block_n: int, rows: int) -> helion.Kernel:
    # The column block is registered first.
    key = (block_n, rows)
    if key not in _KERNELS:
        _KERNELS[key] = helion.kernel(
            _batch_norm_scale_softmax,
            backend="mlir",
            config=helion.Config(block_sizes=[block_n, rows]),
        )
    return _KERNELS[key]


class Model(nn.Module):
    def __init__(
        self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)
    ):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self._linear_cache = None

    def forward(self, x):
        if self.training:
            raise ValueError("BatchNorm with running statistics needs eval mode")
        if self.scale.numel() != 1:
            raise ValueError(f"expected a scalar scale, got {tuple(self.scale.shape)}")
        y, self._linear_cache = linear(x, self.gemm, cache=self._linear_cache)
        m, n = y.shape
        bn = self.bn
        # Rows per tile: at most 16 and every thread gets a tile; 128-wide chunks.
        rows = max(1, min(16, m // torch.get_num_threads()))
        if n == 1:
            # The chunked kernel does not compile for one column. A softmax
            # over one column is 1 (NaN for a NaN or infinite input), as is
            # that of y itself while the affine map keeps finite values finite.
            return softmax(y, in_place=True)
        block_n = min(n, 128)
        return _kernel(block_n, rows)(
            y,
            bn.running_mean,
            bn.running_var,
            bn.weight,
            bn.bias,
            self.scale.view(1),
            hl.constexpr(bn.eps),
            hl.constexpr(n // block_n * block_n),
            hl.constexpr(True),
        )
