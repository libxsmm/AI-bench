import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
from helion_mlir_backend import output_pool
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[16]))
def _mean_gelu_residual(
    y: torch.Tensor,
    subtract: torch.Tensor,
    x: torch.Tensor,
    in_place: hl.constexpr = hl.constexpr(False),
) -> torch.Tensor:
    """``x + gelu(mean(y - subtract))`` per row of ``y`` ``[M, N]``; the
    logsumexp over the singleton mean dim is the identity. ``in_place`` writes
    into ``y``, of ``x``'s shape: a tile reads its rows before writing them."""
    m, n = y.shape
    _, k = x.shape
    hl.specialize(n)
    hl.specialize(k)
    if in_place:
        out = y
    else:
        out = output_pool.empty_like(x)
    for tile_m in hl.tile(m):
        diff = y[tile_m, :].to(torch.float32) - subtract[:].to(torch.float32)
        mean = diff.mean(-1, keepdim=True)
        gelu = mean * 0.5 * (1.0 + torch.erf(mean * 0.7071067811865476))
        out[tile_m, :] = (x[tile_m, :].to(torch.float32) + gelu).to(x.dtype)
    return out


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self._linear_cache = None

    def forward(self, x):
        y, self._linear_cache = linear(x, self.gemm, cache=self._linear_cache)
        # y is ours: reuse it as the output when the shapes allow.
        in_place = y.shape == x.shape and y.is_contiguous()
        return _mean_gelu_residual(y, self.subtract, x, hl.constexpr(in_place))
