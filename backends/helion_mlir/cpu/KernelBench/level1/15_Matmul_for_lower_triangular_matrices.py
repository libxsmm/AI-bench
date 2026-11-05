import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import matmul


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[16, 1024]))
def _tril_(x: torch.Tensor) -> torch.Tensor:
    """Zero the part of ``x`` above the diagonal, in place: a Helion kernel, not
    ``torch.tril``, which writes a second 4096^2 output. Also no torch op
    between kernels: with torch on libgomp and the kernels on libomp,
    ``torch.tril``'s threads compete with libomp's still-spinning workers."""
    m, n = x.shape
    for tile_m, tile_n in hl.tile([m, n]):
        above = tile_n.index[None, :] > tile_m.index[:, None]
        x[tile_m, tile_n] = torch.where(above, 0.0, x[tile_m, tile_n])
    return x


class Model(nn.Module):
    """KernelBench-compatible wrapper"""

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _tril_(matmul(A, B))
