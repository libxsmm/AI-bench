import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import matmul
from helion_mlir_cpu_utils import tall_output_tiles


class Model(nn.Module):
    """``einsum("bijl,lk->bijk")`` as one ``[b*i*j, l] @ [l, k]`` GEMM, of whole
    rows per tile where the shape allows: its rows are many and short."""

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        b, i, j, inner = A.shape
        rows, outer = b * i * j, B.shape[1]
        out = matmul(
            A.reshape(rows, inner),
            B,
            block_sizes=tall_output_tiles(rows, outer, inner),
        )
        return out.view(b, i, j, outer)
