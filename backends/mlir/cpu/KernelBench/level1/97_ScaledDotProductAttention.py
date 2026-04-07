import torch
import torch.nn as nn

import ai_bench.mlir


@torch.compile(
    dynamic=False, backend=ai_bench.mlir.cpu_backend(ai_bench.mlir.cpu_pipeline)
)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        return out
