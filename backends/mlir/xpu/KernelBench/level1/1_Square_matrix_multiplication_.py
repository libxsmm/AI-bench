import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """

    mlir_pipeline = "matmul"

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        assert all(dim == 4096 for dim in A.shape), "A shape must be 4096"
        assert all(dim == 4096 for dim in B.shape), "B shape must be 4096"

        return torch.matmul(A, B)
