import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs matrix-vector multiplication (C = A * B).
    """

    mlir_pipeline = "matvec"

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return torch.matmul(A, B)
