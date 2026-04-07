import torch
import torch.nn as nn

import ai_bench.mlir


@torch.compile(
    dynamic=False, backend=ai_bench.mlir.cpu_backend(ai_bench.mlir.cpu_pipeline)
)
class Model(nn.Module):
    """
    Simple model that performs Layer Normalization.
    """

    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return self.ln(x)
