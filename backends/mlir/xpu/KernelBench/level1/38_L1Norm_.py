import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs L1 normalization.
    """

    mlir_pipeline = "reduction"
    pipeline_parameters = "kb_params_level1-38.json"

    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        return x / torch.mean(torch.abs(x), dim=1, keepdim=True)
