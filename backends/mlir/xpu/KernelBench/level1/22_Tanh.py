import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs a Tanh activation.
    """

    # Currently, there is no dedicated XPU eltwise pipeline.
    mlir_pipeline = "elemwise"
    pipeline_parameters = "kb_params_level1-22.json"

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Tanh activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Tanh applied, same shape as input.
        """
        return torch.tanh(x)
