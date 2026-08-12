import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Swish activation, and scales the result.
    """

    mlir_pipeline = "mlp"
    pipeline_parameters = "kb_params_level2-59.json"

    def __init__(self, in_features, out_features, scaling_factor):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.matmul(x)
        x = x * torch.sigmoid(x)  # Swish activation
        x = x * self.scaling_factor
        return x
