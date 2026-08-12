import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Mish, and applies Mish again.
    """

    mlir_pipeline = "mlp"
    pipeline_parameters = "kb_params_level2-29.json"

    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = torch.nn.functional.mish(x)
        x = torch.nn.functional.mish(x)
        return x
