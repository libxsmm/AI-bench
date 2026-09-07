import torch
import torch.nn as nn


class Model(nn.Module):
    mlir_pipeline = "attention"
    pipeline_parameters = "kb_params_level1-97.json"

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        return out
