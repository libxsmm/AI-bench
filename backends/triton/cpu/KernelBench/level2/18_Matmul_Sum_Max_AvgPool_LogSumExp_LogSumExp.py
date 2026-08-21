# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch.nn as nn

from triton_cpu_utils import SFCMatmulHelper


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self._matmul_helper = None

    def forward(self, x):
        if self._matmul_helper is None:
            self._matmul_helper = SFCMatmulHelper(
                self.linear.weight.data.to(dtype=x.dtype),
                self.linear.bias.data.to(dtype=x.dtype),
            )

        return self._matmul_helper(
            x,
            reduce_last_dim=True,
            keep_dim=True,
        )
