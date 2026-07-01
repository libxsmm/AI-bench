# ruff: noqa: E731
# Example Triton CPU kernel
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch
import torch.nn as nn

from triton_cpu_utils import sfc_matmul


class Model(nn.Module):
    """KernelBench-compatible wrapper"""

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return sfc_matmul(A, B)
