# ruff: noqa: E731
# Example Triton CPU kernel
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


from sfc_matmul import sfc_matmul
import torch
import torch.nn as nn


class Model(nn.Module):
    """KernelBench-compatible wrapper"""

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return sfc_matmul(A, B)
