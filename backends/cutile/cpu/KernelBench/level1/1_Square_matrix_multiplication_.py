# ruff: noqa: E731
# Example Triton CPU kernel
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


from cutile_cpu_utils import sfc_matmul
import torch
import torch.nn as nn


class Model(nn.Module):
    """KernelBench-compatible wrapper"""

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()
        self._prepared = None

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # if self._prepared is None:
        #     self._prepared = prepare_sfc_matmul(A, B)   
        # return self._prepared()
        return sfc_matmul(A, B, options={"assume_in_bounds": True})
