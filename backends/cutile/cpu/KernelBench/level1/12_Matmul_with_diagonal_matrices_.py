# ruff: noqa: E731
# Example CUDA Tile CPU kernel
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative

import cuda.tile as ct
from cuda.tile._backend import cpu
import torch
import torch.nn as nn

ct.set_backend("cpu")
ConstInt = ct.Constant[int]


@ct.autotune(
    configs=[ct.tune.Config({"BLOCK_N": size, "BLOCK_M": size}) for size in [32, 64]],
    key=["N", "M"],
    grid=lambda meta: (ct.cdiv(meta["N"], meta["BLOCK_N"]), ct.cdiv(meta["M"], meta["BLOCK_M"])),
    options=lambda meta: {"assume_in_bounds": meta["N"] % meta["BLOCK_N"] == 0 and meta["M"] % meta["BLOCK_M"] == 0},
)
@ct.kernel
def _diag_matmul_kernel(A, B, C, N: ConstInt, M: ConstInt, BLOCK_N: ConstInt, BLOCK_M: ConstInt):
    pid_n = ct.bid(0)
    pid_m = ct.bid(1)
    offs_n = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=torch.int32)
    offs_m = pid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=torch.int32)
    a = ct.gather(A, offs_n)
    b = ct.gather(B, (offs_n[:, None], offs_m[None, :]))
    result = ct.reshape(a, (BLOCK_N, 1)).astype(ct.float32) * b.astype(ct.float32)
    ct.scatter(C, (offs_n[:, None], offs_m[None, :]), ct.astype(result, C.dtype))


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, A, B):
        N = A.shape[0]
        M = B.shape[1]
        C = torch.empty((N, M), device=A.device, dtype=A.dtype)
        BLOCK_N = BLOCK_M = 64
        _diag_matmul_kernel(None, (A, B, C, N, M))
        return C
