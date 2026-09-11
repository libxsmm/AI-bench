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
    configs=[ct.tune.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_SIZE_M": group}) for group in [1, 2, 4, 8]],
    key=["M"],
    grid=lambda meta: (ct.cdiv(meta["M"], meta["BLOCK_M"]) * ct.cdiv(meta["M"], meta["BLOCK_N"]),),
    options=lambda meta: {"assume_in_bounds": meta["M"] % meta["BLOCK_M"] == 0 and meta["M"] % meta["BLOCK_N"] == 0 and meta["M"] % meta["BLOCK_K"] == 0},
)
@ct.kernel
def tril_matmul_kernel(A, B, C, M: ConstInt, BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt, GROUP_SIZE_M: ConstInt):
    pid = ct.bid(0)
    num = ct.cdiv(M, BLOCK_M)
    width = GROUP_SIZE_M * num
    group = pid // width
    first = group * GROUP_SIZE_M
    size = min(GROUP_SIZE_M, num - first)
    pid_m = first + ((pid % width) % size)
    pid_n = (pid % width) // size
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    if off_n <= off_m + BLOCK_M - 1:
        start = (off_n // BLOCK_K) * BLOCK_K
        end = min(off_m + BLOCK_M, M)
        acc = ct.full((BLOCK_M, BLOCK_N), 0, dtype=ct.float32)
        for k in range(start, end, BLOCK_K):
            acc = ct.mma(ct.load(A, index=(pid_m, k // BLOCK_K), shape=(BLOCK_M, BLOCK_K)), ct.load(B, index=(k // BLOCK_K, pid_n), shape=(BLOCK_K, BLOCK_N)), acc)
        rows = off_m + ct.arange(BLOCK_M, dtype=torch.int32)
        cols = off_n + ct.arange(BLOCK_N, dtype=torch.int32)
        acc = ct.where(rows[:, None] >= cols[None, :], acc, 0.0)
        ct.store(C, index=(pid_m, pid_n), tile=ct.astype(acc, C.dtype))


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, A, B):
        M = A.shape[0]
        C = torch.zeros((M, M), device=A.device, dtype=A.dtype)
        tril_matmul_kernel(None, (A, B, C, M))
        return C
