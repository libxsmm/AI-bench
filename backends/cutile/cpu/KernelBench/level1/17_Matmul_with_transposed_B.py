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
    key=["M", "N", "K"],
    grid=lambda meta: (ct.cdiv(meta["M"], meta["BLOCK_M"]) * ct.cdiv(meta["N"], meta["BLOCK_N"]),),
    options=lambda meta: {"assume_in_bounds": meta["M"] % meta["BLOCK_M"] == 0 and meta["N"] % meta["BLOCK_N"] == 0 and meta["K"] % meta["BLOCK_K"] == 0},
)
@ct.kernel
def _matmul_bt_kernel(A, B, C, M: ConstInt, N: ConstInt, K: ConstInt, BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt, GROUP_SIZE_M: ConstInt):
    pid = ct.bid(0)
    num_m = ct.cdiv(M, BLOCK_M)
    num_n = ct.cdiv(N, BLOCK_N)
    width = GROUP_SIZE_M * num_n
    group = pid // width
    first = group * GROUP_SIZE_M
    size = min(GROUP_SIZE_M, num_m - first)
    pid_m = first + ((pid % width) % size)
    pid_n = (pid % width) // size
    acc = ct.full((BLOCK_M, BLOCK_N), 0, dtype=ct.float32)
    for k in range(ct.cdiv(K, BLOCK_K)):
        a = ct.load(A, index=(pid_m, k), shape=(BLOCK_M, BLOCK_K))
        b = ct.permute(ct.load(B, index=(pid_n, k), shape=(BLOCK_N, BLOCK_K)), (1, 0))
        acc = ct.mma(a, b, acc)
    ct.store(C, index=(pid_m, pid_n), tile=ct.astype(acc, C.dtype))


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M, K = A.shape
        N, _ = B.shape
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        _matmul_bt_kernel(None, (A, B, C, M, N, K))
        return C
