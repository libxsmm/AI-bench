# ruff: noqa: E731, E741
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
def _gemm_kernel(A, B, C, M: ConstInt, N: ConstInt, K: ConstInt, BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt, GROUP_SIZE_M: ConstInt):
    pid = ct.bid(0)
    num_pid_m = ct.cdiv(M, BLOCK_M)
    num_pid_n = ct.cdiv(N, BLOCK_N)
    width = GROUP_SIZE_M * num_pid_n
    group_id = pid // width
    first_pid_m = group_id * GROUP_SIZE_M
    group_size = min(GROUP_SIZE_M, num_pid_m - first_pid_m)
    pid_m = first_pid_m + ((pid % width) % group_size)
    pid_n = (pid % width) // group_size
    acc = ct.full((BLOCK_M, BLOCK_N), 0, dtype=ct.float32)
    for k in range(ct.cdiv(K, BLOCK_K)):
        acc = ct.mma(ct.load(A, index=(pid_m, k), shape=(BLOCK_M, BLOCK_K)), ct.load(B, index=(k, pid_n), shape=(BLOCK_K, BLOCK_N)), acc)
    ct.store(C, index=(pid_m, pid_n), tile=ct.astype(acc, C.dtype))


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, A, B):
        b_dim, i_dim, j_dim, l_dim = A.shape
        k_dim = B.shape[1]
        a = A.contiguous().view(-1, l_dim)
        if A.dtype != torch.bfloat16:
            a = a.to(torch.bfloat16)
        b = B.contiguous()
        if B.dtype != torch.bfloat16:
            b = b.to(torch.bfloat16)
        M, N, K = a.shape[0], k_dim, l_dim
        c = torch.empty((M, N), device=A.device, dtype=torch.bfloat16)
        _gemm_kernel(None, (a, b, c, M, N, K))
        result = c.view(b_dim, i_dim, j_dim, k_dim)
        return result if A.dtype == torch.bfloat16 else result.to(A.dtype)
