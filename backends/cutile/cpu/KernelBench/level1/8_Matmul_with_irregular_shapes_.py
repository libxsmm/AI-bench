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
def matmul_kernel(
    A,
    B,
    C,
    M: ConstInt,
    N: ConstInt,
    K: ConstInt,
    BLOCK_M: ConstInt,
    BLOCK_N: ConstInt,
    BLOCK_K: ConstInt,
    GROUP_SIZE_M: ConstInt,
):
    pid = ct.bid(0)
    num_pid_m = ct.cdiv(M, BLOCK_M)
    num_pid_n = ct.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    acc = ct.full((BLOCK_M, BLOCK_N), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    for k in range(ct.cdiv(K, BLOCK_K)):
        a = ct.load(
            A,
            index=(pid_m, k),
            shape=(BLOCK_M, BLOCK_K),
            padding_mode=zero_pad,
        )
        b = ct.load(
            B,
            index=(k, pid_n),
            shape=(BLOCK_K, BLOCK_N),
            padding_mode=zero_pad,
        )
        acc = ct.mma(a, b, acc)

    ct.store(C, index=(pid_m, pid_n), tile=ct.astype(acc, C.dtype))


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.contiguous()
        B = B.contiguous()
        M, K = A.shape
        _, N = B.shape
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        matmul_kernel(None, (A, B, C, M, N, K))
        return C
