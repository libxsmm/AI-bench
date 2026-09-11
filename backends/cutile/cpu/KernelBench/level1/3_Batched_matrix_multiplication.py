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
    grid=lambda meta: ((meta["A"].shape[0] // meta["M"]) * ct.cdiv(meta["M"], meta["BLOCK_M"]) * ct.cdiv(meta["N"], meta["BLOCK_N"]),),
    options=lambda meta: {"assume_in_bounds": meta["M"] % meta["BLOCK_M"] == 0 and meta["N"] % meta["BLOCK_N"] == 0 and meta["K"] % meta["BLOCK_K"] == 0},
)
@ct.kernel
def _batched_matmul_kernel(
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
    num_tiles_mn = num_pid_m * num_pid_n

    batch_id = pid // num_tiles_mn
    tile_id = pid % num_tiles_mn

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m

    batch_offset_m = batch_id * num_pid_m
    batch_offset_k = batch_id * ct.cdiv(K, BLOCK_K)

    acc = ct.full((BLOCK_M, BLOCK_N), 0, dtype=ct.float32)
    for k in range(ct.cdiv(K, BLOCK_K)):
        a = ct.load(
            A,
            index=(batch_offset_m + pid_m, k),
            shape=(BLOCK_M, BLOCK_K),
        )
        b = ct.load(
            B,
            index=(batch_offset_k + k, pid_n),
            shape=(BLOCK_K, BLOCK_N),
        )
        acc = ct.mma(
            a,
            b,
            acc,
        )
    ct.store(
        C,
        index=(batch_offset_m + pid_m, pid_n),
        tile=ct.astype(acc, C.dtype),
    )


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        BATCH, M, K = A.shape
        _, _, N = B.shape

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((BATCH, M, N), device=A.device, dtype=A.dtype)

        _batched_matmul_kernel(None, (A.reshape(BATCH * M, K), B.reshape(BATCH * K, N), C.reshape(BATCH * M, N), M, N, K))
        return C
