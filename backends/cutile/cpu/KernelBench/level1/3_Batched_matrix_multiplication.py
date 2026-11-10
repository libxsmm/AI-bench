# ruff: noqa: E731
# Example CUDA Tile CPU kernel
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative

import cuda.tile as ct
import torch
import torch.nn as nn

ct.set_backend("cpu")

ConstInt = ct.Constant[int]


@ct.autotune(
    configs=[
        ct.tune.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_SIZE_M": group}
        )
        for group in [1, 2, 4, 8]
    ],
    key=["M", "N", "K"],
    grid=lambda meta: (
        meta["A"].shape[0]
        * ct.cdiv(meta["M"], meta["BLOCK_M"])
        * ct.cdiv(meta["N"], meta["BLOCK_N"]),
    ),
    options=lambda meta: {
        "assume_in_bounds": meta["M"] % meta["BLOCK_M"] == 0
        and meta["N"] % meta["BLOCK_N"] == 0
        and meta["K"] % meta["BLOCK_K"] == 0
    },
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

    acc = ct.full((BLOCK_M, BLOCK_N), 0, dtype=ct.float32)
    for k in range(ct.cdiv(K, BLOCK_K)):
        # 3D tiles keep the batch index loop-invariant, so the K loop has no
        # index math and ct.mma lowers to AMX on CPU.
        a = ct.load(
            A,
            index=(batch_id, pid_m, k),
            shape=(1, BLOCK_M, BLOCK_K),
        )
        b = ct.load(
            B,
            index=(batch_id, k, pid_n),
            shape=(1, BLOCK_K, BLOCK_N),
        )
        acc = ct.mma(
            ct.reshape(a, (BLOCK_M, BLOCK_K)),
            ct.reshape(b, (BLOCK_K, BLOCK_N)),
            acc,
        )
    ct.store(
        C,
        index=(batch_id, pid_m, pid_n),
        tile=ct.reshape(ct.astype(acc, C.dtype), (1, BLOCK_M, BLOCK_N)),
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

        _batched_matmul_kernel(
            None,
            (
                A,
                B,
                C,
                M,
                N,
                K,
            ),
        )
        return C
