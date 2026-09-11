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
    configs=[ct.tune.Config({"BLOCK_K": size}) for size in [16, 32, 64, 128, 256]],
    key=["K"],
    grid=lambda meta: (meta["A"].shape[0],),
    options=lambda meta: {"assume_in_bounds": meta["K"] % meta["BLOCK_K"] == 0},
)
@ct.kernel
def _gemv_kernel(
    A,
    B,
    C,
    K: ConstInt,
    stride_am: ConstInt,
    BLOCK_K: ConstInt,
):
    row = ct.bid(0)

    acc = ct.full((BLOCK_K,), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    for k in range(ct.cdiv(K, BLOCK_K)):
        a_vals = ct.load(
            A,
            index=(row, k),
            shape=(1, BLOCK_K),
            padding_mode=zero_pad,
        )
        b_vals = ct.load(
            B,
            index=(k,),
            shape=(BLOCK_K,),
            padding_mode=zero_pad,
        )
        acc += ct.reshape(a_vals, (BLOCK_K,)).astype(ct.float32) * b_vals.astype(
            ct.float32
        )

    result = ct.astype(ct.sum(acc, axis=0), ct.bfloat16)
    ct.store(C, index=(row,), tile=ct.astype(result, C.dtype))


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M, K = A.shape
        C = torch.empty(M, device=A.device, dtype=A.dtype)

        B_flat = B.view(-1).contiguous()

        _gemv_kernel(None, (A, B_flat, C, K, A.stride(0)))
        return C.view(M, 1)
