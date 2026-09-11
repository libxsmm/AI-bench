# ruff: noqa: E731, E741
# Example CUDA Tile CPU kernel
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative

import math

import cuda.tile as ct
from cuda.tile._backend import cpu
import torch
import torch.nn as nn

ct.set_backend("cpu")
ConstInt = ct.Constant[int]


@ct.autotune(
    configs=[ct.tune.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32})],
    key=["SEQ_LEN", "HEAD_DIM"],
    grid=lambda meta: (meta["BH"], ct.cdiv(meta["SEQ_LEN"], meta["BLOCK_M"]) * ct.cdiv(meta["SEQ_LEN"], meta["BLOCK_N"])),
    options={"assume_in_bounds": False},
)
@ct.kernel
def qk_gemm_kernel(Q, Kt, S, BH: ConstInt, SEQ_LEN: ConstInt, HEAD_DIM: ConstInt, scale, BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt):
    batch = ct.bid(0)
    tile = ct.bid(1)
    num_n = ct.cdiv(SEQ_LEN, BLOCK_N)
    pid_m = tile // num_n
    pid_n = tile % num_n
    rows = pid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=torch.int32)
    cols = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=torch.int32)
    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
    Q_mem = Q.get_raw_memory()
    Kt_mem = Kt.get_raw_memory()
    for block in range(ct.cdiv(HEAD_DIM, BLOCK_K)):
        q_offsets = batch * SEQ_LEN * HEAD_DIM + rows[:, None] * HEAD_DIM + block * BLOCK_K + ct.arange(BLOCK_K, dtype=torch.int32)[None, :]
        k_offsets = batch * HEAD_DIM * SEQ_LEN + block * BLOCK_K * SEQ_LEN + ct.arange(BLOCK_K, dtype=torch.int32)[:, None] * SEQ_LEN + cols[None, :]
        q_valid = (rows[:, None] < SEQ_LEN) & (block * BLOCK_K + ct.arange(BLOCK_K, dtype=torch.int32)[None, :] < HEAD_DIM)
        k_valid = (block * BLOCK_K + ct.arange(BLOCK_K, dtype=torch.int32)[:, None] < HEAD_DIM) & (cols[None, :] < SEQ_LEN)
        q_safe_offsets = ct.minimum(ct.maximum(q_offsets, 0), BH * SEQ_LEN * HEAD_DIM - 1)
        k_safe_offsets = ct.minimum(ct.maximum(k_offsets, 0), BH * HEAD_DIM * SEQ_LEN - 1)
        q = Q_mem.load_offset(q_safe_offsets, mask=q_valid, padding_value=0.0).astype(ct.float32)
        k = Kt_mem.load_offset(k_safe_offsets, mask=k_valid, padding_value=0.0).astype(ct.float32)
        acc = ct.mma(q, k, acc)
    out_offsets = batch * SEQ_LEN * SEQ_LEN + rows[:, None] * SEQ_LEN + cols[None, :]
    out_valid = (rows[:, None] < SEQ_LEN) & (cols[None, :] < SEQ_LEN)
    ct.scatter(S, out_offsets, acc * scale, mask=out_valid)


@ct.autotune(
    configs=[ct.tune.Config({"BLOCK_M": 32, "BLOCK_K": 32, "BLOCK_N": 128})],
    key=["SEQ_LEN", "HEAD_DIM"],
    grid=lambda meta: (meta["BH"], ct.cdiv(meta["SEQ_LEN"], meta["BLOCK_M"]) * ct.cdiv(meta["HEAD_DIM"], meta["BLOCK_N"])),
    options={"assume_in_bounds": False},
)
@ct.kernel
def softmax_pv_kernel(S, V, O, BH: ConstInt, SEQ_LEN: ConstInt, HEAD_DIM: ConstInt, BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt):
    batch = ct.bid(0)
    tile = ct.bid(1)
    num_n = ct.cdiv(HEAD_DIM, BLOCK_N)
    pid_m = tile // num_n
    pid_n = tile % num_n
    rows = pid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=torch.int32)
    cols = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=torch.int32)
    m_i = ct.full((BLOCK_M,), float("-inf"), dtype=ct.float32)
    l_i = ct.full((BLOCK_M,), 0.0, dtype=ct.float32)
    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
    S_mem = S.get_raw_memory()
    V_mem = V.get_raw_memory()
    for block in range(ct.cdiv(SEQ_LEN, BLOCK_K)):
        keys = block * BLOCK_K + ct.arange(BLOCK_K, dtype=torch.int32)
        s_offsets = batch * SEQ_LEN * SEQ_LEN + rows[:, None] * SEQ_LEN + keys[None, :]
        s_valid = (rows[:, None] < SEQ_LEN) & (keys[None, :] < SEQ_LEN)
        s_safe_offsets = ct.minimum(ct.maximum(s_offsets, 0), BH * SEQ_LEN * SEQ_LEN - 1)
        scores = S_mem.load_offset(s_safe_offsets, mask=s_valid, padding_value=float("-inf")).astype(ct.float32)
        chunk_max = ct.max(scores, axis=1)
        new_max = ct.maximum(m_i, chunk_max)
        alpha = ct.exp2((m_i - new_max) * 1.4426950408889634)
        exp_scores = ct.exp2((scores - new_max[:, None]) * 1.4426950408889634)
        l_i = alpha * l_i + ct.sum(exp_scores, axis=1)
        acc = acc * alpha[:, None]
        v_offsets = batch * SEQ_LEN * HEAD_DIM + keys[:, None] * HEAD_DIM + cols[None, :]
        v_valid = (keys[:, None] < SEQ_LEN) & (cols[None, :] < HEAD_DIM)
        v_safe_offsets = ct.minimum(ct.maximum(v_offsets, 0), BH * SEQ_LEN * HEAD_DIM - 1)
        values = V_mem.load_offset(v_safe_offsets, mask=v_valid, padding_value=0.0).astype(ct.float32)
        acc = ct.mma(exp_scores, values, acc)
        m_i = new_max
    result = acc / l_i[:, None]
    o_offsets = batch * SEQ_LEN * HEAD_DIM + rows[:, None] * HEAD_DIM + cols[None, :]
    o_valid = (rows[:, None] < SEQ_LEN) & (cols[None, :] < HEAD_DIM)
    ct.scatter(O, o_offsets, result.astype(O.dtype), mask=o_valid)


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._s_buf = None
        self._o_buf = None

    def forward(self, Q, K, V):
        B, H, S, D = Q.shape
        BH = B * H
        scale = 1.0 / math.sqrt(D)
        Q = Q.reshape(BH, S, D).contiguous()
        Kt = K.reshape(BH, S, D).transpose(1, 2).contiguous()
        V = V.reshape(BH, S, D).contiguous()
        S_mat = torch.empty((BH, S, S), device=Q.device, dtype=torch.float32)
        output = torch.empty((BH, S, D), device=Q.device, dtype=Q.dtype)
        qk_gemm_kernel(None, (Q.view(-1), Kt.view(-1), S_mat.view(-1), BH, S, D, scale))
        softmax_pv_kernel(None, (S_mat.view(-1), V.view(-1), output.view(-1), BH, S, D))
        return output.reshape(B, H, S, D)
