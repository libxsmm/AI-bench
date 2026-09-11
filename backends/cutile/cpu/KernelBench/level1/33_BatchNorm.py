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


@ct.kernel
def _bn_reduce_kernel(x, partial_sum, partial_sq, B: ConstInt, C: ConstInt, HW: ConstInt, BLOCK_HW: ConstInt):
    channel = ct.bid(0)
    batch = ct.bid(1)
    cols = ct.arange(BLOCK_HW, dtype=torch.int32)
    total = 0.0
    total_sq = 0.0
    base = batch * C * HW + channel * HW
    for block in range(ct.cdiv(HW, BLOCK_HW)):
        offsets = base + block * BLOCK_HW + cols
        valid = block * BLOCK_HW + cols < HW
        values = ct.where(valid, ct.gather(x, offsets), 0.0).astype(ct.float32)
        total += ct.sum(values, axis=0)
        total_sq += ct.sum(values * values, axis=0)
    ct.store(partial_sum, index=(channel * B + batch,), tile=total)
    ct.store(partial_sq, index=(channel * B + batch,), tile=total_sq)


@ct.kernel
def _bn_stats_kernel(partial_sum, partial_sq, scale, shift, weight, bias, B: ConstInt, eps, BLOCK_B: ConstInt):
    channel = ct.bid(0)
    batches = ct.arange(BLOCK_B, dtype=torch.int32)
    valid = batches < B
    values = ct.where(valid, ct.gather(partial_sum, channel * B + batches), 0.0).astype(ct.float32)
    squares = ct.where(valid, ct.gather(partial_sq, channel * B + batches), 0.0).astype(ct.float32)
    total = ct.sum(values, axis=0)
    total_sq = ct.sum(squares, axis=0)
    mean = total / (B * 1.0)
    variance = total_sq / (B * 1.0) - mean * mean
    w = ct.load(weight, index=(channel,), shape=()).astype(ct.float32)
    b = ct.load(bias, index=(channel,), shape=()).astype(ct.float32)
    scale_value = w / ct.sqrt(variance + eps)
    ct.store(scale, index=(channel,), tile=scale_value)
    ct.store(shift, index=(channel,), tile=b - mean * scale_value)


@ct.kernel
def _bn_normalize_kernel(x, output, scale, shift, C: ConstInt, HW: ConstInt, total: ConstInt, BLOCK_SIZE: ConstInt):
    pid = ct.bid(0)
    offsets = pid * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=torch.int32)
    valid = offsets < total
    channels = (offsets // HW) % C
    values = ct.where(valid, ct.gather(x, offsets), 0.0).astype(ct.float32)
    scales = ct.gather(scale, channels)
    shifts = ct.gather(shift, channels)
    ct.scatter(output, offsets, values * scales + shifts)


class Model(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.eps = 1e-5
        self.momentum = 0.1
        self._moved = False
        self._bufs_ready = False

    def _move_params(self, device):
        self.weight.data = self.weight.data.to(device, dtype=torch.float32).contiguous()
        self.bias.data = self.bias.data.to(device, dtype=torch.float32).contiguous()
        self.running_mean = self.running_mean.to(device, dtype=torch.float32).contiguous()
        self.running_var = self.running_var.to(device, dtype=torch.float32).contiguous()
        self._moved = True

    def _alloc_bufs(self, B, C, device):
        self._partial_sum = torch.empty((C, B), device=device, dtype=torch.float32)
        self._partial_sq = torch.empty((C, B), device=device, dtype=torch.float32)
        self._scale = torch.empty(C, device=device, dtype=torch.float32)
        self._shift = torch.empty(C, device=device, dtype=torch.float32)
        self._bufs_ready = True

    def forward(self, x):
        device = x.device
        if not self._moved:
            self._move_params(device)
        x = x.to(dtype=torch.float32).contiguous()
        B, C, H, W = x.shape
        HW = H * W
        total = B * C * HW
        if not self._bufs_ready:
            self._alloc_bufs(B, C, device)
        if self.training:
            with cpu.compile_options({"assume_in_bounds": HW % 8192 == 0}):
                ct.launch(None, (C, B), _bn_reduce_kernel, (x.view(-1), self._partial_sum.view(-1), self._partial_sq.view(-1), B, C, HW, 8192))
            block_b = 1
            while block_b < B:
                block_b *= 2
            with cpu.compile_options({"assume_in_bounds": B % block_b == 0}):
                ct.launch(None, (C,), _bn_stats_kernel, (self._partial_sum.view(-1), self._partial_sq.view(-1), self._scale, self._shift, self.weight, self.bias, B, self.eps, block_b))
        else:
            inv_std = 1.0 / torch.sqrt(self.running_var + self.eps)
            self._scale.copy_(self.weight * inv_std)
            self._shift.copy_(self.bias - self.running_mean * self._scale)
        output = torch.empty_like(x)
        with cpu.compile_options({"assume_in_bounds": total % 32 == 0}):
            ct.launch(None, (ct.cdiv(total, 32),), _bn_normalize_kernel, (x.view(-1), output.view(-1), self._scale, self._shift, C, HW, total, 32))
        return output
