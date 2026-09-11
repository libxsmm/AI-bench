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
def group_norm_stats_kernel(x, mean, invstd, B: ConstInt, C: ConstInt, HW: ConstInt, groups: ConstInt, channels_per_group: ConstInt, eps, BLOCK_HW: ConstInt):
    pid = ct.bid(0)
    batch = pid // groups
    group = pid % groups
    cols = ct.arange(BLOCK_HW, dtype=torch.int32)
    total = 0.0
    total_sq = 0.0
    channel_start = group * channels_per_group
    for channel in range(channels_per_group):
        base = batch * C * HW + (channel_start + channel) * HW
        for block in range(ct.cdiv(HW, BLOCK_HW)):
            offsets = base + block * BLOCK_HW + cols
            valid = block * BLOCK_HW + cols < HW
            values = ct.where(valid, ct.gather(x, offsets), 0.0).astype(ct.float32)
            total += ct.sum(values, axis=0)
            total_sq += ct.sum(values * values, axis=0)
    count = channels_per_group * HW
    average = total / count
    variance = total_sq / count - average * average
    ct.store(mean, index=(pid,), tile=average)
    ct.store(invstd, index=(pid,), tile=1.0 / ct.sqrt(variance + eps))


@ct.kernel
def group_norm_apply_kernel(x, output, mean, invstd, weight, bias, B: ConstInt, C: ConstInt, HW: ConstInt, groups: ConstInt, channels_per_group: ConstInt, BLOCK_HW: ConstInt):
    pid = ct.bid(0)
    batch = pid // C
    channel = pid % C
    group = channel // channels_per_group
    stats = batch * groups + group
    average = ct.load(mean, index=(stats,), shape=())
    inverse = ct.load(invstd, index=(stats,), shape=())
    scale = inverse * ct.load(weight, index=(channel,), shape=()).astype(ct.float32)
    shift = ct.load(bias, index=(channel,), shape=()).astype(ct.float32) - average * scale
    cols = ct.arange(BLOCK_HW, dtype=torch.int32)
    base = batch * C * HW + channel * HW
    for block in range(ct.cdiv(HW, BLOCK_HW)):
        offsets = base + block * BLOCK_HW + cols
        valid = block * BLOCK_HW + cols < HW
        values = ct.where(valid, ct.gather(x, offsets), 0.0).astype(ct.float32)
        ct.scatter(output, offsets, ct.astype(values * scale + shift, ct.bfloat16))


class Model(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super(Model, self).__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        self.num_features = num_features
        self.num_groups = num_groups
        self._packed = False

    def _pack_weights(self, device):
        self.weight_packed = self.gn.weight.data.to(device).contiguous()
        self.bias_packed = self.gn.bias.data.to(device).contiguous()
        self._packed = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        x = x.contiguous()
        if not self._packed:
            self._pack_weights(device)
        B, C, H, W = x.shape
        HW = H * W
        channels_per_group = C // self.num_groups
        output = torch.empty_like(x)
        mean = torch.empty(B * self.num_groups, device=device, dtype=torch.float32)
        invstd = torch.empty(B * self.num_groups, device=device, dtype=torch.float32)
        with cpu.compile_options({"assume_in_bounds": HW % 32 == 0}):
            ct.launch(None, (B * self.num_groups,), group_norm_stats_kernel, (x.view(-1), mean, invstd, B, C, HW, self.num_groups, channels_per_group, self.gn.eps, 32))
            ct.launch(None, (B * C,), group_norm_apply_kernel, (x.view(-1), output.view(-1), mean, invstd, self.weight_packed, self.bias_packed, B, C, HW, self.num_groups, channels_per_group, 32))
        return output
