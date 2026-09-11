# ruff: noqa: E731

import cuda.tile as ct
from cuda.tile._backend import cpu
import torch
import torch.nn as nn

ct.set_backend("cpu")
ConstInt = ct.Constant[int]


@ct.kernel
def _depthwise_conv2d(x, weight, output, C: ConstInt, H: ConstInt, W: ConstInt, OH: ConstInt, OW: ConstInt, KH: ConstInt, KW: ConstInt, BLOCK_SIZE: ConstInt):
    nc = ct.bid(1); pid = ct.bid(0); channel = nc % C
    offsets = pid * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=torch.int32); valid = offsets < OH * OW; oh = offsets // OW; ow = offsets % OW
    acc = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32); x_mem = x.get_raw_memory(); w_mem = weight.get_raw_memory(); max_x = C * H * W * (nc // C + 1) - 1
    for kh in range(KH):
        for kw in range(KW):
            x_indices = ((nc * H + oh + kh) * W + ow + kw); safe_x = ct.minimum(ct.maximum(x_indices, 0), max_x); x_value = x_mem.load_offset(safe_x, mask=safe_x >= 0).astype(ct.float32); w_index = channel * KH * KW + kh * KW + kw; w_value = w_mem.load_offset(w_index + ct.arange(1, dtype=torch.int32), mask=w_index >= 0).astype(ct.float32); acc += x_value * w_value * valid.astype(ct.float32)
    output_indices = nc * OH * OW + offsets; safe_output = ct.minimum(ct.maximum(output_indices, 0), (nc // C + 1) * C * OH * OW - 1); output.get_raw_memory().store_offset(safe_output, ct.astype(acc, output.dtype), mask=valid)


class Model(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__(); self.conv2d = nn.Conv2d(in_channels, in_channels, (kernel_size, 1), stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)

    def forward(self, x):
        x=x.contiguous(); B,C,H,W=x.shape; KH,KW=self.conv2d.kernel_size; OH,OW=H-KH+1,W; weight=self.conv2d.weight.squeeze(1).contiguous().to(x.dtype); output=torch.empty((B,C,OH,OW),device=x.device,dtype=x.dtype)
        with cpu.compile_options({"assume_in_bounds": False}): ct.launch(None,(ct.cdiv(OH*OW,32),B*C),_depthwise_conv2d,(x,weight,output,C,H,W,OH,OW,KH,KW,32))
        return output
