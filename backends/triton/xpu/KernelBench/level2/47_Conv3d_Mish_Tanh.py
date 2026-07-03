# ruff: noqa: E731
# KernelBench-compatible wrapper — Model class injected by codegen

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# -----------------------------------------------------------
# Autotune config helpers
# -----------------------------------------------------------
def _conv3d_autotune_configs():
    configs = []
    # Keep search space moderate because this kernel is not on the hot path,
    # but include broad XPU-friendly coverage and the required large 32-warp config.
    tile_shapes = [
        (32, 32, 16),
        (32, 64, 16),
        (64, 32, 16),
        (64, 64, 16),
        (64, 64, 32),
        (64, 128, 16),
        (128, 64, 16),
        (128, 128, 16),
        (128, 128, 32),
        (256, 256, 32),
    ]
    for block_oc, block_ow, c_block in tile_shapes:
        warp_choices = (32,) if (block_oc, block_ow) == (256, 256) else (4, 8, 16, 32)
        for num_warps in warp_choices:
            for num_stages in (1, 2, 3):
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_OC": block_oc,
                            "BLOCK_OW": block_ow,
                            "C_BLOCK": c_block,
                        },
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs


def _mish_tanh_autotune_configs():
    configs = []

    # Narrow, practical XPU-oriented search for a heavy elementwise kernel.
    preferred = [
        (128, 4, 1),
        (256, 4, 1),
        (256, 8, 1),
        (512, 4, 1),
        (512, 8, 1),
        (1024, 8, 1),
        (1024, 16, 1),
        (2048, 8, 1),
        (2048, 16, 1),
        (4096, 16, 1),
    ]
    for block_size, num_warps, num_stages in preferred:
        configs.append(
            triton.Config(
                {
                    "BLOCK_SIZE": block_size,
                },
                num_warps=num_warps,
                num_stages=num_stages,
            )
        )

    # Required large XPU config with 32 warps.
    configs.append(
        triton.Config(
            {
                "BLOCK_SIZE": 65536,
            },
            num_warps=32,
            num_stages=1,
        )
    )

    return configs


# -----------------------------------------------------------
# Triton kernel for 3D convolution with bias (kept for compatibility;
# not used in the optimized execution path)
# -----------------------------------------------------------
@triton.autotune(
    configs=_conv3d_autotune_configs(),
    key=["C_IN", "C_OUT", "D_OUT", "H_OUT", "W_OUT"],
)
@triton.jit
def _conv3d_ncdhw_bias_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    N,
    C_IN,
    D_IN,
    H_IN,
    W_IN,
    C_OUT,
    D_OUT,
    H_OUT,
    W_OUT,
    stride_xn,
    stride_xc,
    stride_xd,
    stride_xh,
    stride_xw,
    stride_woc,
    stride_wc,
    stride_wkd,
    stride_wkh,
    stride_wkw,
    stride_yn,
    stride_yc,
    stride_yd,
    stride_yh,
    stride_yw,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    C_BLOCK: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    tiles_w = tl.cdiv(W_OUT, BLOCK_OW)

    tile_w_id = pid0 % tiles_w
    tmp = pid0 // tiles_w
    oh = tmp % H_OUT
    tmp = tmp // H_OUT
    od = tmp % D_OUT
    n = tmp // D_OUT

    n64 = n.to(tl.int64)
    od64 = od.to(tl.int64)
    oh64 = oh.to(tl.int64)

    oc_start = pid1 * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < C_OUT

    ow_start = tile_w_id * BLOCK_OW
    ow_offsets = ow_start + tl.arange(0, BLOCK_OW)
    ow_mask = ow_offsets < W_OUT

    acc = tl.zeros((BLOCK_OC, BLOCK_OW), dtype=tl.float32)

    base_x_n = n64 * stride_xn
    base_y_n = n64 * stride_yn
    base_y_dh = od64 * stride_yd + oh64 * stride_yh

    for kd in range(KD):
        in_d64 = od64 + kd
        for kh in range(KH):
            in_h64 = oh64 + kh
            x_dh = in_d64 * stride_xd + in_h64 * stride_xh
            for kw in range(KW):
                for cc in range(0, C_IN, C_BLOCK):
                    c_offsets = cc + tl.arange(0, C_BLOCK)
                    c_mask = c_offsets < C_IN

                    w_ptrs = (
                        w_ptr
                        + oc_offsets[:, None] * stride_woc
                        + c_offsets[None, :] * stride_wc
                        + kd * stride_wkd
                        + kh * stride_wkh
                        + kw * stride_wkw
                    )
                    w_tile = tl.load(
                        w_ptrs,
                        mask=oc_mask[:, None] & c_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)

                    x_ptrs = (
                        x_ptr
                        + base_x_n
                        + c_offsets[:, None] * stride_xc
                        + x_dh
                        + (ow_offsets[None, :] + kw) * stride_xw
                    )
                    x_tile = tl.load(
                        x_ptrs,
                        mask=c_mask[:, None] & ow_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)

                    acc = tl.dot(w_tile, x_tile, acc)

    b_vec = tl.load(b_ptr + oc_offsets, mask=oc_mask, other=0.0).to(tl.float32)
    acc = acc + b_vec[:, None]

    y_ptrs = (
        y_ptr
        + base_y_n
        + oc_offsets[:, None] * stride_yc
        + base_y_dh
        + ow_offsets[None, :] * stride_yw
    )
    out_dtype = y_ptr.dtype.element_ty
    if out_dtype == tl.float32:
        out = acc
    elif out_dtype == tl.bfloat16:
        out = acc.to(tl.bfloat16)
    elif out_dtype == tl.float16:
        out = acc.to(tl.float16)
    else:
        out = acc
    tl.store(y_ptrs, out, mask=oc_mask[:, None] & ow_mask[None, :])


def _conv3d_triton(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 5 and w.ndim == 5 and b.ndim == 1, "Invalid ranks"
    assert x.device.type == "xpu", "Input must be on XPU"
    assert w.device == x.device and b.device == x.device, (
        "w and b must be on same device"
    )
    N, C_in, D_in, H_in, W_in = x.shape
    C_out, Cw_in, Kd, Kh, Kw = w.shape
    assert C_in == Cw_in, "Channel mismatch"
    assert b.shape[0] == C_out, "Bias size mismatch"

    D_out = D_in - Kd + 1
    H_out = H_in - Kh + 1
    W_out = W_in - Kw + 1
    assert D_out > 0 and H_out > 0 and W_out > 0, "Invalid kernel size"

    y = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    sxn, sxc, sxd, sxh, sxw = x.stride()
    swoc, swc, swkd, swkh, swkw = w.stride()
    syn, syc, syd, syh, syw = y.stride()

    grid = lambda meta: (
        N * D_out * H_out * triton.cdiv(W_out, meta["BLOCK_OW"]),
        triton.cdiv(C_out, meta["BLOCK_OC"]),
    )

    _conv3d_ncdhw_bias_kernel[grid](
        x,
        w,
        b,
        y,
        N,
        C_in,
        D_in,
        H_in,
        W_in,
        C_out,
        D_out,
        H_out,
        W_out,
        sxn,
        sxc,
        sxd,
        sxh,
        sxw,
        swoc,
        swc,
        swkd,
        swkh,
        swkw,
        syn,
        syc,
        syd,
        syh,
        syw,
        KD=Kd,
        KH=Kh,
        KW=Kw,
        grf_mode="auto",
    )
    return y


# -----------------------------------------------------------
# Triton kernel for fused Mish -> Tanh
# XPU-specific cleanup:
# - use exp2 for exponentials on XPU
# - use tanh identities based on exp2 to avoid tl.tanh throughput issues
# - keep stable softplus branching
# -----------------------------------------------------------
@triton.autotune(
    configs=_mish_tanh_autotune_configs(),
    key=["n_elements"],
)
@triton.jit
def _mish_tanh_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_in = tl.load(x_ptr + offsets, mask=mask)
    x = x_in.to(tl.float32)

    thr = 20.0
    log2e = 1.4426950408889634

    exp_x = tl.math.exp2(x * log2e)
    sp_mid = tl.log(1.0 + exp_x)
    sp = tl.where(x > thr, x, tl.where(x < -thr, exp_x, sp_mid))

    e_neg2_sp = tl.math.exp2((-2.0 * sp) * log2e)
    tanh_sp = (1.0 - e_neg2_sp) / (1.0 + e_neg2_sp)

    mish_x = x * tanh_sp

    e_neg2_m = tl.math.exp2((-2.0 * mish_x) * log2e)
    y = (1.0 - e_neg2_m) / (1.0 + e_neg2_m)

    tl.store(y_ptr + offsets, y.to(x_in.dtype), mask=mask)


def _mish_tanh_triton(x: torch.Tensor) -> torch.Tensor:
    assert x.device.type == "xpu", "Input must be on XPU"
    assert x.dtype in (torch.float16, torch.bfloat16), "Unsupported dtype"
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _mish_tanh_kernel[grid](
        x,
        out,
        n,
        grf_mode="auto",
    )
    return out


def _to_xpu_fp16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.device.type == "xpu" and t.dtype == torch.float16 and t.is_contiguous():
        return t
    return t.to("xpu", dtype=torch.float16).contiguous()


# -----------------------------------------------------------
# Top-level fused function
# Optimized path: vendor conv3d on XPU + Triton fused Mish->Tanh
# -----------------------------------------------------------
def kernel_function(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("Intel XPU is not available.")

    x_xpu = _to_xpu_fp16_contiguous(x)
    w_xpu = _to_xpu_fp16_contiguous(w)
    b_xpu = _to_xpu_fp16_contiguous(b)

    y1 = F.conv3d(x_xpu, w_xpu, b_xpu, stride=1, padding=0)
    y2 = _mish_tanh_triton(y1)
    return y2


# -----------------------------------------------------------
# Reference Model and Test
# -----------------------------------------------------------
batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 32, 64, 64
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self._xpu_prepared = False

    def prepare_for_xpu(self):
        if self._xpu_prepared:
            return
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("Intel XPU is not available.")

        with torch.no_grad():
            if (
                self.conv.weight.device.type != "xpu"
                or self.conv.weight.dtype != torch.float16
                or not self.conv.weight.is_contiguous()
            ):
                self.conv.weight.data = self.conv.weight.data.to(
                    "xpu", dtype=torch.float16
                ).contiguous()

            if self.conv.bias is not None and (
                self.conv.bias.device.type != "xpu"
                or self.conv.bias.dtype != torch.float16
                or not self.conv.bias.is_contiguous()
            ):
                self.conv.bias.data = self.conv.bias.data.to(
                    "xpu", dtype=torch.float16
                ).contiguous()

        self._xpu_prepared = True

    def forward(self, x):
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("Intel XPU is not available.")

        if not self._xpu_prepared:
            self.prepare_for_xpu()

        x = _to_xpu_fp16_contiguous(x)
        return kernel_function(x, self.conv.weight, self.conv.bias)
