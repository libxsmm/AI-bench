import cuda.tile as ct
from cuda.tile._backend import cpu
import torch

ct.set_backend("cpu")
ConstInt = ct.Constant[int]


@ct.kernel
def _transpose_conv_stride2_3d_kernel(
    x,
    weight,
    bias,
    output,
    B: ConstInt,
    C_IN: ConstInt,
    C_OUT: ConstInt,
    D: ConstInt,
    H: ConstInt,
    W: ConstInt,
    OD: ConstInt,
    OH: ConstInt,
    OW: ConstInt,
    KD: ConstInt,
    KH: ConstInt,
    KW: ConstInt,
    D_ACT: ConstInt,
    H_ACT: ConstInt,
    W_ACT: ConstInt,
    BLOCK_W: ConstInt,
    BLOCK_N: ConstInt,
):
    pid_w = ct.bid(0)
    pid_bdh = ct.bid(1)
    pid_n = ct.bid(2)
    batch = pid_bdh // (D_ACT * H_ACT)
    rem = pid_bdh % (D_ACT * H_ACT)
    d_idx = rem // H_ACT
    h_idx = rem % H_ACT
    rows = pid_w * BLOCK_W + ct.arange(BLOCK_W, dtype=torch.int32)
    cols = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=torch.int32)
    row_valid = rows < W_ACT
    col_valid = cols < C_OUT
    acc = ct.full((BLOCK_W, BLOCK_N), 0.0, dtype=ct.float32)
    x_mem = x.get_raw_memory()
    weight_mem = weight.get_raw_memory()
    bias_mem = bias.get_raw_memory()
    output_mem = output.get_raw_memory()
    cin = ct.arange(C_IN, dtype=torch.int32)
    max_x_offset = B * D * H * W * C_IN - 1
    max_weight_offset = KD * KH * KW * C_IN * C_OUT - 1

    for kd in range(KD):
        input_d = d_idx + 1 - kd
        valid_d = (input_d >= 0) & (input_d < D)
        for kh in range(KH):
            input_h = h_idx + 1 - kh
            valid_h = valid_d & (input_h >= 0) & (input_h < H)
            for kw in range(KW):
                input_w = rows + 1 - kw
                valid = valid_h & row_valid & (input_w >= 0) & (input_w < W)
                x_indices = ((((batch * D + input_d) * H + input_h) * W + input_w[:, None]) * C_IN + cin[None, :])
                safe_x_indices = ct.minimum(ct.maximum(x_indices, 0), max_x_offset)
                x_values = x_mem.load_offset(safe_x_indices, mask=safe_x_indices >= 0)
                x_values = x_values * valid[:, None].astype(torch.float16)
                kernel_index = kd * KH * KW + kh * KW + kw
                    weight_base = group * C_IN_PG * KD * KH * KW * C_OUT_PG + kernel_index + cin[:, None] * KD * KH * KW * C_OUT_PG + cols[None, :]
                    safe_weight_base = ct.minimum(ct.maximum(weight_base, 0), C_IN * KD * KH * KW * C_OUT_PG - 1)
                weight_values = weight_mem.load_offset(safe_weight_base, mask=col_valid[None, :], padding_value=0.0)
                acc = ct.mma(x_values, weight_values, acc)

    safe_cols = ct.minimum(ct.maximum(cols, 0), C_OUT - 1)
    acc += bias_mem.load_offset(safe_cols, mask=col_valid, padding_value=0.0)[None, :].astype(ct.float32)
    max_output_offset = B * OD * OH * OW * C_OUT - 1
    output_base = ((batch * OD + (2 * d_idx + 1)) * OH + (2 * h_idx + 1)) * OW * C_OUT
    output_value = acc.extract((0, 0), (1, 32)).reshape((32,))
    output_w = pid_w * 16
    output_indices = output_base + (2 * output_w + 1) * C_OUT + cols
    safe_output_indices = ct.minimum(ct.maximum(output_indices, 0), max_output_offset)
    output_mem.store_offset(safe_output_indices, ct.astype(output_value, output.dtype), mask=(output_w < W_ACT) & col_valid)
    output_value = acc.extract((1, 0), (1, 32)).reshape((32,))
    output_w = pid_w * 16 + 1
    output_indices = output_base + (2 * output_w + 1) * C_OUT + cols
    safe_output_indices = ct.minimum(ct.maximum(output_indices, 0), max_output_offset)
    output_mem.store_offset(safe_output_indices, ct.astype(output_value, output.dtype), mask=(output_w < W_ACT) & col_valid)
    output_value = acc.extract((2, 0), (1, 32)).reshape((32,))
    output_w = pid_w * 16 + 2
    output_indices = output_base + (2 * output_w + 1) * C_OUT + cols
    safe_output_indices = ct.minimum(ct.maximum(output_indices, 0), max_output_offset)
    output_mem.store_offset(safe_output_indices, ct.astype(output_value, output.dtype), mask=(output_w < W_ACT) & col_valid)
    output_value = acc.extract((3, 0), (1, 32)).reshape((32,))
    output_w = pid_w * 16 + 3
    output_indices = output_base + (2 * output_w + 1) * C_OUT + cols
    safe_output_indices = ct.minimum(ct.maximum(output_indices, 0), max_output_offset)
    output_mem.store_offset(safe_output_indices, ct.astype(output_value, output.dtype), mask=(output_w < W_ACT) & col_valid)
    output_value = acc.extract((4, 0), (1, 32)).reshape((32,))
    output_w = pid_w * 16 + 4
    output_indices = output_base + (2 * output_w + 1) * C_OUT + cols
    safe_output_indices = ct.minimum(ct.maximum(output_indices, 0), max_output_offset)
    output_mem.store_offset(safe_output_indices, ct.astype(output_value, output.dtype), mask=(output_w < W_ACT) & col_valid)
    output_value = acc.extract((5, 0), (1, 32)).reshape((32,))
    output_w = pid_w * 16 + 5
    output_indices = output_base + (2 * output_w + 1) * C_OUT + cols
    safe_output_indices = ct.minimum(ct.maximum(output_indices, 0), max_output_offset)
    output_mem.store_offset(safe_output_indices, ct.astype(output_value, output.dtype), mask=(output_w < W_ACT) & col_valid)
    output_value = acc.extract((6, 0), (1, 32)).reshape((32,))
    output_w = pid_w * 16 + 6
    output_indices = output_base + (2 * output_w + 1) * C_OUT + cols
    safe_output_indices = ct.minimum(ct.maximum(output_indices, 0), max_output_offset)
    output_mem.store_offset(safe_output_indices, ct.astype(output_value, output.dtype), mask=(output_w < W_ACT) & col_valid)
    output_value = acc.extract((7, 0), (1, 32)).reshape((32,))
    output_w = pid_w * 16 + 7
    output_indices = output_base + (2 * output_w + 1) * C_OUT + cols
    safe_output_indices = ct.minimum(ct.maximum(output_indices, 0), max_output_offset)
    output_mem.store_offset(safe_output_indices, ct.astype(output_value, output.dtype), mask=(output_w < W_ACT) & col_valid)
    output_value = acc.extract((8, 0), (1, 32)).reshape((32,))
    output_w = pid_w * 16 + 8
    output_indices = output_base + (2 * output_w + 1) * C_OUT + cols
    safe_output_indices = ct.minimum(ct.maximum(output_indices, 0), max_output_offset)
    output_mem.store_offset(safe_output_indices, ct.astype(output_value, output.dtype), mask=(output_w < W_ACT) & col_valid)
    output_value = acc.extract((9, 0), (1, 32)).reshape((32,))
    output_w = pid_w * 16 + 9
    output_indices = output_base + (2 * output_w + 1) * C_OUT + cols
    safe_output_indices = ct.minimum(ct.maximum(output_indices, 0), max_output_offset)
    output_mem.store_offset(safe_output_indices, ct.astype(output_value, output.dtype), mask=(output_w < W_ACT) & col_valid)
    output_value = acc.extract((10, 0), (1, 32)).reshape((32,))
    output_w = pid_w * 16 + 10
    output_indices = output_base + (2 * output_w + 1) * C_OUT + cols
    safe_output_indices = ct.minimum(ct.maximum(output_indices, 0), max_output_offset)
    output_mem.store_offset(safe_output_indices, ct.astype(output_value, output.dtype), mask=(output_w < W_ACT) & col_valid)
    output_value = acc.extract((11, 0), (1, 32)).reshape((32,))
    output_w = pid_w * 16 + 11
    output_indices = output_base + (2 * output_w + 1) * C_OUT + cols
    safe_output_indices = ct.minimum(ct.maximum(output_indices, 0), max_output_offset)
    output_mem.store_offset(safe_output_indices, ct.astype(output_value, output.dtype), mask=(output_w < W_ACT) & col_valid)
    output_value = acc.extract((12, 0), (1, 32)).reshape((32,))
    output_w = pid_w * 16 + 12
    output_indices = output_base + (2 * output_w + 1) * C_OUT + cols
    safe_output_indices = ct.minimum(ct.maximum(output_indices, 0), max_output_offset)
    output_mem.store_offset(safe_output_indices, ct.astype(output_value, output.dtype), mask=(output_w < W_ACT) & col_valid)
    output_value = acc.extract((13, 0), (1, 32)).reshape((32,))
    output_w = pid_w * 16 + 13
    output_indices = output_base + (2 * output_w + 1) * C_OUT + cols
    safe_output_indices = ct.minimum(ct.maximum(output_indices, 0), max_output_offset)
    output_mem.store_offset(safe_output_indices, ct.astype(output_value, output.dtype), mask=(output_w < W_ACT) & col_valid)
    output_value = acc.extract((14, 0), (1, 32)).reshape((32,))
    output_w = pid_w * 16 + 14
    output_indices = output_base + (2 * output_w + 1) * C_OUT + cols
    safe_output_indices = ct.minimum(ct.maximum(output_indices, 0), max_output_offset)
    output_mem.store_offset(safe_output_indices, ct.astype(output_value, output.dtype), mask=(output_w < W_ACT) & col_valid)
    output_value = acc.extract((15, 0), (1, 32)).reshape((32,))
    output_w = pid_w * 16 + 15
    output_indices = output_base + (2 * output_w + 1) * C_OUT + cols
    safe_output_indices = ct.minimum(ct.maximum(output_indices, 0), max_output_offset)
    output_mem.store_offset(safe_output_indices, ct.astype(output_value, output.dtype), mask=(output_w < W_ACT) & col_valid)


@ct.kernel
def _transpose_conv_kernel(
    x,
    weight,
    bias,
    output,
    M: ConstInt,
    C_IN: ConstInt,
    C_OUT: ConstInt,
    C_IN_PG: ConstInt,
    C_OUT_PG: ConstInt,
    GROUPS: ConstInt,
    D: ConstInt,
    H: ConstInt,
    W: ConstInt,
    OD: ConstInt,
    OH: ConstInt,
    OW: ConstInt,
    KD: ConstInt,
    KH: ConstInt,
    KW: ConstInt,
    stride_d: ConstInt,
    stride_h: ConstInt,
    stride_w: ConstInt,
    pad_d: ConstInt,
    pad_h: ConstInt,
    pad_w: ConstInt,
    dilation_d: ConstInt,
    dilation_h: ConstInt,
    dilation_w: ConstInt,
    DIM: ConstInt,
    BLOCK_M: ConstInt,
    BLOCK_N: ConstInt,
    BLOCK_K: ConstInt,
):
    pid = ct.bid(0)
    # Blocks are tiled per group so that `group` stays a scalar and never leaks into the M axis.
    num_n = ct.cdiv(C_OUT_PG, BLOCK_N)
    pid_m = pid // (num_n * GROUPS)
    rest = pid % (num_n * GROUPS)
    group = rest // num_n
    pid_n = rest % num_n
    rows = pid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=torch.int32)
    output_local = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=torch.int32)
    cols = group * C_OUT_PG + output_local
    row_valid = rows < M
    col_valid = output_local < C_OUT_PG
    spatial = OD * OH * OW
    max_x_offset = C_IN * D * H * W * (M // spatial) - 1
    batch = rows // spatial
    rem = rows % spatial
    od = rem // (OH * OW)
    rem = rem % (OH * OW)
    oh = rem // OW
    ow = rem % OW
    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
    x_mem = x.get_raw_memory()
    weight_mem = weight.get_raw_memory()
    bias_mem = bias.get_raw_memory()
    k_offsets = ct.arange(BLOCK_K, dtype=torch.int32)
    for block in range(ct.cdiv(C_IN_PG, BLOCK_K)):
        cin_local = block * BLOCK_K + k_offsets
        valid_c = cin_local < C_IN_PG
        if DIM == 1:
            x_base = batch[:, None] * C_IN * W + (group * C_IN_PG + cin_local[None, :]) * W
        elif DIM == 2:
            x_base = batch[:, None] * C_IN * H * W + (group * C_IN_PG + cin_local[None, :]) * H * W
        else:
            x_base = batch[:, None] * C_IN * D * H * W + (group * C_IN_PG + cin_local[None, :]) * D * H * W
        for kd in range(KD):
            if DIM == 1:
                valid_d = row_valid[:, None]
                base_x = x_base
            elif DIM == 2:
                valid_d = row_valid[:, None]
                base_x = x_base
            else:
                input_d_num = od[:, None] + pad_d - kd * dilation_d
                input_d = input_d_num // stride_d
                valid_d = row_valid[:, None] & (input_d_num % stride_d == 0) & (input_d_num >= 0) & (input_d < D)
                base_x = x_base + input_d * H * W
            for kh in range(KH):
                if DIM == 1:
                    valid_h = row_valid[:, None]
                    base_xy = base_x
                elif DIM == 2:
                    input_h_num = oh[:, None] + pad_h - kh * dilation_h
                    input_h = input_h_num // stride_h
                    valid_h = row_valid[:, None] & (input_h_num % stride_h == 0) & (input_h_num >= 0) & (input_h < H)
                    base_xy = x_base + input_h * W
                else:
                    input_h_num = oh[:, None] + pad_h - kh * dilation_h
                    input_h = input_h_num // stride_h
                    valid_h = valid_d & (input_h_num % stride_h == 0) & (input_h_num >= 0) & (input_h < H)
                    base_xy = base_x + input_h * W
                for kw in range(KW):
                    input_w_num = ow[:, None] + pad_w - kw * dilation_w
                    input_w = input_w_num // stride_w
                    valid = valid_h & (input_w_num % stride_w == 0) & (input_w_num >= 0) & (input_w < W) & valid_c[None, :]
                    x_indices = base_xy + input_w
                    safe_x_indices = ct.minimum(ct.maximum(x_indices, 0), max_x_offset)
                    x_values = x_mem.load_offset(safe_x_indices, mask=safe_x_indices >= 0).astype(ct.float32) * valid.astype(ct.float32)
                    kernel_index = kd * KH * KW + kh * KW + kw
                    # mma computes x @ y, so the weight tile must be laid out as (BLOCK_K, BLOCK_N).
                    weight_base = (group * KD * KH * KW * C_IN_PG * C_OUT_PG + kernel_index * C_IN_PG * C_OUT_PG + cin_local[:, None] * C_OUT_PG + output_local[None, :])
                    weight_values = weight_mem.load_offset(weight_base, mask=valid_c[:, None] & col_valid[None, :], padding_value=0.0).astype(ct.float32)
                    acc = ct.mma(x_values, weight_values, acc)
    if bias is not None:
        acc += bias_mem.load_offset(cols, mask=col_valid, padding_value=0.0)[None, :].astype(ct.float32)
    if DIM == 1:
        output_indices = batch[:, None] * C_OUT * OW + cols[None, :] * OW + ow[:, None]
    elif DIM == 2:
        output_indices = batch[:, None] * C_OUT * OH * OW + cols[None, :] * OH * OW + oh[:, None] * OW + ow[:, None]
    else:
        output_indices = batch[:, None] * C_OUT * OD * OH * OW + cols[None, :] * OD * OH * OW + od[:, None] * OH * OW + oh[:, None] * OW + ow[:, None]
    valid_output = row_valid[:, None] & col_valid[None, :]
    ct.scatter(output, output_indices, ct.astype(acc, output.dtype), mask=valid_output)


def transpose_conv(x, conv, dim):
    x = x.contiguous()
    weight = conv.weight
    if dim == 1:
        B, C_IN, W = x.shape
        D = H = OD = OH = 1
        KD = KH = 1
        KW = weight.shape[2]
        stride = conv.stride[0]
        padding = conv.padding[0]
        dilation = conv.dilation[0]
        OW = (W - 1) * stride - 2 * padding + dilation * (KW - 1) + 1 + conv.output_padding[0]
    elif dim == 2:
        B, C_IN, H, W = x.shape
        D = OD = KD = 1
        KH, KW = weight.shape[-2:]
        stride_h, stride_w = conv.stride
        pad_h, pad_w = conv.padding
        dilation_h, dilation_w = conv.dilation
        OH = (H - 1) * stride_h - 2 * pad_h + dilation_h * (KH - 1) + 1 + conv.output_padding[0]
        OW = (W - 1) * stride_w - 2 * pad_w + dilation_w * (KW - 1) + 1 + conv.output_padding[1]
    else:
        B, C_IN, D, H, W = x.shape
        KD, KH, KW = weight.shape[-3:]
        stride_d, stride_h, stride_w = conv.stride
        pad_d, pad_h, pad_w = conv.padding
        dilation_d, dilation_h, dilation_w = conv.dilation
        OD = (D - 1) * stride_d - 2 * pad_d + dilation_d * (KD - 1) + 1 + conv.output_padding[0]
        OH = (H - 1) * stride_h - 2 * pad_h + dilation_h * (KH - 1) + 1 + conv.output_padding[1]
        OW = (W - 1) * stride_w - 2 * pad_w + dilation_w * (KW - 1) + 1 + conv.output_padding[2]
    groups = conv.groups
    C_OUT = conv.out_channels
    C_IN_PG = C_IN // groups
    C_OUT_PG = C_OUT // groups
    if (
        dim == 3
        and x.dtype == torch.float16
        and groups == 1
        and conv.stride == (2, 2, 2)
        and conv.padding == (1, 1, 1)
        and conv.dilation == (2, 2, 2)
        and conv.output_padding == (0, 0, 0)
    ):
        x_channels_last = x.contiguous(memory_format=torch.channels_last_3d)
        packed = weight.permute(2, 3, 4, 0, 1).contiguous().to(dtype=torch.float16)
        output = torch.zeros((B, C_OUT, OD, OH, OW), device=x.device, dtype=x.dtype).contiguous(memory_format=torch.channels_last_3d)
        bias = torch.zeros(C_OUT, device=x.device, dtype=x.dtype) if conv.bias is None else conv.bias.contiguous().to(dtype=x.dtype)
        D_ACT = OD // 2
        H_ACT = OH // 2
        W_ACT = OW // 2
        with cpu.compile_options({"assume_in_bounds": False}):
            ct.launch(None, (ct.cdiv(W_ACT, 16), B * D_ACT * H_ACT, ct.cdiv(C_OUT, 32),), _transpose_conv_stride2_3d_kernel, (x_channels_last, packed, bias, output, B, C_IN, C_OUT, D, H, W, OD, OH, OW, KD, KH, KW, D_ACT, H_ACT, W_ACT, 16, 32))
        return output
    packed = weight.reshape(groups, C_IN_PG, C_OUT_PG, *weight.shape[2:]).permute(0, *range(3, 3 + dim), 1, 2).contiguous().reshape(groups, KD * KH * KW * C_IN_PG * C_OUT_PG).contiguous()
    output = torch.empty((B, C_OUT, OD, OH, OW), device=x.device, dtype=x.dtype)
    M = B * OD * OH * OW
    bias = torch.zeros(C_OUT, device=x.device, dtype=x.dtype) if conv.bias is None else conv.bias.contiguous()
    with cpu.compile_options({"assume_in_bounds": False}):
        ct.launch(None, (ct.cdiv(M, 32) * ct.cdiv(C_OUT_PG, 32) * groups,), _transpose_conv_kernel, (x.view(-1), packed.view(-1), bias, output.view(-1), M, C_IN, C_OUT, C_IN_PG, C_OUT_PG, groups, D, H, W, OD, OH, OW, KD, KH, KW, stride_d if dim == 3 else 1, stride_h if dim > 1 else 1, stride if dim == 1 else stride_w, pad_d if dim == 3 else 0, pad_h if dim > 1 else 0, padding if dim == 1 else pad_w, dilation_d if dim == 3 else 1, dilation_h if dim > 1 else 1, dilation if dim == 1 else dilation_w, dim, 32, 32, 32))
    return output
