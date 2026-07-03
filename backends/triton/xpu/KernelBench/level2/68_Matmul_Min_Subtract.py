# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _build_linear_configs():
    return [
        triton.Config(
            {"BLOCK_N": 64, "BLOCK_M": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_N": 64, "BLOCK_M": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_M": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_M": 128, "BLOCK_K": 16}, num_warps=16, num_stages=3
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_M": 128, "BLOCK_K": 32}, num_warps=16, num_stages=2
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_M": 128, "BLOCK_K": 64}, num_warps=16, num_stages=2
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_M": 256, "BLOCK_K": 32}, num_warps=16, num_stages=3
        ),
        triton.Config(
            {"BLOCK_N": 256, "BLOCK_M": 128, "BLOCK_K": 32}, num_warps=16, num_stages=3
        ),
        triton.Config(
            {"BLOCK_N": 256, "BLOCK_M": 256, "BLOCK_K": 32}, num_warps=32, num_stages=3
        ),
        triton.Config(
            {"BLOCK_N": 256, "BLOCK_M": 256, "BLOCK_K": 64}, num_warps=32, num_stages=2
        ),
    ]


def _build_fused_gemm_configs():
    return [
        triton.Config(
            {"BLOCK_N": 64, "BLOCK_M": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_N": 64, "BLOCK_M": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_N": 64, "BLOCK_M": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_N": 64, "BLOCK_M": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_M": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_M": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_M": 128, "BLOCK_K": 16, "GROUP_SIZE_M": 1},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_M": 128, "BLOCK_K": 16, "GROUP_SIZE_M": 4},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_M": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_M": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_M": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_M": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_M": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_M": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_M": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_M": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_N": 256, "BLOCK_M": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_N": 256, "BLOCK_M": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_N": 256, "BLOCK_M": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_N": 256, "BLOCK_M": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_N": 256, "BLOCK_M": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_N": 256, "BLOCK_M": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_warps=32,
            num_stages=2,
        ),
    ]


linear_configs = _build_linear_configs()
linear_fused_configs = _build_fused_gemm_configs()
shape_specialized_configs = _build_fused_gemm_configs()


@triton.autotune(configs=linear_configs, key=["N", "M", "K"])
@triton.jit
def _linear_bias_kernel(
    x_ptr,
    w_ptr,  # packed as [K, M]
    b_ptr,
    y_ptr,
    N,
    M,
    K,
    stride_xn,
    stride_xk,
    stride_wk,
    stride_wm,
    stride_yn,
    stride_ym,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    grf_mode: tl.constexpr = "auto",
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    k_tiles = tl.cdiv(K, BLOCK_K)
    offs_k = tl.arange(0, BLOCK_K)
    for ki in range(k_tiles):
        k0 = ki * BLOCK_K + offs_k

        a_ptrs = x_ptr + (offs_n[:, None] * stride_xn + k0[None, :] * stride_xk)
        a_mask = (offs_n[:, None] < N) & (k0[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = w_ptr + (k0[:, None] * stride_wk + offs_m[None, :] * stride_wm)
        b_mask = (k0[:, None] < K) & (offs_m[None, :] < M)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc)

    bias = tl.load(b_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    y_ptrs = y_ptr + (offs_n[:, None] * stride_yn + offs_m[None, :] * stride_ym)
    y_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


@triton.jit
def _min_sub_scalar_kernel(
    x_ptr,
    c_ptr,
    y_ptr,
    B,
    O,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK_SIZE: tl.constexpr,
):
    pid_col = tl.program_id(0)
    pid_row = tl.program_id(1)

    col_start = pid_col * BLOCK_SIZE
    offs_n = col_start + tl.arange(0, BLOCK_SIZE)
    offs_n = tl.max_contiguous(offs_n, BLOCK_SIZE)

    row_in = pid_row < B
    col_in = offs_n < O
    mask = row_in & col_in

    row_off_x = pid_row.to(tl.int64) * stride_xm
    row_off_y = pid_row.to(tl.int64) * stride_ym
    x_row = x_ptr + row_off_x
    y_row = y_ptr + row_off_y

    x_ptrs = x_row + offs_n * stride_xn
    y_ptrs = y_row + offs_n * stride_yn

    c_val = tl.load(c_ptr).to(tl.float32)
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    y_f32 = tl.minimum(x - c_val, 0.0)

    tl.store(y_ptrs, y_f32.to(y_ptr.dtype.element_ty), mask=mask)


@triton.autotune(configs=linear_fused_configs, key=["N", "M", "K"])
@triton.jit
def _linear_bias_minsub_kernel(
    x_ptr,
    w_ptr,  # packed [K, M]
    b_ptr,
    c_ptr,
    y_ptr,
    N,
    M,
    K,
    stride_xn,
    stride_xk,
    stride_wk,
    stride_wm,
    stride_yn,
    stride_ym,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    grf_mode: tl.constexpr = "auto",
):
    pid = tl.program_id(0)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_m = tl.cdiv(M, BLOCK_M)

    if GROUP_SIZE_M > 1 and num_pid_n > 1:
        group_size = GROUP_SIZE_M * num_pid_m
        group_id = pid // group_size
        first_pid_n = group_id * GROUP_SIZE_M
        group_n = tl.minimum(num_pid_n - first_pid_n, GROUP_SIZE_M)
        pid_n = first_pid_n + (pid % group_n)
        pid_m = (pid % group_size) // group_n
    else:
        pid_n = pid // num_pid_m
        pid_m = pid % num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = tl.max_contiguous(offs_m, BLOCK_M)

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    x_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(N, K),
        strides=(stride_xn, stride_xk),
        offsets=(pid_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    w_bp = tl.make_block_ptr(
        base=w_ptr,
        shape=(K, M),
        strides=(stride_wk, stride_wm),
        offsets=(0, pid_m * BLOCK_M),
        block_shape=(BLOCK_K, BLOCK_M),
        order=(1, 0),
    )

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(x_bp, boundary_check=(0, 1))
        b = tl.load(w_bp, boundary_check=(0, 1))
        acc = tl.dot(a, b, acc)
        x_bp = tl.advance(x_bp, (0, BLOCK_K))
        w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    bias = tl.load(b_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
    c_val = tl.load(c_ptr).to(tl.float32)
    acc = tl.minimum(acc + bias[None, :] - c_val, 0.0)

    y_bp = tl.make_block_ptr(
        base=y_ptr,
        shape=(N, M),
        strides=(stride_yn, stride_ym),
        offsets=(pid_n * BLOCK_N, pid_m * BLOCK_M),
        block_shape=(BLOCK_N, BLOCK_M),
        order=(1, 0),
    )
    tl.store(y_bp, acc.to(y_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=shape_specialized_configs, key=["N", "M", "K"])
@triton.jit
def _linear_bias_minsub_kernel_aligned(
    x_ptr,
    w_ptr,  # packed [K, M]
    b_ptr,
    c_ptr,
    y_ptr,
    N,
    M,
    K,
    stride_xn,
    stride_xk,
    stride_wk,
    stride_wm,
    stride_yn,
    stride_ym,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    grf_mode: tl.constexpr = "auto",
):
    pid = tl.program_id(0)

    num_pid_n = N // BLOCK_N
    num_pid_m = M // BLOCK_M

    if GROUP_SIZE_M > 1 and num_pid_n > 1:
        group_size = GROUP_SIZE_M * num_pid_m
        group_id = pid // group_size
        first_pid_n = group_id * GROUP_SIZE_M
        group_n = tl.minimum(num_pid_n - first_pid_n, GROUP_SIZE_M)
        pid_n = first_pid_n + (pid % group_n)
        pid_m = (pid % group_size) // group_n
    else:
        pid_n = pid // num_pid_m
        pid_m = pid % num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = tl.max_contiguous(offs_m, BLOCK_M)

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    x_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(N, K),
        strides=(stride_xn, stride_xk),
        offsets=(pid_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    w_bp = tl.make_block_ptr(
        base=w_ptr,
        shape=(K, M),
        strides=(stride_wk, stride_wm),
        offsets=(0, pid_m * BLOCK_M),
        block_shape=(BLOCK_K, BLOCK_M),
        order=(1, 0),
    )

    for _ in range(0, K // BLOCK_K):
        a = tl.load(x_bp)
        b = tl.load(w_bp)
        acc = tl.dot(a, b, acc)
        x_bp = tl.advance(x_bp, (0, BLOCK_K))
        w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    bias = tl.load(b_ptr + offs_m).to(tl.float32)
    c_val = tl.load(c_ptr).to(tl.float32)
    acc = tl.minimum(acc + bias[None, :] - c_val, 0.0)

    y_bp = tl.make_block_ptr(
        base=y_ptr,
        shape=(N, M),
        strides=(stride_yn, stride_ym),
        offsets=(pid_n * BLOCK_N, pid_m * BLOCK_M),
        block_shape=(BLOCK_N, BLOCK_M),
        order=(1, 0),
    )
    tl.store(y_bp, acc.to(y_ptr.dtype.element_ty))


def linear_bias_triton(
    x: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    if not (
        isinstance(x, torch.Tensor)
        and isinstance(weight_t, torch.Tensor)
        and isinstance(bias, torch.Tensor)
    ):
        raise TypeError("x, weight_t, bias must be tensors")
    if (
        x.device.type != "xpu"
        or weight_t.device.type != "xpu"
        or bias.device.type != "xpu"
    ):
        raise RuntimeError("All tensors must be on 'xpu'")
    if x.ndim != 2 or weight_t.ndim != 2 or bias.ndim != 1:
        raise ValueError("x: [N,K], weight_t: [K,M], bias: [M]")

    N, K = x.shape
    Kt, M = weight_t.shape
    if K != Kt or bias.shape[0] != M:
        raise ValueError("Shape mismatch")
    if x.dtype != weight_t.dtype or bias.dtype != weight_t.dtype:
        raise TypeError("dtypes must match")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("dtype not supported")

    x_xpu = x.contiguous()
    wt_xpu = weight_t
    b_xpu = bias.contiguous()

    y = torch.empty((N, M), device="xpu", dtype=x_xpu.dtype)

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_N"]), triton.cdiv(M, meta["BLOCK_M"]))

    _linear_bias_kernel[grid](
        x_xpu,
        wt_xpu,
        b_xpu,
        y,
        N,
        M,
        K,
        x_xpu.stride(0),
        x_xpu.stride(1),
        wt_xpu.stride(0),
        wt_xpu.stride(1),
        y.stride(0),
        y.stride(1),
    )
    return y


def min_sub_scalar_triton(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    if not (isinstance(x, torch.Tensor) and isinstance(c, torch.Tensor)):
        raise TypeError("x and c must be tensors")
    if x.device.type != "xpu" or c.device.type != "xpu":
        raise RuntimeError("x and c must be on 'xpu'")
    if x.ndim != 2 or c.ndim != 0 or c.numel() != 1:
        raise ValueError("x:[B,O], c:scalar")
    if x.dtype != c.dtype:
        raise TypeError("dtype mismatch")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("dtype not supported for min_sub")

    x_xpu = x.contiguous()
    c_xpu = c

    B, O = x_xpu.shape
    y = torch.empty_like(x_xpu)
    BLOCK_SIZE = 256
    grid = (triton.cdiv(O, BLOCK_SIZE), B)
    _min_sub_scalar_kernel[grid](
        x_xpu,
        c_xpu,
        y,
        B,
        O,
        x_xpu.stride(0),
        x_xpu.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )
    return y


def linear_bias_minsub_triton(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    bias: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    if not (
        isinstance(x, torch.Tensor)
        and isinstance(weight_t, torch.Tensor)
        and isinstance(bias, torch.Tensor)
        and isinstance(c, torch.Tensor)
    ):
        raise TypeError("x, weight_t, bias, c must be tensors")
    if (
        x.device.type != "xpu"
        or weight_t.device.type != "xpu"
        or bias.device.type != "xpu"
        or c.device.type != "xpu"
    ):
        raise RuntimeError("All tensors must be on 'xpu'")
    if x.ndim != 2 or weight_t.ndim != 2 or bias.ndim != 1:
        raise ValueError("x: [N,K], weight_t: [K,M], bias: [M]")
    if c.ndim != 0 or c.numel() != 1:
        raise ValueError("c must be a scalar tensor")

    N, K = x.shape
    Kt, M = weight_t.shape
    if K != Kt or bias.shape[0] != M:
        raise ValueError("Shape mismatch")
    if (
        x.dtype != weight_t.dtype
        or bias.dtype != weight_t.dtype
        or c.dtype != weight_t.dtype
    ):
        raise TypeError("dtypes must match")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("dtype not supported")

    x_xpu = x.contiguous()
    wt_xpu = weight_t
    b_xpu = bias.contiguous()
    c_xpu = c

    y = torch.empty((N, M), device="xpu", dtype=x_xpu.dtype)

    if (N % 128 == 0) and (M % 128 == 0) and (K % 32 == 0):

        def grid_aligned(meta):
            num_pid_n = N // meta["BLOCK_N"]
            num_pid_m = M // meta["BLOCK_M"]
            return (num_pid_n * num_pid_m,)

        _linear_bias_minsub_kernel_aligned[grid_aligned](
            x_xpu,
            wt_xpu,
            b_xpu,
            c_xpu,
            y,
            N,
            M,
            K,
            x_xpu.stride(0),
            x_xpu.stride(1),
            wt_xpu.stride(0),
            wt_xpu.stride(1),
            y.stride(0),
            y.stride(1),
        )
    else:

        def grid(meta):
            num_pid_n = triton.cdiv(N, meta["BLOCK_N"])
            num_pid_m = triton.cdiv(M, meta["BLOCK_M"])
            return (num_pid_n * num_pid_m,)

        _linear_bias_minsub_kernel[grid](
            x_xpu,
            wt_xpu,
            b_xpu,
            c_xpu,
            y,
            N,
            M,
            K,
            x_xpu.stride(0),
            x_xpu.stride(1),
            wt_xpu.stride(0),
            wt_xpu.stride(1),
            y.stride(0),
            y.stride(1),
        )
    return y


def kernel_function(
    x: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """
    Performs: y = min(x @ weight_t + bias, c) - c
    where weight_t is packed/transposed as [K, M].
    All on Intel XPU via Triton.
    """
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("XPU is not available")

    if x.device.type != "xpu":
        x_xpu = x.to(device="xpu", dtype=torch.float16)
    elif x.dtype != torch.float16:
        x_xpu = x.to(dtype=torch.float16)
    else:
        x_xpu = x
    x_xpu = x_xpu.contiguous()

    if weight_t.device.type != "xpu":
        wt_xpu = weight_t.to(device="xpu", dtype=x_xpu.dtype).contiguous()
    elif weight_t.dtype != x_xpu.dtype:
        wt_xpu = weight_t.to(dtype=x_xpu.dtype).contiguous()
    else:
        wt_xpu = weight_t

    if bias.device.type != "xpu":
        b_xpu = bias.to(device="xpu", dtype=x_xpu.dtype)
    elif bias.dtype != x_xpu.dtype:
        b_xpu = bias.to(dtype=x_xpu.dtype)
    else:
        b_xpu = bias
    b_xpu = b_xpu.contiguous()

    if c.device.type != "xpu":
        c_xpu = c.to(device="xpu", dtype=x_xpu.dtype)
    elif c.dtype != x_xpu.dtype:
        c_xpu = c.to(dtype=x_xpu.dtype)
    else:
        c_xpu = c

    return linear_bias_minsub_triton(x_xpu, wt_xpu, b_xpu, c_xpu)


batch_size = 128
in_features = 16384
out_features = 16384
constant = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, constant]


class Model(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self._out_features = out_features
        self.constant = constant
        self._cached_c = None
        self._params_on_xpu = False
        self._packed_weight_t = None
        self._packed_weight_version = None
        self._packed_weight_shape = None
        self._packed_weight_dtype = None
        self._packed_weight_device = None

    def _ensure_xpu_params_and_packed_weight(self, x_dtype):
        if not self._params_on_xpu:
            self.linear.weight.data = self.linear.weight.data.to(
                device="xpu", dtype=x_dtype
            ).contiguous()
            self.linear.bias.data = self.linear.bias.data.to(
                device="xpu", dtype=x_dtype
            ).contiguous()
            self._params_on_xpu = True
            self._packed_weight_t = None
            self._packed_weight_version = None
            self._packed_weight_shape = None
            self._packed_weight_dtype = None
            self._packed_weight_device = None

        weight = self.linear.weight
        current_version = weight._version
        need_repack = (
            self._packed_weight_t is None
            or self._packed_weight_version != current_version
            or self._packed_weight_shape != tuple(weight.shape)
            or self._packed_weight_dtype != weight.dtype
            or self._packed_weight_device != weight.device
        )
        if need_repack:
            self._packed_weight_t = weight.t().contiguous()
            self._packed_weight_version = current_version
            self._packed_weight_shape = tuple(weight.shape)
            self._packed_weight_dtype = weight.dtype
            self._packed_weight_device = weight.device

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to(device="xpu", dtype=torch.float16)
        x = x.contiguous()

        self._ensure_xpu_params_and_packed_weight(x.dtype)

        b = self.linear.bias
        if not b.is_contiguous():
            b = b.contiguous()

        if (
            self._cached_c is None
            or self._cached_c.device != x.device
            or self._cached_c.dtype != x.dtype
        ):
            self._cached_c = torch.tensor(self.constant, device=x.device, dtype=x.dtype)

        c = self._cached_c
        wt = self._packed_weight_t
        return kernel_function(x, wt, b, c)
