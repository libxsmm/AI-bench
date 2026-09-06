import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_last_dim_kernel(
    inp_ptr,
    out_ptr,
    M,
    N,
    BLOCK_SIZE_N: tl.constexpr,
    INIT_VAL: tl.constexpr,
    REDUCTION_OP: tl.constexpr,
    POST_OP: tl.constexpr,
):
    inp_desc = tl.make_tensor_descriptor(
        inp_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[1, BLOCK_SIZE_N],
    )

    row_val = INIT_VAL(dtype=inp_ptr.type.element_ty)
    m = tl.program_id(0)
    for n in range(0, N, BLOCK_SIZE_N):
        x = inp_desc.load([m, n]).reshape([BLOCK_SIZE_N])
        row_val = REDUCTION_OP(row_val, x, BLOCK_SIZE_N=BLOCK_SIZE_N)

    row_val = POST_OP(row_val, N=N)

    tl.store(out_ptr + m, tl.cast(row_val, out_ptr.type.element_ty))


@triton.jit
def _reduce_first_dim_kernel(
    inp_ptr,
    out_ptr,
    M,
    N,
    BLOCK_SIZE_N: tl.constexpr,
    INIT_VAL: tl.constexpr,
    REDUCTION_OP: tl.constexpr,
    POST_OP: tl.constexpr,
):
    inp_desc = tl.make_tensor_descriptor(
        inp_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[1, BLOCK_SIZE_N],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[1, N],
        strides=[N, 1],
        block_shape=[1, BLOCK_SIZE_N],
    )

    col_vals = INIT_VAL(dtype=inp_ptr.type.element_ty, BLOCK_SIZE_N=BLOCK_SIZE_N)
    n = tl.program_id(0) * BLOCK_SIZE_N
    for m in range(0, M):
        x = inp_desc.load([m, n]).reshape([BLOCK_SIZE_N])
        col_vals = REDUCTION_OP(col_vals, x, BLOCK_SIZE_N=BLOCK_SIZE_N)

    col_vals = POST_OP(col_vals, M=M)

    out_desc.store(
        [0, n], col_vals.to(out_ptr.type.element_ty).reshape([1, BLOCK_SIZE_N])
    )


@triton.jit
def _no_post_op_last_dim(x, N):
    return x


@triton.jit
def _default_init_val_last_dim(dtype):
    return tl.zeros([], dtype=dtype)


@triton.jit
def _no_post_op_first_dim(x, M):
    return x


@triton.jit
def _default_init_val_first_dim(dtype, BLOCK_SIZE_N):
    return tl.zeros([BLOCK_SIZE_N], dtype=dtype)


def reduce_last_dim(
    inp, out_dtype, reduction_op, init_val=None, post_op=None, keep_dim=False
):
    assert inp.ndim == 2, "Input tensor must be 2D"
    M, N = inp.shape
    out_shape = (M, 1) if keep_dim else (M,)
    out = torch.empty(out_shape, dtype=out_dtype, device=inp.device)
    BLOCK_SIZE_N = 256  # AVX-512 optimized block size
    assert N % BLOCK_SIZE_N == 0, "N must be divisible by BLOCK_SIZE_N"
    _reduce_last_dim_kernel[(M,)](
        inp,
        out,
        M,
        N,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        INIT_VAL=init_val or _default_init_val_last_dim,
        REDUCTION_OP=reduction_op,
        POST_OP=post_op or _no_post_op_last_dim,
        assume_in_bounds=True,
    )
    return out


def reduce_first_dim(
    inp, out_dtype, reduction_op, init_val=None, post_op=None, keep_dim=False
):
    assert inp.ndim == 2, "Input tensor must be 2D"
    M, N = inp.shape
    out_shape = (1, N) if keep_dim else (N,)
    out = torch.empty(out_shape, dtype=out_dtype, device=inp.device)
    BLOCK_SIZE_N = 256  # AVX-512 optimized block size
    assert N % BLOCK_SIZE_N == 0, "N must be divisible by BLOCK_SIZE_N"
    _reduce_first_dim_kernel[(N // BLOCK_SIZE_N,)](
        inp,
        out,
        M,
        N,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        INIT_VAL=init_val or _default_init_val_first_dim,
        REDUCTION_OP=reduction_op,
        POST_OP=post_op or _no_post_op_first_dim,
        assume_in_bounds=True,
    )
    return out


@triton.jit
def _softmax_kernel(inp_ptr, out_ptr, M, N, BLOCK_SIZE_N: tl.constexpr):
    inp_desc = tl.make_tensor_descriptor(
        inp_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[1, BLOCK_SIZE_N],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[1, BLOCK_SIZE_N],
    )

    row_max = -float("inf")
    row_sum = tl.zeros([], dtype=inp_ptr.type.element_ty)
    m = tl.program_id(0)
    for n in range(0, N, BLOCK_SIZE_N):
        x = inp_desc.load([m, n]).reshape([BLOCK_SIZE_N])
        row_max = tl.maximum(row_max, tl.max(x))

    for n in range(0, N, BLOCK_SIZE_N):
        x = inp_desc.load([m, n]).reshape([BLOCK_SIZE_N])
        e = tl.exp(x - row_max)
        row_sum += tl.sum(e)

    inv_row_sum = 1.0 / row_sum
    for n in range(0, N, BLOCK_SIZE_N):
        x = inp_desc.load([m, n]).reshape([BLOCK_SIZE_N])
        e = tl.exp(x - row_max)
        y = e * inv_row_sum
        out_desc.store([m, n], y.to(out_ptr.type.element_ty).reshape([1, BLOCK_SIZE_N]))


def softmax(inp, out_dtype, try_inplace=False):
    assert inp.ndim == 2, "Input tensor must be 2D"
    M, N = inp.shape
    BLOCK_SIZE_N = 256  # AVX-512 optimized block size
    assert N % BLOCK_SIZE_N == 0, "N must be divisible by BLOCK_SIZE_N"
    if try_inplace and inp.dtype == out_dtype:
        out = inp
    else:
        out = torch.empty_like(inp, dtype=out_dtype)
    _softmax_kernel[(M,)](
        inp,
        out,
        M,
        N,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        assume_in_bounds=True,
    )
    return out


@triton.jit
def _affine_groupnorm_2d_kernel(
    inp_ptr,
    out_ptr,
    post_op_arg_ptr,
    N,
    C,
    num_groups,
    eps,
    BLOCK_SIZE_C: tl.constexpr,
    POST_OP: tl.constexpr,
    POST_OP_HAS_ARG: tl.constexpr,
):
    group_size = C // num_groups

    inp_desc = tl.make_tensor_descriptor(
        inp_ptr,
        shape=[N, num_groups, group_size],
        strides=[C, group_size, 1],
        block_shape=[1, 1, BLOCK_SIZE_C],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[N, num_groups, group_size],
        strides=[C, group_size, 1],
        block_shape=[1, 1, BLOCK_SIZE_C],
    )

    n = tl.program_id(0)
    g = tl.program_id(1)

    # Welford's online algorithm: single pass over the data, merging one
    # block of BLOCK_SIZE_C elements at a time using Chan's parallel formula.
    mean_val = tl.zeros([], dtype=inp_ptr.type.element_ty)
    m2_val = tl.zeros([], dtype=inp_ptr.type.element_ty)
    count = 0
    for c in range(0, group_size, BLOCK_SIZE_C):
        x = inp_desc.load([n, g, c]).reshape([BLOCK_SIZE_C])
        block_mean = tl.sum(x) / BLOCK_SIZE_C
        block_m2 = tl.sum((x - block_mean) * (x - block_mean))
        new_count = count + BLOCK_SIZE_C
        delta = block_mean - mean_val
        mean_val += delta * BLOCK_SIZE_C / new_count
        m2_val += block_m2 + delta * delta * count * BLOCK_SIZE_C / new_count
        count = new_count

    var_val = m2_val / group_size
    inv_std_val = tl.rsqrt(var_val + eps)

    for c in range(0, group_size, BLOCK_SIZE_C):
        x = inp_desc.load([n, g, c]).reshape([BLOCK_SIZE_C])
        y = (x - mean_val) * inv_std_val
        if POST_OP_HAS_ARG:
            y = POST_OP(
                y,
                n=n,
                g=g,
                c=c,
                post_op_arg_ptr=post_op_arg_ptr,
                N=N,
                C=C,
                group_size=group_size,
                BLOCK_SIZE_C=BLOCK_SIZE_C,
            )
        else:
            y = POST_OP(y)
        out_desc.store(
            [n, g, c], y.to(out_ptr.type.element_ty).reshape([1, 1, BLOCK_SIZE_C])
        )


@triton.jit
def _no_post_op_norm(y):
    return y


@triton.jit
def _no_post_op_with_args(y, n, g, c, post_op_arg_ptr, N, C, group_size, BLOCK_SIZE_C):
    return y


def groupnorm(
    inp,
    out_dtype,
    num_groups,
    eps=1e-5,
    post_op=None,
    post_op_arg=None,
    try_inplace=False,
):
    assert inp.ndim == 2, "Input tensor must be 2D"
    N, C = inp.shape
    assert C % num_groups == 0, "Number of channels must be divisible by num_groups"
    group_size = C // num_groups
    BLOCK_SIZE_C = min(256, group_size)
    assert (
        group_size % BLOCK_SIZE_C == 0
        and triton.next_power_of_2(BLOCK_SIZE_C) == BLOCK_SIZE_C
    ), (
        "Group size must be divisible by BLOCK_SIZE_C, and BLOCK_SIZE_C must be a power of 2"
    )
    if try_inplace and inp.dtype == out_dtype:
        out = inp
    else:
        out = torch.empty_like(inp, dtype=out_dtype)
    _affine_groupnorm_2d_kernel[(N, num_groups)](
        inp,
        out,
        post_op_arg,
        N,
        C,
        num_groups,
        eps,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        POST_OP=post_op
        or (_no_post_op_norm if post_op_arg is None else _no_post_op_with_args),
        POST_OP_HAS_ARG=post_op_arg is not None,
        assume_in_bounds=True,
    )
    return out
