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
    REDUCTION: tl.constexpr,
    POST_OP: tl.constexpr,
):
    inp_desc = tl.make_tensor_descriptor(
        inp_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[1, BLOCK_SIZE_N],
    )

    row_val = tl.zeros([], dtype=out_ptr.type.element_ty)
    m = tl.program_id(0)
    for n in range(0, N, BLOCK_SIZE_N):
        x = inp_desc.load([m, n]).reshape([BLOCK_SIZE_N])
        row_val = REDUCTION(row_val, x)

    row_val = POST_OP(row_val)

    tl.store(out_ptr + m, tl.cast(row_val, out_ptr.type.element_ty))


@triton.jit
def _scalar_zero_init_op(dtype):
    return tl.zeros([], dtype=dtype)


@triton.jit
def _no_post_op(x):
    return x


def reduce_last_dim(inp, out_dtype, reduction, post_op=None, keep_dim=False):
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
        REDUCTION=reduction,
        POST_OP=post_op or _no_post_op,
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


def softmax(inp, out_dtype):
    assert inp.ndim == 2, "Input tensor must be 2D"
    M, N = inp.shape
    BLOCK_SIZE_N = 256  # AVX-512 optimized block size
    assert N % BLOCK_SIZE_N == 0, "N must be divisible by BLOCK_SIZE_N"
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
