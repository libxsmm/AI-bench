import functools

import cuda.tile as ct
from cuda.tile._backend import cpu
import torch

ct.set_backend("cpu")

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


def _no_post_op(value, **kwargs):
    return value


def _default_init_val_last_dim(dtype, **kwargs):
    return ct.full((), 0, dtype=dtype)


def _default_init_val_first_dim(dtype, BLOCK_SIZE_N, **kwargs):
    return ct.full((BLOCK_SIZE_N,), 0, dtype=dtype)


@functools.lru_cache()
def _make_reduce_last_dim_kernel(init_val, reduction_op, post_op):
    @ct.kernel
    def kernel(Input, Output, M, N, BLOCK_SIZE_N: ConstInt):
        row = ct.bid(0)
        row_value = init_val(Input.dtype)
        for block in range(N // BLOCK_SIZE_N):
            values = ct.load(
                Input,
                index=(row, block),
                shape=(1, BLOCK_SIZE_N),
            )
            values = ct.reshape(values, (BLOCK_SIZE_N,))
            row_value = reduction_op(
                row_value,
                values,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
            )
        row_value = post_op(row_value, N=N)
        ct.store(Output, index=(row,), tile=row_value.astype(Output.dtype))

    return kernel


@functools.lru_cache()
def _make_reduce_first_dim_kernel(init_val, reduction_op, post_op):
    @ct.kernel
    def kernel(Input, Output, M, N, BLOCK_SIZE_N: ConstInt):
        block = ct.bid(0)
        column_values = init_val(Input.dtype, BLOCK_SIZE_N=BLOCK_SIZE_N)
        for row in range(M):
            values = ct.load(
                Input,
                index=(row, block),
                shape=(1, BLOCK_SIZE_N),
            )
            values = ct.reshape(values, (BLOCK_SIZE_N,))
            column_values = reduction_op(
                column_values,
                values,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
            )
        column_values = post_op(column_values, M=M)
        ct.store(
            Output,
            index=(block,),
            tile=column_values.astype(Output.dtype),
        )

    return kernel


def reduce_last_dim(
    inp,
    out_dtype,
    reduction_op,
    init_val=None,
    post_op=None,
    keep_dim=False,
):
    assert inp.ndim == 2, "Input tensor must be 2D"
    M, N = inp.shape
    block_size_n = 256
    assert N % block_size_n == 0, "N must be divisible by BLOCK_SIZE_N"
    out = torch.empty((M,), dtype=out_dtype, device=inp.device)
    kernel = _make_reduce_last_dim_kernel(
        init_val or _default_init_val_last_dim,
        reduction_op,
        post_op or _no_post_op,
    )
    with cpu.compile_options({"assume_in_bounds": True}):
        ct.launch(
            None,
            (M,),
            kernel,
            (inp, out, M, N, block_size_n),
        )
    return out.reshape(M, 1) if keep_dim else out


def reduce_first_dim(
    inp,
    out_dtype,
    reduction_op,
    init_val=None,
    post_op=None,
    keep_dim=False,
):
    assert inp.ndim == 2, "Input tensor must be 2D"
    M, N = inp.shape
    block_size_n = 256
    assert N % block_size_n == 0, "N must be divisible by BLOCK_SIZE_N"
    out = torch.empty((N,), dtype=out_dtype, device=inp.device)
    kernel = _make_reduce_first_dim_kernel(
        init_val or _default_init_val_first_dim,
        reduction_op,
        post_op or _no_post_op,
    )
    with cpu.compile_options({"assume_in_bounds": True}):
        ct.launch(
            None,
            (N // block_size_n,),
            kernel,
            (inp, out, M, N, block_size_n),
        )
    return out.reshape(1, N) if keep_dim else out


@functools.lru_cache()
def _make_affine_groupnorm_2d_kernel(post_op):
    @ct.kernel
    def _affine_groupnorm_2d_kernel(
        Input,
        Output,
        PostOpArg,
        NUM_GROUPS: ConstInt,
        EPS,
        BLOCK_SIZE_C: ConstInt,
        POST_OP_HAS_ARG: ConstBool,
    ):
        group_id = ct.bid(0)
        batch = group_id // NUM_GROUPS
        group = group_id % NUM_GROUPS
        group_size = Input.shape[2]

        mean_value = ct.full((), 0, dtype=ct.float32)
        m2_value = ct.full((), 0, dtype=ct.float32)
        count = 0
        for block in range(group_size // BLOCK_SIZE_C):
            values = ct.load(
                Input,
                index=(batch, group, block),
                shape=(1, 1, BLOCK_SIZE_C),
            )
            values = ct.reshape(values, (BLOCK_SIZE_C,)).astype(ct.float32)
            block_mean = ct.sum(values, axis=0) / BLOCK_SIZE_C
            block_m2 = ct.sum(
                (values - block_mean) * (values - block_mean), axis=0
            )
            new_count = count + BLOCK_SIZE_C
            delta = block_mean - mean_value
            mean_value += delta * BLOCK_SIZE_C / new_count
            m2_value += (
                block_m2 + delta * delta * count * BLOCK_SIZE_C / new_count
            )
            count = new_count

        inverse_std = 1.0 / ct.sqrt(m2_value / group_size + EPS)
        for block in range(group_size // BLOCK_SIZE_C):
            values = ct.load(
                Input,
                index=(batch, group, block),
                shape=(1, 1, BLOCK_SIZE_C),
            )
            values = ct.reshape(values, (BLOCK_SIZE_C,)).astype(ct.float32)
            values = (values - mean_value) * inverse_std
            if POST_OP_HAS_ARG:
                values = post_op(
                    values,
                    n=batch,
                    g=group,
                    c=block * BLOCK_SIZE_C,
                    post_op_arg_ptr=PostOpArg,
                    N=Input.shape[0],
                    C=Input.shape[1] * Input.shape[2],
                    group_size=group_size,
                    BLOCK_SIZE_C=BLOCK_SIZE_C,
                )
            else:
                values = post_op(values)
            ct.store(
                Output,
                index=(batch, group, block),
                tile=ct.reshape(values.astype(Output.dtype), (1, 1, BLOCK_SIZE_C)),
            )

    return _affine_groupnorm_2d_kernel


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
    block_size_c = min(256, group_size)
    assert (
        group_size % block_size_c == 0
        and (1 << (block_size_c - 1).bit_length()) == block_size_c
    ), (
        "Group size must be divisible by BLOCK_SIZE_C, and BLOCK_SIZE_C must be a power of 2"
    )

    if try_inplace and inp.dtype == out_dtype:
        out = inp
    else:
        out = torch.empty_like(inp, dtype=out_dtype)

    input_view = inp.view(N, num_groups, group_size)
    output_view = out.view(N, num_groups, group_size)
    post_op_has_arg = post_op_arg is not None
    kernel = _make_affine_groupnorm_2d_kernel(post_op or _no_post_op)
    post_op_arg = inp if post_op_arg is None else post_op_arg
    with cpu.compile_options({"assume_in_bounds": True}):
        ct.launch(
            None,
            (N * num_groups,),
            kernel,
            (
                input_view,
                output_view,
                post_op_arg,
                num_groups,
                eps,
                block_size_c,
                post_op_has_arg,
            ),
        )
    return out
