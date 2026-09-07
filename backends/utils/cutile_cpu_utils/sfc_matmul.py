# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from dataclasses import replace
import functools
import threading

import cuda.tile as ct
from cuda.tile._backend import cpu
from cuda.tile._backend._signature import build_signature
from cuda.tile.compilation import ArrayConstraint, KernelSignature
import torch

from .gilbert_d2xy import gilbert_d2xy

ct.set_backend("cpu")

ConstBool = ct.Constant[bool]
ConstInt = ct.Constant[int]

_BLOCK_SIZE_M = 32
_BLOCK_SIZE_N = 32
_BLOCK_SIZE_K = 32


def _thread_lru_cache(maxsize=128, typed=False):
    local = threading.local()

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                cached_func = local.cached_func
            except AttributeError:
                cached_func = functools.lru_cache(maxsize=maxsize, typed=typed)(func)
                local.cached_func = cached_func
            return cached_func(*args, **kwargs)

        return wrapper

    return decorator


@_thread_lru_cache()
def _make_sfc_tensor(width, height, dtype=torch.int32, device="cpu"):
    return torch.tensor(
        [gilbert_d2xy(index, width, height) for index in range(width * height)],
        dtype=dtype,
        device=device,
    )


def _get_accum_dtype(torch_dtype):
    if torch_dtype in (torch.float16, torch.bfloat16, torch.float32):
        return torch.float32
    if torch_dtype == torch.int8:
        return torch.int32
    raise ValueError(f"Unsupported dtype: {torch_dtype}")


def _specialize_packed_argument(parameters, argument_index, array, strides):
    if array.ndim != len(strides) or tuple(array.stride())[1:] != strides[1:]:
        return False

    constraint = parameters[argument_index]
    assert isinstance(constraint, ArrayConstraint)
    parameters[argument_index] = replace(constraint, stride_constant=strides)
    return True


def _build_matmul_signature(kernel, kernel_args):
    signature = build_signature(kernel, kernel_args)
    vnni = kernel_args[11]
    parameters = list(signature.parameters)

    changed = _specialize_packed_argument(
        parameters,
        0,
        kernel_args[0],
        (None, _BLOCK_SIZE_K * _BLOCK_SIZE_N, _BLOCK_SIZE_K, 1),
    )
    changed |= _specialize_packed_argument(
        parameters,
        1,
        kernel_args[1],
        (
            None,
            _BLOCK_SIZE_K * _BLOCK_SIZE_N,
            _BLOCK_SIZE_N * vnni,
            vnni,
            1,
        ),
    )
    if not changed:
        return signature

    signature = KernelSignature(parameters, signature.calling_convention)
    return signature.with_mangled_symbol(kernel._annotated_function.pyfunc.__name__)


@ct.kernel
def _block_pack_kernel(
    A,
    APacked,
    ASfcMap,
    B,
    BPacked,
    BSfcMap,
    VNNI: ConstInt,
    B_IS_PREPACKED: ConstBool,
):
    program_id = ct.bid(0)

    if program_id < APacked.shape[0] * APacked.shape[1]:
        coordinates = ct.load(ASfcMap, index=(program_id, 0), shape=(1, 2))
        block_m = ct.extract(coordinates, index=(0, 0), shape=()).item()
        block_k = ct.extract(coordinates, index=(0, 1), shape=()).item()
        a = ct.load(
            A,
            index=(block_m, block_k),
            shape=(_BLOCK_SIZE_M, _BLOCK_SIZE_K),
        )
        ct.store(
            APacked,
            index=(block_m, block_k, 0, 0),
            tile=ct.reshape(a, (1, 1, _BLOCK_SIZE_M, _BLOCK_SIZE_K)),
        )

    if B_IS_PREPACKED:
        return

    if program_id < BPacked.shape[0] * BPacked.shape[1]:
        coordinates = ct.load(BSfcMap, index=(program_id, 0), shape=(1, 2))
        block_k = ct.extract(coordinates, index=(0, 0), shape=()).item()
        block_n = ct.extract(coordinates, index=(0, 1), shape=()).item()
        b = ct.load(
            B,
            index=(block_k, block_n),
            shape=(_BLOCK_SIZE_K, _BLOCK_SIZE_N),
        )
        b = ct.reshape(
            ct.permute(
                ct.reshape(
                    b,
                    (_BLOCK_SIZE_K // VNNI, VNNI, _BLOCK_SIZE_N),
                ),
                (0, 2, 1),
            ),
            (1, 1, _BLOCK_SIZE_K // VNNI, _BLOCK_SIZE_N, VNNI),
        )
        ct.store(BPacked, index=(block_n, block_k, 0, 0, 0), tile=b)


@_thread_lru_cache()
def _make_intermediate_buffers(
    M,
    N,
    K,
    BLOCK_SIZE_N,
    in_dtype,
    accum_dtype,
    out_dtype,
    reduce_last_dim,
    softmax_last_dim,
    b_is_prepacked,
    c_is_owned,
):
    blocks_m = M // _BLOCK_SIZE_M
    blocks_n = N // BLOCK_SIZE_N
    blocks_k = K // _BLOCK_SIZE_K
    vnni = 32 // (in_dtype.itemsize * 8)
    ap = torch.empty(
        (blocks_m, blocks_k, _BLOCK_SIZE_M, _BLOCK_SIZE_K), dtype=in_dtype
    )
    bp = (
        None
        if b_is_prepacked
        else torch.empty(
            (
                blocks_n,
                blocks_k,
                _BLOCK_SIZE_K // vnni,
                BLOCK_SIZE_N,
                vnni,
            ),
            dtype=in_dtype,
        )
    )
    ctmp = torch.empty(
        (blocks_m, blocks_n, _BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accum_dtype
    )
    if reduce_last_dim:
        cred = torch.empty((blocks_n, M), dtype=accum_dtype)
    elif softmax_last_dim:
        cred = torch.empty((blocks_n, 2 * M), dtype=accum_dtype)
    else:
        cred = torch.empty((1,), dtype=accum_dtype)
    c = torch.empty((M, 1 if reduce_last_dim else N), dtype=out_dtype) if c_is_owned else None
    return ap, bp, ctmp, cred, c


def _no_post_op(value, **kwargs):
    return value


def _default_init_val(dtype, BLOCK_SIZE_M, **kwargs):
    return ct.full((BLOCK_SIZE_M,), 0, dtype=dtype)


def _default_reduction_op(value, **kwargs):
    return ct.sum(value, axis=1)


def _default_combine_op(lhs, rhs, **kwargs):
    return lhs + rhs


@functools.lru_cache()
def _make_matmul_kernel(post_op, reduction_block_op):
    @ct.kernel
    def kernel(
        A,
        B,
        C,
        Ctmp,
        Cred,
        Bias,
        PostOpArg,
        SfcMap,
        ik: ConstInt,
        M: ConstInt,
        N: ConstInt,
        VNNI: ConstInt,
        BLOCKING_FACTOR_K: ConstInt,
        IS_FIRST_K_BLOCK: ConstBool,
        IS_LAST_K_BLOCK: ConstBool,
        HAS_BIAS: ConstBool,
        POST_OP_HAS_ARG: ConstBool,
        REDUCE_LAST_DIM: ConstBool,
        SOFTMAX_LAST_DIM: ConstBool,
    ):
        coordinates = ct.load(SfcMap, index=(ct.bid(0), 0), shape=(1, 2))
        block_m = ct.extract(coordinates, index=(0, 0), shape=()).item()
        block_n = ct.extract(coordinates, index=(0, 1), shape=()).item()
        blocks_k = A.shape[1]
        blocks_k_per_program = ct.cdiv(blocks_k, BLOCKING_FACTOR_K)
        first_block_k = ik * blocks_k_per_program
        last_block_k = min(first_block_k + blocks_k_per_program, blocks_k)
        accum_dtype = ct.int32 if A.dtype == ct.int8 else ct.float32
        accumulator = ct.full(
            (_BLOCK_SIZE_M, _BLOCK_SIZE_N), 0, dtype=accum_dtype
        )

        for block_k in range(first_block_k, last_block_k):
            a = ct.load(
                A,
                index=(block_m, block_k, 0, 0),
                shape=(1, 1, _BLOCK_SIZE_M, _BLOCK_SIZE_K),
            )
            a = ct.reshape(a, (_BLOCK_SIZE_M, _BLOCK_SIZE_K))
            b = ct.load(
                B,
                index=(block_n, block_k, 0, 0, 0),
                shape=(
                    1,
                    1,
                    _BLOCK_SIZE_K // VNNI,
                    _BLOCK_SIZE_N,
                    VNNI,
                ),
            )
            b = ct.reshape(
                ct.permute(b, (0, 1, 2, 4, 3)),
                (_BLOCK_SIZE_K, _BLOCK_SIZE_N),
            )
            accumulator = ct.mma(a, b, accumulator)

        if not IS_FIRST_K_BLOCK:
            previous = ct.load(
                Ctmp,
                index=(block_m, block_n, 0, 0),
                shape=(1, 1, _BLOCK_SIZE_M, _BLOCK_SIZE_N),
            )
            accumulator += ct.reshape(previous, (_BLOCK_SIZE_M, _BLOCK_SIZE_N))

        if not IS_LAST_K_BLOCK:
            ct.store(
                Ctmp,
                index=(block_m, block_n, 0, 0),
                tile=ct.reshape(
                    accumulator, (1, 1, _BLOCK_SIZE_M, _BLOCK_SIZE_N)
                ),
            )
            return

        if HAS_BIAS:
            bias = ct.load(
                Bias, index=(block_n,), shape=(_BLOCK_SIZE_N,)
            ).astype(accum_dtype)
            accumulator += bias[None, :]

        if POST_OP_HAS_ARG:
            accumulator = post_op(
                accumulator,
                block_m=block_m,
                block_n=block_n,
                post_op_arg_ptr=PostOpArg,
                M=M,
                N=N,
                BLOCK_SIZE_M=_BLOCK_SIZE_M,
                BLOCK_SIZE_N=_BLOCK_SIZE_N,
            )
        else:
            accumulator = post_op(accumulator)

        if REDUCE_LAST_DIM:
            reduced = reduction_block_op(
                accumulator,
                BLOCK_SIZE_M=_BLOCK_SIZE_M,
                BLOCK_SIZE_N=_BLOCK_SIZE_N,
            )
            ct.store(
                Cred,
                index=(block_n, block_m),
                tile=ct.reshape(reduced, (1, _BLOCK_SIZE_M)),
            )
            return

        if SOFTMAX_LAST_DIM:
            ct.store(
                Ctmp,
                index=(block_m, block_n, 0, 0),
                tile=ct.reshape(
                    accumulator, (1, 1, _BLOCK_SIZE_M, _BLOCK_SIZE_N)
                ),
            )
            block_max = ct.max(accumulator, axis=1, keepdims=True)
            block_sum = ct.sum(
                ct.exp(accumulator - block_max), axis=1, keepdims=True
            )
            stats = ct.reshape(ct.cat((block_max, block_sum), axis=1), (1, 2 * _BLOCK_SIZE_M))
            ct.store(Cred, index=(block_n, block_m), tile=stats)
            return

        ct.store(
            C,
            index=(block_m, block_n),
            tile=accumulator.astype(C.dtype),
        )

    return kernel


@functools.lru_cache()
def _make_finish_reduction_kernel(
    reduction_init_val, reduction_combine_op, reduction_post_op
):
    @ct.kernel
    def kernel(
        Input,
        Output,
        M: ConstInt,
        N: ConstInt,
        BLOCK_SIZE_M: ConstInt,
    ):
        block_m = ct.bid(0)
        column_values = reduction_init_val(
            Input.dtype, BLOCK_SIZE_M=BLOCK_SIZE_M
        )
        for block_n in range(Input.shape[0]):
            block = ct.load(
                Input,
                index=(block_n, block_m),
                shape=(1, BLOCK_SIZE_M),
            )
            column_values = reduction_combine_op(
                column_values, ct.reshape(block, (BLOCK_SIZE_M,))
            )
        column_values = reduction_post_op(column_values, M=M, N=N)
        ct.store(
            Output,
            index=(block_m,),
            tile=ct.reshape(column_values, (BLOCK_SIZE_M,)).astype(Output.dtype),
        )

    return kernel


@ct.kernel
def _finish_softmax_kernel(
    Ctmp,
    Stats,
    C,
    BLOCKS_N: ConstInt,
):
    block_m = ct.bid(0)
    block_max = ct.full((_BLOCK_SIZE_M,), float("-inf"), dtype=Ctmp.dtype)
    block_sum = ct.full((_BLOCK_SIZE_M,), 0, dtype=Ctmp.dtype)
    for block_n in range(BLOCKS_N):
        stats = ct.load(
            Stats,
            index=(block_n, block_m),
            shape=(1, 2 * _BLOCK_SIZE_M),
        )
        stats = ct.reshape(stats, (_BLOCK_SIZE_M, 2))
        next_max = ct.reshape(
            ct.extract(stats, index=(0, 0), shape=(_BLOCK_SIZE_M, 1)),
            (_BLOCK_SIZE_M,),
        )
        next_sum = ct.reshape(
            ct.extract(stats, index=(0, 1), shape=(_BLOCK_SIZE_M, 1)),
            (_BLOCK_SIZE_M,),
        )
        combined_max = ct.maximum(block_max, next_max)
        block_sum = block_sum * ct.exp(block_max - combined_max)
        block_sum += next_sum * ct.exp(next_max - combined_max)
        block_max = combined_max

    for block_n in range(BLOCKS_N):
        values = ct.load(
            Ctmp,
            index=(block_m, block_n, 0, 0),
            shape=(1, 1, _BLOCK_SIZE_M, _BLOCK_SIZE_N),
        )
        values = ct.reshape(values, (_BLOCK_SIZE_M, _BLOCK_SIZE_N))
        result = ct.exp(values - block_max[:, None]) / block_sum[:, None]
        ct.store(C, index=(block_m, block_n), tile=result.astype(C.dtype))


def pack_weights_for_sfc_matmul(
    weights: torch.Tensor, BLOCK_SIZE_N, BLOCK_SIZE_K
) -> torch.Tensor:
    N, K = weights.shape
    assert weights.element_size() <= 4, "Only 32-bit or smaller data types are supported"
    vnni = 32 // (weights.element_size() * 8)
    return (
        weights.reshape(
            N // BLOCK_SIZE_N,
            BLOCK_SIZE_N,
            K // BLOCK_SIZE_K,
            BLOCK_SIZE_K // vnni,
            vnni,
        )
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .reshape(K, N)
    )


class PreparedSFCMatmul:
    def __init__(self, launches, output, options):
        self._launches = launches
        self._output = output
        self._options = None if options is None else dict(options)

    def __call__(self):
        context = (
            nullcontext()
            if self._options is None
            else cpu.compile_options(self._options)
        )
        with context:
            for grid, compiled, kernel_args in self._launches:
                ct.launch_compiled(None, grid, compiled, kernel_args)
        return self._output


def prepare_sfc_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    bias=None,
    post_op=None,
    post_op_arg=None,
    reduce_last_dim=False,
    reduction_init_val=None,
    reduction_block_op=None,
    reduction_combine_op=None,
    reduction_post_op=None,
    keep_dim=False,
    softmax_last_dim=False,
    trunc_output=True,
    b_is_prepacked=False,
    c_is_owned=False,
    blocking_factor_k=1,
    options=None,
) -> torch.Tensor:
    assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
    assert a.device.type == "cpu" and b.device.type == "cpu", "A and B must be on CPU"
    assert a.dtype == b.dtype, f"dtype mismatch: {a.dtype} vs {b.dtype}"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Incompatible K dimensions: {K} vs {K2}"
    assert not (reduce_last_dim and softmax_last_dim), (
        "Cannot reduce and softmax at the same time"
    )
    assert M % _BLOCK_SIZE_M == 0 and N % _BLOCK_SIZE_N == 0 and K % _BLOCK_SIZE_K == 0, (
        "Masking currently not supported, matrix dimensions must be multiples of block size"
    )

    blocks_m = M // _BLOCK_SIZE_M
    blocks_n = N // _BLOCK_SIZE_N
    blocks_k = K // _BLOCK_SIZE_K
    accum_dtype = _get_accum_dtype(a.dtype)
    out_dtype = a.dtype if trunc_output else accum_dtype
    sfc_map_mk = _make_sfc_tensor(blocks_m, blocks_k)
    sfc_map_kn = _make_sfc_tensor(blocks_k, blocks_n)
    sfc_map_mn = _make_sfc_tensor(blocks_m, blocks_n)
    ap, bp, ctmp, cred, owned_c = _make_intermediate_buffers(
        M,
        N,
        K,
        _BLOCK_SIZE_N,
        a.dtype,
        accum_dtype,
        out_dtype,
        reduce_last_dim,
        softmax_last_dim,
        b_is_prepacked,
        c_is_owned,
    )

    if owned_c is not None:
        c = owned_c
    elif reduce_last_dim:
        c = torch.empty((M,), device=a.device, dtype=out_dtype)
    else:
        c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    vnni = 32 // (b.element_size() * 8)
    kernel_b = (
        b.view(blocks_n, blocks_k, _BLOCK_SIZE_K // vnni, _BLOCK_SIZE_N, vnni)
        if b_is_prepacked
        else bp
    )
    placeholder = ctmp
    matmul_kernel = _make_matmul_kernel(
        post_op or _no_post_op,
        reduction_block_op or _default_reduction_op,
    )

    launches = []
    context = (
        nullcontext()
        if options is None
        else cpu.compile_options(dict(options))
    )
    with context:
        pack_args = (
            a,
            ap,
            sfc_map_mk,
            b,
            kernel_b,
            sfc_map_kn,
            vnni,
            b_is_prepacked,
        )
        pack_grid = max(
            blocks_m * blocks_k,
            0 if b_is_prepacked else blocks_k * blocks_n,
        )
        launches.append(
            (
                (pack_grid, 1, 1),
                ct.compile_kernel_for_launch(_block_pack_kernel, pack_args),
                pack_args,
            )
        )

        for ik in range(blocking_factor_k):
            kernel_args = (
                ap,
                kernel_b,
                c,
                ctmp,
                cred,
                bias if bias is not None else placeholder,
                post_op_arg if post_op_arg is not None else placeholder,
                sfc_map_mn,
                ik,
                M,
                N,
                vnni,
                blocking_factor_k,
                ik == 0,
                ik == blocking_factor_k - 1,
                bias is not None,
                post_op_arg is not None,
                reduce_last_dim,
                softmax_last_dim,
            )
            launches.append(
                (
                    (blocks_m * blocks_n, 1, 1),
                    cpu.compile_for_launch(
                        matmul_kernel,
                        kernel_args,
                        signature_builder=_build_matmul_signature,
                    ),
                    kernel_args,
                )
            )

        if reduce_last_dim:
            finish_block_size_m = 256 if M % 256 == 0 else _BLOCK_SIZE_M
            finish_kernel = _make_finish_reduction_kernel(
                reduction_init_val or _default_init_val,
                reduction_combine_op or _default_combine_op,
                reduction_post_op or _no_post_op,
            )
            finish_args = (cred, c.reshape(M), M, N, finish_block_size_m)
            launches.append(
                (
                    (M // finish_block_size_m, 1, 1),
                    ct.compile_kernel_for_launch(finish_kernel, finish_args),
                    finish_args,
                )
            )
        elif softmax_last_dim:
            finish_args = (ctmp, cred, c, blocks_n)
            launches.append(
                (
                    (blocks_m, 1, 1),
                    ct.compile_kernel_for_launch(
                        _finish_softmax_kernel, finish_args
                    ),
                    finish_args,
                )
            )

    if reduce_last_dim and (keep_dim or c_is_owned):
        output = c.reshape(M, 1)
    else:
        output = c
    return PreparedSFCMatmul(launches, output, options)


def sfc_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    bias=None,
    post_op=None,
    post_op_arg=None,
    reduce_last_dim=False,
    reduction_init_val=None,
    reduction_block_op=None,
    reduction_combine_op=None,
    reduction_post_op=None,
    keep_dim=False,
    softmax_last_dim=False,
    trunc_output=True,
    b_is_prepacked=False,
    c_is_owned=False,
    blocking_factor_k=1,
    options=None,
) -> torch.Tensor:
    return prepare_sfc_matmul(
        a,
        b,
        bias=bias,
        post_op=post_op,
        post_op_arg=post_op_arg,
        reduce_last_dim=reduce_last_dim,
        reduction_init_val=reduction_init_val,
        reduction_block_op=reduction_block_op,
        reduction_combine_op=reduction_combine_op,
        reduction_post_op=reduction_post_op,
        keep_dim=keep_dim,
        softmax_last_dim=softmax_last_dim,
        trunc_output=trunc_output,
        b_is_prepacked=b_is_prepacked,
        c_is_owned=c_is_owned,
        blocking_factor_k=blocking_factor_k,
        options=options,
    )()
