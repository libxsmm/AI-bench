# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import functools
import threading

import cuda.tile as ct
import torch

from shared_cpu_utils import output_pool

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


def _compile_options(options):
    # All dimensions are asserted to be exact multiples of the block size, so
    # loads never need masking; Triton passes assume_in_bounds=True likewise.
    effective = {"assume_in_bounds": True}
    if options:
        effective.update(options)
    return effective


@ct.kernel
def _block_pack_b_kernel(
    B,
    BPacked,
    BSfcMap,
    VNNI: ConstInt,
):
    program_id = ct.bid(0)
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


def _make_buffers(
    a,
    b,
    reduce_last_dim,
    softmax_last_dim,
    trunc_output,
    b_is_prepacked,
):
    M, K = a.shape
    _, N = b.shape
    blocks_m = M // _BLOCK_SIZE_M
    blocks_n = N // _BLOCK_SIZE_N
    blocks_k = K // _BLOCK_SIZE_K
    vnni = 32 // (a.element_size() * 8)
    accum_dtype = _get_accum_dtype(a.dtype)
    out_dtype = a.dtype if trunc_output else accum_dtype
    bp = (
        output_pool.empty(
            (blocks_n, blocks_k, _BLOCK_SIZE_K // vnni, _BLOCK_SIZE_N, vnni),
            dtype=a.dtype,
            device=a.device,
        )
        if not b_is_prepacked
        else None
    )
    ctmp = output_pool.empty(
        (blocks_m, blocks_n, _BLOCK_SIZE_M, _BLOCK_SIZE_N),
        dtype=accum_dtype,
        device=a.device,
    )
    cred = (
        output_pool.empty(
            (blocks_n, 2 * M if softmax_last_dim else M),
            dtype=accum_dtype,
            device=a.device,
        )
        if reduce_last_dim or softmax_last_dim
        else ctmp
    )
    c = output_pool.empty(
        (M,) if reduce_last_dim else (M, N), dtype=out_dtype, device=a.device
    )
    return bp, ctmp, cred, c


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
    def _sfc_matmul_kernel(
        A,
        B,
        C,
        Ctmp,
        Cred,
        Bias,
        PostOpArg,
        TileMap,
        M,
        N,
        K,
        VNNI: ConstInt,
        BLOCKING_FACTOR_K: ConstInt,
        HAS_BIAS: ConstBool,
        POST_OP_HAS_ARG: ConstBool,
        REDUCE_LAST_DIM: ConstBool,
        SOFTMAX_LAST_DIM: ConstBool,
    ):
        program = ct.bid(0)
        programs = ct.num_blocks(0)
        blocks_m = M // _BLOCK_SIZE_M
        blocks_n = N // _BLOCK_SIZE_N
        blocks_k = K // _BLOCK_SIZE_K
        blocks_k_per_pass = ct.cdiv(blocks_k, BLOCKING_FACTOR_K)
        tile_count = blocks_m * blocks_n
        tiles_per_program = tile_count // programs
        remainder = tile_count % programs
        tile_begin = program * tiles_per_program + min(program, remainder)
        tile_end = (program + 1) * tiles_per_program + min(program + 1, remainder)
        phases = 3 if BLOCKING_FACTOR_K > 3 else BLOCKING_FACTOR_K
        accum_dtype = ct.int32 if A.dtype == ct.int8 else ct.float32

        for phase in ct.static_iter(range(phases)):
            if phase == 0:
                pass_begin = 0
                pass_end = 1
            elif phase == phases - 1:
                pass_begin = BLOCKING_FACTOR_K - 1
                pass_end = BLOCKING_FACTOR_K
            else:
                pass_begin = 1
                pass_end = BLOCKING_FACTOR_K - 1

            for k_pass in range(pass_begin, pass_end):
                first_block_k = k_pass * blocks_k_per_pass
                last_block_k = min(first_block_k + blocks_k_per_pass, blocks_k)

                for tile in range(tile_begin, tile_end):
                    coordinates = ct.load(TileMap, index=(tile, 0), shape=(1, 2))
                    block_m = ct.extract(coordinates, index=(0, 0), shape=()).item()
                    block_n = ct.extract(coordinates, index=(0, 1), shape=()).item()

                    if phase == 0:
                        accumulator = ct.full(
                            (_BLOCK_SIZE_M, _BLOCK_SIZE_N), 0, dtype=accum_dtype
                        )
                    else:
                        previous = ct.load(
                            Ctmp,
                            index=(block_m, block_n, 0, 0),
                            shape=(1, 1, _BLOCK_SIZE_M, _BLOCK_SIZE_N),
                        )
                        accumulator = ct.reshape(
                            previous, (_BLOCK_SIZE_M, _BLOCK_SIZE_N)
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

                    if phase < phases - 1:
                        ct.store(
                            Ctmp,
                            index=(block_m, block_n, 0, 0),
                            tile=ct.reshape(
                                accumulator,
                                (1, 1, _BLOCK_SIZE_M, _BLOCK_SIZE_N),
                            ),
                        )
                    else:
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
                        elif SOFTMAX_LAST_DIM:
                            ct.store(
                                Ctmp,
                                index=(block_m, block_n, 0, 0),
                                tile=ct.reshape(
                                    accumulator,
                                    (1, 1, _BLOCK_SIZE_M, _BLOCK_SIZE_N),
                                ),
                            )
                            block_max = ct.max(accumulator, axis=1, keepdims=True)
                            block_sum = ct.sum(
                                ct.exp(accumulator - block_max),
                                axis=1,
                                keepdims=True,
                            )
                            stats = ct.reshape(
                                ct.cat((block_max, block_sum), axis=1),
                                (1, 2 * _BLOCK_SIZE_M),
                            )
                            ct.store(Cred, index=(block_n, block_m), tile=stats)
                        else:
                            ct.store(
                                C,
                                index=(block_m, block_n),
                                tile=accumulator.astype(C.dtype),
                            )

    return _sfc_matmul_kernel


@functools.lru_cache(maxsize=1)
def _openmp_libraries():
    from cuda.tile._backends.cpu import _cpu

    return ctypes.CDLL(_cpu.__file__), ctypes.CDLL(None)


def _worker_count(options):
    workers = (options or {}).get("num_cpu_threads", 0)
    if workers > 0:
        return workers
    linked, global_runtime = (
        library.omp_get_max_threads() for library in _openmp_libraries()
    )
    return linked if linked == global_runtime else None


def _matmul_programs(tile_count, progs_per_thread, options):
    workers = _worker_count(options)
    if workers is None:
        return tile_count
    return min(tile_count, workers * progs_per_thread)


@functools.lru_cache()
def _make_finish_reduction_kernel(
    reduction_init_val, reduction_combine_op, reduction_post_op
):
    @ct.kernel
    def _finish_reduction_kernel(
        Input,
        Output,
        M,
        N,
        BLOCK_SIZE_M: ConstInt,
        BLOCK_SIZE_N: ConstInt,
    ):
        block_m = ct.bid(0)
        blocks_n = N // BLOCK_SIZE_N
        column_values = reduction_init_val(Input.dtype, BLOCK_SIZE_M=BLOCK_SIZE_M)
        for block_n in range(blocks_n):
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

    return _finish_reduction_kernel


@ct.kernel
def _finish_softmax_kernel(
    Ctmp,
    Stats,
    C,
    M,
    N,
):
    blocks_n = N // _BLOCK_SIZE_N
    block_m = ct.bid(0)
    block_max = ct.full((_BLOCK_SIZE_M,), float("-inf"), dtype=Ctmp.dtype)
    block_sum = ct.full((_BLOCK_SIZE_M,), 0, dtype=Ctmp.dtype)
    for block_n in range(blocks_n):
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

    for block_n in range(blocks_n):
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
    assert weights.element_size() <= 4, (
        "Only 32-bit or smaller data types are supported"
    )
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
        self._options = _compile_options(options)

    def __call__(self):
        with ct.backend_options(self._options):
            for grid, kernel, kernel_args in self._launches:
                ct.launch(None, grid, kernel, kernel_args)
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
    blocking_factor_k=1,
    progs_per_thread=1,
    options=None,
    _tile_map=None,
    _buffers=None,
) -> PreparedSFCMatmul:
    assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
    assert a.device.type == "cpu" and b.device.type == "cpu", "A and B must be on CPU"
    assert a.dtype == b.dtype, f"dtype mismatch: {a.dtype} vs {b.dtype}"
    assert a.is_contiguous(), "A must be contiguous"
    assert b.is_contiguous(), "B must be contiguous"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Incompatible K dimensions: {K} vs {K2}"
    assert not (reduce_last_dim and softmax_last_dim), (
        "Cannot reduce and softmax at the same time"
    )
    assert (
        M % _BLOCK_SIZE_M == 0 and N % _BLOCK_SIZE_N == 0 and K % _BLOCK_SIZE_K == 0
    ), (
        "Masking currently not supported, matrix dimensions must be multiples of block size"
    )

    blocks_m = M // _BLOCK_SIZE_M
    blocks_n = N // _BLOCK_SIZE_N
    blocks_k = K // _BLOCK_SIZE_K
    sfc_map_kn = _make_sfc_tensor(blocks_k, blocks_n)
    programs = _matmul_programs(blocks_m * blocks_n, progs_per_thread, options)
    # Custom orders may decline a shape; SFC order covers every shape.
    tile_map = (
        _tile_map(
            blocks_m,
            blocks_n,
            programs,
            -(-blocks_k // blocking_factor_k),
            a.element_size(),
        )
        if _tile_map is not None
        else None
    )
    if tile_map is None:
        tile_map = _make_sfc_tensor(blocks_m, blocks_n)
    bp, ctmp, cred, c = (
        _buffers
        if _buffers is not None
        else _make_buffers(
            a,
            b,
            reduce_last_dim,
            softmax_last_dim,
            trunc_output,
            b_is_prepacked,
        )
    )

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
    with ct.backend_options(_compile_options(options)):
        if not b_is_prepacked:
            pack_args = (b, kernel_b, sfc_map_kn, vnni)
            launches.append(
                (
                    (blocks_k * blocks_n, 1, 1),
                    _block_pack_b_kernel,
                    pack_args,
                )
            )

        a_view = a.view(blocks_m, _BLOCK_SIZE_M, blocks_k, _BLOCK_SIZE_K).permute(
            0, 2, 1, 3
        )
        kernel_args = (
            a_view,
            kernel_b,
            c,
            ctmp,
            cred,
            bias if bias is not None else placeholder,
            post_op_arg if post_op_arg is not None else placeholder,
            tile_map,
            M,
            N,
            K,
            vnni,
            blocking_factor_k,
            bias is not None,
            post_op_arg is not None,
            reduce_last_dim,
            softmax_last_dim,
        )
        launches.append(((programs, 1, 1), matmul_kernel, kernel_args))

        if reduce_last_dim:
            finish_block_size_m = 256 if M % 256 == 0 else _BLOCK_SIZE_M
            finish_kernel = _make_finish_reduction_kernel(
                reduction_init_val or _default_init_val,
                reduction_combine_op or _default_combine_op,
                reduction_post_op or _no_post_op,
            )
            finish_args = (
                cred,
                c.reshape(M),
                M,
                N,
                finish_block_size_m,
                _BLOCK_SIZE_N,
            )
            launches.append(
                (
                    (M // finish_block_size_m, 1, 1),
                    finish_kernel,
                    finish_args,
                )
            )
        elif softmax_last_dim:
            finish_args = (ctmp, cred, c, M, N)
            launches.append(
                (
                    (blocks_m, 1, 1),
                    _finish_softmax_kernel,
                    finish_args,
                )
            )

    if reduce_last_dim and keep_dim:
        output = c.reshape(M, 1)
    else:
        output = c
    return PreparedSFCMatmul(launches, output, options)


def _autotune_configs(blocking_factors, programs_per_thread):
    return [
        ct.tune.Config(
            {
                "blocking_factor_k": blocking_factor,
                "progs_per_thread": programs,
            }
        )
        for blocking_factor in blocking_factors
        for programs in programs_per_thread
        if blocking_factor != 1 or programs == 1 or 1 not in programs_per_thread
    ]


@_thread_lru_cache()
def _make_autotuned_sfc_matmul(
    blocking_factors, programs_per_thread, options_key, tile_map, _specialization_key
):
    options = dict(options_key)

    @ct.autotune(
        configs=_autotune_configs(blocking_factors, programs_per_thread),
        key=(
            "reduce_last_dim",
            "keep_dim",
            "softmax_last_dim",
            "trunc_output",
        ),
    )
    def autotuned(
        a,
        b,
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
        _buffers=None,
        blocking_factor_k=1,
        progs_per_thread=1,
    ):
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
            blocking_factor_k=blocking_factor_k,
            progs_per_thread=progs_per_thread,
            options=options,
            _tile_map=tile_map,
            _buffers=_buffers,
        )

    return autotuned


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
    blocking_factor_k=None,
    progs_per_thread=None,
    options=None,
    _tile_map=None,
) -> torch.Tensor:
    if blocking_factor_k is None or progs_per_thread is None:
        K = a.shape[1]
        blocking_factors = (
            (blocking_factor_k,)
            if blocking_factor_k is not None
            else tuple(
                factor
                for factor in (1, 2, 4, 8, 16)
                if factor == 1 or K // factor >= 1024
            )
        )
        programs = (
            (progs_per_thread,) if progs_per_thread is not None else (1, 2, 4, 8, 16)
        )
        buffers = _make_buffers(
            a, b, reduce_last_dim, softmax_last_dim, trunc_output, b_is_prepacked
        )
        options_key = tuple(sorted(_compile_options(options).items()))
        autotuned = _make_autotuned_sfc_matmul(
            blocking_factors,
            programs,
            options_key,
            _tile_map,
            (
                b_is_prepacked,
                post_op,
                reduction_init_val,
                reduction_block_op,
                reduction_combine_op,
                reduction_post_op,
            ),
        )
        return autotuned(
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
            _buffers=buffers,
        )

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
        blocking_factor_k=blocking_factor_k,
        progs_per_thread=progs_per_thread,
        options=options,
        _tile_map=_tile_map,
    )()
