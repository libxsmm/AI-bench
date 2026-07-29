import functools

import torch
import triton
import triton.language as tl

from triton_cpu_utils.gilbert_d2xy import gilbert_d2xy


# Transforms the A matrix into a tensor of shape:
#
#  (BLOCKS_M, BLOCKS_K, BLOCK_SIZE_M, BLOCK_SIZE_K)
#
# and the B matrix into a tensor of shape:
#
#  (BLOCKS_N, BLOCKS_K, BLOCK_SIZE_K, BLOCK_SIZE_N)
#
# Data is block-packed into contiguous chunks of memory. Neighboring blocks in
# the K dimension will also be neighboring in memory. In addition, the B matrix
# is also packed in VNNI format.
@triton.jit
def _block_pack_kernel(
    a_in_ptr,
    a_out_ptr,
    a_sfc_map_ptr,
    b_in_ptr,
    b_out_ptr,
    b_sfc_map_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    B_IS_PREPACKED: tl.constexpr = False,
):
    VNNI: tl.constexpr = 32 // b_in_ptr.type.element_ty.primitive_bitwidth

    pid = tl.program_id(axis=0)

    BLOCKS_M = M // BLOCK_SIZE_M
    BLOCKS_N = N // BLOCK_SIZE_N
    BLOCKS_K = K // BLOCK_SIZE_K

    # Block-pack A
    if pid < BLOCKS_M * BLOCKS_K:
        block_m = tl.load(a_sfc_map_ptr + 2 * pid)
        block_k = tl.load(a_sfc_map_ptr + 2 * pid + 1)

        a_in_desc = tl.make_tensor_descriptor(
            base=a_in_ptr,
            shape=(M, K),
            strides=(K, 1),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        )
        a_out_desc = tl.make_tensor_descriptor(
            base=a_out_ptr,
            shape=(BLOCKS_M, BLOCKS_K, BLOCK_SIZE_M, BLOCK_SIZE_K),
            strides=(BLOCK_SIZE_M * K, BLOCK_SIZE_M * BLOCK_SIZE_K, BLOCK_SIZE_K, 1),
            block_shape=(1, 1, BLOCK_SIZE_M, BLOCK_SIZE_K),
        )

        block = a_in_desc.load(
            (block_m * BLOCK_SIZE_M, block_k * BLOCK_SIZE_K)
        ).reshape((1, 1, BLOCK_SIZE_M, BLOCK_SIZE_K))
        a_out_desc.store((block_m, block_k, 0, 0), block)

    # Block-and-VNNI-pack B
    if B_IS_PREPACKED:
        return
    if pid < BLOCKS_K * BLOCKS_N:
        block_k = tl.load(b_sfc_map_ptr + 2 * pid)
        block_n = tl.load(b_sfc_map_ptr + 2 * pid + 1)

        b_in_desc = tl.make_tensor_descriptor(
            base=b_in_ptr, shape=(K, N), strides=(N, 1), block_shape=(1, BLOCK_SIZE_N)
        )
        b_out_desc = tl.make_tensor_descriptor(
            base=b_out_ptr,
            shape=(BLOCKS_N, BLOCKS_K, BLOCK_SIZE_K // VNNI, BLOCK_SIZE_N * VNNI),
            strides=(
                BLOCK_SIZE_N * K,
                BLOCK_SIZE_K * BLOCK_SIZE_N,
                BLOCK_SIZE_N * VNNI,
                1,
            ),
            block_shape=(1, 1, 1, BLOCK_SIZE_N * VNNI),
        )
        for i in tl.range(0, BLOCK_SIZE_K // VNNI):
            row1 = b_in_desc.load(
                (block_k * BLOCK_SIZE_K + i * VNNI, block_n * BLOCK_SIZE_N)
            ).reshape((BLOCK_SIZE_N,))
            if VNNI > 1:
                row2 = b_in_desc.load(
                    (block_k * BLOCK_SIZE_K + i * VNNI + 1, block_n * BLOCK_SIZE_N)
                ).reshape((BLOCK_SIZE_N,))
                if VNNI > 2:
                    row3 = b_in_desc.load(
                        (block_k * BLOCK_SIZE_K + i * VNNI + 2, block_n * BLOCK_SIZE_N)
                    ).reshape((BLOCK_SIZE_N,))
                    row4 = b_in_desc.load(
                        (block_k * BLOCK_SIZE_K + i * VNNI + 3, block_n * BLOCK_SIZE_N)
                    ).reshape((BLOCK_SIZE_N,))
                    row1 = tl.ravel(tl.join(row1, row3))
                    row2 = tl.ravel(tl.join(row2, row4))
                row1 = tl.ravel(tl.join(row1, row2))
            b_out_desc.store(
                (block_n, block_k, i, 0), row1.reshape((1, 1, 1, BLOCK_SIZE_N * VNNI))
            )


# Matmul kernel using the space curve filling approach in
# https://arxiv.org/abs/2601.16294v1, based on the generalized hilbert curve
# implementation from https://github.com/jakubcerveny/gilbert
#
# Each program computes a single output tile with the 2D coordinates derived
# from the precomputed SFC mapping. If `BLOCKING_FACTOR_K == 1`, then program
# handles all `BLOCKS_K = K // BLOCK_SIZE_K` blocks along the common dimension,
# otherwise the program performs a partial accumulation of the K blocks in the
# half-open interval:
#    [ ik * (BLOCKS_K // BLOCKING_FACTOR_K), (ik + 1) * (BLOCKS_K // BLOCKING_FACTOR_K) )
@triton.jit
def _sfc_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    ctmp_ptr,
    bias_ptr,
    post_op_arg_ptr,
    sfc_map_ptr,
    M,
    N,
    K,
    ik,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCKING_FACTOR_K: tl.constexpr,
    IS_FIRST_K_BLOCK: tl.constexpr,
    IS_LAST_K_BLOCK: tl.constexpr,
    ACCUM_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    POST_OP: tl.constexpr,
    POST_OP_HAS_ARG: tl.constexpr,
):
    VNNI: tl.constexpr = 32 // b_ptr.type.element_ty.primitive_bitwidth

    BLOCKS_M = M // BLOCK_SIZE_M
    BLOCKS_N = N // BLOCK_SIZE_N
    BLOCKS_K = K // BLOCK_SIZE_K
    BLOCKS_K_PER_PROG = tl.cdiv(BLOCKS_K, BLOCKING_FACTOR_K)

    pid = tl.program_id(axis=0)
    block_m = tl.load(sfc_map_ptr + 2 * pid)
    block_n = tl.load(sfc_map_ptr + 2 * pid + 1)
    block_k = ik * BLOCKS_K_PER_PROG

    a_desc = tl.make_tensor_descriptor(
        base=a_ptr,
        shape=(BLOCKS_M, BLOCKS_K, BLOCK_SIZE_M, BLOCK_SIZE_K),
        strides=(BLOCK_SIZE_M * K, BLOCK_SIZE_M * BLOCK_SIZE_K, BLOCK_SIZE_K, 1),
        block_shape=(1, 1, BLOCK_SIZE_M, BLOCK_SIZE_K),
    )

    b_desc = tl.make_tensor_descriptor(
        base=b_ptr,
        shape=(BLOCKS_N, BLOCKS_K, BLOCK_SIZE_K // VNNI, BLOCK_SIZE_N * VNNI),
        strides=(BLOCK_SIZE_N * K, BLOCK_SIZE_K * BLOCK_SIZE_N, BLOCK_SIZE_N * VNNI, 1),
        block_shape=(1, 1, BLOCK_SIZE_K // VNNI, BLOCK_SIZE_N * VNNI),
    )

    if BLOCKING_FACTOR_K > 1:
        ctmp_desc = tl.make_tensor_descriptor(
            base=ctmp_ptr,
            shape=(BLOCKS_M * BLOCKS_N, BLOCK_SIZE_M, BLOCK_SIZE_N),
            strides=(BLOCK_SIZE_M * BLOCK_SIZE_N, BLOCK_SIZE_N, 1),
            block_shape=(1, BLOCK_SIZE_M, BLOCK_SIZE_N),
        )

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUM_DTYPE)

    for block_ki in range(block_k, min(block_k + BLOCKS_K_PER_PROG, BLOCKS_K)):
        a = a_desc.load([block_m, block_ki, 0, 0]).reshape((BLOCK_SIZE_M, BLOCK_SIZE_K))
        b = b_desc.load([block_n, block_ki, 0, 0]).reshape(
            (BLOCK_SIZE_K // VNNI, BLOCK_SIZE_N * VNNI)
        )

        if VNNI > 1:
            b = tl.extra.cpu.vnni_decode(b)

        c = tl.dot(a, b, acc=c, out_dtype=ACCUM_DTYPE)

    if not IS_FIRST_K_BLOCK:
        c += ctmp_desc.load([pid, 0, 0]).reshape((BLOCK_SIZE_M, BLOCK_SIZE_N))

    if not IS_LAST_K_BLOCK:
        ctmp_desc.store([pid, 0, 0], c.reshape((1, BLOCK_SIZE_M, BLOCK_SIZE_N)))
        return

    if HAS_BIAS:
        bias_desc = tl.make_tensor_descriptor(
            base=bias_ptr,
            shape=(N,),
            strides=(1,),
            block_shape=(BLOCK_SIZE_N,),
        )
        bias = bias_desc.load([block_n * BLOCK_SIZE_N]).to(ACCUM_DTYPE)
        c += bias[None, :]

    if POST_OP_HAS_ARG:
        c = POST_OP(
            c, block_m, block_n, post_op_arg_ptr, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N
        )
    else:
        c = POST_OP(c)

    c = c.to(OUT_DTYPE).reshape((1, 1, BLOCK_SIZE_M, BLOCK_SIZE_N))

    c_desc = tl.make_tensor_descriptor(
        base=c_ptr,
        shape=(BLOCKS_M, BLOCKS_N, BLOCK_SIZE_M, BLOCK_SIZE_N),
        strides=(BLOCK_SIZE_M * N, BLOCK_SIZE_N, N, 1),
        block_shape=(1, 1, BLOCK_SIZE_M, BLOCK_SIZE_N),
    )
    c_desc.store([block_m, block_n, 0, 0], c)


@triton.jit
def _no_post_op(x):
    return x


@functools.lru_cache
def _make_sfc_tensor(x, y, dtype=torch.int32, device="cpu"):
    gilbert = (gilbert_d2xy(i, x, y) for i in range(x * y))
    return torch.tensor([c for xy in gilbert for c in xy], dtype=dtype, device=device)


def pack_weights_for_sfc_matmul(
    weights: torch.Tensor, BLOCK_SIZE_N, BLOCK_SIZE_K
) -> torch.Tensor:
    N, K = weights.shape
    assert weights.element_size() <= 4, (
        "Only 32-bit or smaller data types are supported"
    )
    VF = 32 // (weights.element_size() * 8)  # VNNI factor based on data type
    return (
        weights.reshape(
            N // BLOCK_SIZE_N,
            BLOCK_SIZE_N,
            K // BLOCK_SIZE_K,
            BLOCK_SIZE_K // VF,
            VF,
        )
        .permute(
            0,  # N // BLOCK_SIZE_N
            2,  # K // BLOCK_SIZE_K
            3,  # BLOCK_SIZE_K // VNNI
            1,  # BLOCK_SIZE_N
            4,  # VNNI
        )
        .contiguous()
        .reshape(K, N)
    )


def _get_accum_dtype(torch_dtype):
    if torch_dtype in [torch.float16, torch.bfloat16, torch.float32]:
        return torch.float32
    elif torch_dtype == torch.int8:
        return torch.int32
    else:
        raise ValueError(f"Unsupported dtype: {torch_dtype}")


def _torch_to_triton_dtype(torch_dtype):
    if torch_dtype == torch.float16:
        return tl.float16
    elif torch_dtype == torch.bfloat16:
        return tl.bfloat16
    elif torch_dtype == torch.float32:
        return tl.float32
    elif torch_dtype == torch.int8:
        return tl.int8
    elif torch_dtype == torch.int32:
        return tl.int32
    else:
        raise ValueError(f"Unsupported dtype: {torch_dtype}")


def sfc_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    bias=None,
    post_op=None,
    post_op_arg=None,
    trunc_output=True,
    b_is_prepacked=False,
    blocking_factor_k=1,
) -> torch.Tensor:
    assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
    assert a.device.type == "cpu" and b.device.type == "cpu", "A and B must be on CPU"
    assert a.dtype == b.dtype, f"dtype mismatch: {a.dtype} vs {b.dtype}"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Incompatible K dimensions: {K} vs {K2}"

    # AMX
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    # TODO: Currently masked load is not supported yet.
    assert (
        (M % BLOCK_SIZE_M == 0) and (N % BLOCK_SIZE_N == 0) and (K % BLOCK_SIZE_K == 0)
    ), (
        "Masking currently not supported, matrix dimensions must be multiples of block size"
    )

    BLOCKS_M = M // BLOCK_SIZE_M
    BLOCKS_N = N // BLOCK_SIZE_N
    BLOCKS_K = K // BLOCK_SIZE_K

    sfc_map_mk = _make_sfc_tensor(BLOCKS_M, BLOCKS_K)
    sfc_map_kn = _make_sfc_tensor(BLOCKS_K, BLOCKS_N)

    ap = torch.empty(
        (BLOCKS_M, BLOCKS_K, BLOCK_SIZE_M, BLOCK_SIZE_K), device=a.device, dtype=a.dtype
    )

    if b_is_prepacked:
        bp = b
    else:
        bp = torch.empty(
            (BLOCKS_N, BLOCKS_K, BLOCK_SIZE_K, BLOCK_SIZE_N),
            device=b.device,
            dtype=b.dtype,
        )

    num_blocks = max(
        BLOCKS_M * BLOCKS_K, BLOCKS_K * BLOCKS_N if not b_is_prepacked else 0
    )
    _block_pack_kernel[(num_blocks,)](
        a,
        ap,
        sfc_map_mk,
        b,
        bp,
        sfc_map_kn,
        M,
        N,
        K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        B_IS_PREPACKED=b_is_prepacked,
        assume_in_bounds=True,
    )

    sfc_map_mn = _make_sfc_tensor(BLOCKS_M, BLOCKS_N)

    accum_dtype = _get_accum_dtype(a.dtype)
    out_dtype = a.dtype if trunc_output else accum_dtype

    c = torch.empty((M, N), device=a.device, dtype=out_dtype)
    if blocking_factor_k > 1:
        if not trunc_output:
            ctmp = c
        else:
            ctmp = torch.empty(
                (M, N),
                device=a.device,
                dtype=accum_dtype,
            )
    else:
        ctmp = None

    for ik in range(blocking_factor_k):
        _sfc_matmul_kernel[(BLOCKS_M * BLOCKS_N,)](
            ap,
            bp,
            c,
            ctmp,
            bias,
            post_op_arg,
            sfc_map_mn,
            M,
            N,
            K,
            ik,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,  #
            BLOCKING_FACTOR_K=blocking_factor_k,
            IS_FIRST_K_BLOCK=ik == 0,
            IS_LAST_K_BLOCK=ik == blocking_factor_k - 1,
            ACCUM_DTYPE=_torch_to_triton_dtype(accum_dtype),
            OUT_DTYPE=_torch_to_triton_dtype(out_dtype),
            HAS_BIAS=bias is not None,
            POST_OP=post_op if post_op is not None else _no_post_op,
            POST_OP_HAS_ARG=post_op_arg is not None,
            assume_in_bounds=True,
        )

    return c
