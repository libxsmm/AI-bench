import torch
import triton
import triton.language as tl

from .gilbert_d2xy import gilbert_d2xy
from .thread_lru_cache import thread_lru_cache


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
    cred_ptr,
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
    REDUCE_LAST_DIM: tl.constexpr,
    REDUCTION_BLOCK_OP: tl.constexpr,
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
            c,
            block_m=block_m,
            block_n=block_n,
            post_op_arg_ptr=post_op_arg_ptr,
            M=M,
            N=N,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
    else:
        c = POST_OP(c)

    if REDUCE_LAST_DIM:
        cred = REDUCTION_BLOCK_OP(
            c, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N
        )
        cred_desc = tl.make_tensor_descriptor(
            base=cred_ptr,
            shape=(BLOCKS_N, M),
            strides=(M, 1),
            block_shape=(1, BLOCK_SIZE_M),
        )
        # Transposed write to intermediate buffer for final reduction
        cred = cred.reshape((1, BLOCK_SIZE_M))
        cred_desc.store([block_n, block_m * BLOCK_SIZE_M], cred)
        return

    # Normal writeback to output tensor
    c = c.to(OUT_DTYPE).reshape((1, 1, BLOCK_SIZE_M, BLOCK_SIZE_N))

    c_desc = tl.make_tensor_descriptor(
        base=c_ptr,
        shape=(BLOCKS_M, BLOCKS_N, BLOCK_SIZE_M, BLOCK_SIZE_N),
        strides=(BLOCK_SIZE_M * N, BLOCK_SIZE_N, N, 1),
        block_shape=(1, 1, BLOCK_SIZE_M, BLOCK_SIZE_N),
    )
    c_desc.store([block_m, block_n, 0, 0], c)


@triton.jit
def _finish_reduction_kernel(
    inp_ptr,
    out_ptr,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    REDUCTION_INIT_VAL: tl.constexpr,
    REDUCTION_COMBINE_OP: tl.constexpr,
    REDUCTION_POST_OP: tl.constexpr,
):
    BLOCKS_N = N // BLOCK_SIZE_N
    inp_desc = tl.make_tensor_descriptor(
        base=inp_ptr,
        shape=(BLOCKS_N, M),
        strides=(M, 1),
        block_shape=(1, BLOCK_SIZE_M),
    )
    m = tl.program_id(0) * BLOCK_SIZE_M
    column_vals = REDUCTION_INIT_VAL(inp_ptr.type.element_ty, BLOCK_SIZE_M=BLOCK_SIZE_M)
    for block_n in range(BLOCKS_N):
        block = inp_desc.load([block_n, m]).reshape((BLOCK_SIZE_M,))
        column_vals = REDUCTION_COMBINE_OP(column_vals, block)

    out_desc = tl.make_tensor_descriptor(
        base=out_ptr,
        shape=(M,),
        strides=(1,),
        block_shape=(BLOCK_SIZE_M,),
    )
    column_vals = REDUCTION_POST_OP(column_vals, M=M, N=N)
    out_desc.store([m], column_vals.to(out_ptr.type.element_ty))


@triton.jit
def _no_post_op(x, **kwargs):
    return x


@triton.jit
def _default_init_val(dtype, BLOCK_SIZE_M, **kwargs):
    return tl.zeros((BLOCK_SIZE_M,), dtype=dtype)


@triton.jit
def _default_reduction_op(x, **kwargs):
    return x.sum(axis=1)


@triton.jit
def _default_combine_op(a, b, **kwargs):
    return a + b


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


@thread_lru_cache()
def _make_sfc_tensor(x, y, dtype=torch.int32, device="cpu"):
    gilbert = (gilbert_d2xy(i, x, y) for i in range(x * y))
    return torch.tensor([c for xy in gilbert for c in xy], dtype=dtype, device=device)


@thread_lru_cache()
def _make_intermediate_buffers(
    M,
    N,
    K,
    BLOCK_SIZE_N,
    in_dtype,
    accum_dtype,
    out_dtype,
    reduce_last_dim,
    b_is_prepacked,
    c_is_owned,
):
    BLOCKS_N = N // BLOCK_SIZE_N
    OUT_N = 1 if reduce_last_dim else N

    ap_size = M * K * in_dtype.itemsize
    bp_size = K * N * in_dtype.itemsize if not b_is_prepacked else 0
    ctmp_size = M * N * accum_dtype.itemsize
    cred_size = M * BLOCKS_N * accum_dtype.itemsize if reduce_last_dim else 0
    c_size = M * OUT_N * out_dtype.itemsize if c_is_owned else 0

    buf = torch.empty(
        ap_size + bp_size + ctmp_size + cred_size + c_size, dtype=torch.uint8
    )
    ap = buf[:ap_size].view(in_dtype).reshape((M, K))
    bp = (
        buf[ap_size : ap_size + bp_size].view(in_dtype).reshape((K, N))
        if not b_is_prepacked
        else None
    )
    ctmp = (
        buf[ap_size + bp_size : ap_size + bp_size + ctmp_size]
        .view(accum_dtype)
        .reshape((M, N))
    )
    cred = (
        buf[ap_size + bp_size + ctmp_size : ap_size + bp_size + ctmp_size + cred_size]
        .view(accum_dtype)
        .reshape((BLOCKS_N, M))  # Transpose intentional
        if reduce_last_dim
        else None
    )
    c = (
        buf[ap_size + bp_size + ctmp_size + cred_size :]
        .view(out_dtype)
        .reshape((M, OUT_N))
        if c_is_owned
        else None
    )
    return ap, bp, ctmp, cred, c


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
    trunc_output=True,
    b_is_prepacked=False,
    c_is_owned=False,
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
    sfc_map_mn = _make_sfc_tensor(BLOCKS_M, BLOCKS_N)

    accum_dtype = _get_accum_dtype(a.dtype)
    out_dtype = a.dtype if trunc_output else accum_dtype

    ap, bp, ctmp, cred, c = _make_intermediate_buffers(
        M,
        N,
        K,
        BLOCK_SIZE_N,
        a.dtype,
        accum_dtype,
        out_dtype,
        reduce_last_dim,
        b_is_prepacked,
        c_is_owned,
    )
    if b_is_prepacked:
        bp = b
    if not c_is_owned:
        if not reduce_last_dim:
            c = torch.empty((M, N), device=a.device, dtype=out_dtype)
        elif keep_dim:
            c = torch.empty((M, 1), device=a.device, dtype=out_dtype)
        else:
            c = torch.empty((M,), device=a.device, dtype=out_dtype)

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

    for ik in range(blocking_factor_k):
        _sfc_matmul_kernel[(BLOCKS_M * BLOCKS_N,)](
            ap,
            bp,
            c,
            ctmp,
            cred,
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
            REDUCE_LAST_DIM=reduce_last_dim,
            REDUCTION_BLOCK_OP=reduction_block_op
            if reduction_block_op is not None
            else _default_reduction_op,
            assume_in_bounds=True,
        )

    if reduce_last_dim:
        _finish_reduction_kernel[(BLOCKS_M,)](
            cred,
            c,
            M,
            N,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            REDUCTION_INIT_VAL=reduction_init_val
            if reduction_init_val is not None
            else _default_init_val,
            REDUCTION_COMBINE_OP=reduction_combine_op
            if reduction_combine_op is not None
            else _default_combine_op,
            REDUCTION_POST_OP=reduction_post_op
            if reduction_post_op is not None
            else _no_post_op,
            assume_in_bounds=True,
        )

    return c
