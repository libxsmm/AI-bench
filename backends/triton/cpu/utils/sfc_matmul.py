import functools

from gilbert_d2xy import gilbert_d2xy
import torch
import triton
import triton.language as tl


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
    sfc_map_ptr,
    M,
    N,
    K,
    ik,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCKING_FACTOR_K: tl.constexpr,
):
    VNNI: tl.constexpr = 32 // b_ptr.type.element_ty.primitive_bitwidth

    BLOCKS_M = M // BLOCK_SIZE_M
    BLOCKS_N = N // BLOCK_SIZE_N
    BLOCKS_K = K // BLOCK_SIZE_K
    BLOCKS_K_PER_PROG = BLOCKS_K // BLOCKING_FACTOR_K

    dtype: tl.constexpr = a_ptr.type.element_ty
    accum_dtype: tl.constexpr = tl.float32 if dtype.is_floating() else tl.int32

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

    c_desc = tl.make_tensor_descriptor(
        base=c_ptr,
        shape=(BLOCKS_M, BLOCKS_N, BLOCK_SIZE_M, BLOCK_SIZE_N),
        strides=(BLOCK_SIZE_M * N, BLOCK_SIZE_N, N, 1),
        block_shape=(1, 1, BLOCK_SIZE_M, BLOCK_SIZE_N),
    )

    if ik == 0:
        c0 = tl.zeros((1, 1, BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=dtype)
        c_desc.store([block_m, block_n, 0, 0], c0)

    c = (
        c_desc.load([block_m, block_n, 0, 0])
        .reshape((BLOCK_SIZE_M, BLOCK_SIZE_N))
        .to(accum_dtype)
    )

    for block_ki in range(block_k, block_k + BLOCKS_K_PER_PROG):
        a = a_desc.load([block_m, block_ki, 0, 0]).reshape((BLOCK_SIZE_M, BLOCK_SIZE_K))
        b = b_desc.load([block_n, block_ki, 0, 0]).reshape(
            (BLOCK_SIZE_K // VNNI, BLOCK_SIZE_N * VNNI)
        )

        b = tl.extra.cpu.vnni_decode(b)

        c = tl.dot(a, b, acc=c, out_dtype=accum_dtype)

    c = c.to(dtype).reshape((1, 1, BLOCK_SIZE_M, BLOCK_SIZE_N))
    c_desc.store([block_m, block_n, 0, 0], c)


@functools.lru_cache
def _make_sfc_tensor(x, y, dtype=torch.int32, device="cpu"):
    gilbert = (gilbert_d2xy(i, x, y) for i in range(x * y))
    return torch.tensor([c for xy in gilbert for c in xy], dtype=dtype, device=device)


def sfc_matmul(a: torch.Tensor, b: torch.Tensor, blocking_factor_k=1):
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

    sfc_map_mn = _make_sfc_tensor(BLOCKS_M, BLOCKS_N)
    sfc_map_mk = _make_sfc_tensor(BLOCKS_M, BLOCKS_K)
    sfc_map_kn = _make_sfc_tensor(BLOCKS_K, BLOCKS_N)

    ap = torch.empty(
        (BLOCKS_M, BLOCKS_K, BLOCK_SIZE_M, BLOCK_SIZE_K), device=a.device, dtype=a.dtype
    )
    bp = torch.empty(
        (BLOCKS_N, BLOCKS_K, BLOCK_SIZE_K, BLOCK_SIZE_N), device=b.device, dtype=b.dtype
    )
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    num_blocks = max(BLOCKS_M * BLOCKS_K, BLOCKS_K * BLOCKS_N)
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
        assume_in_bounds=True,
    )

    for ik in range(blocking_factor_k):
        _sfc_matmul_kernel[(BLOCKS_M * BLOCKS_N,)](
            ap,
            bp,
            c,  #
            sfc_map_mn,  #
            M,
            N,
            K,  #
            ik,  #
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,  #
            BLOCKING_FACTOR_K=blocking_factor_k,
            assume_in_bounds=True,
        )

    return c
