import functools

from gilbert_d2xy import gilbert_d2xy
import torch
import triton
import triton.language as tl


# Transforms the B matrix into a tensor of shape:
#
#  (BLOCKS_N, BLOCKS_K, BLOCK_SIZE_K, BLOCK_SIZE_N)
#
# Data is blocked into contiguous chunks of memory. Neighboring blocks in the K
# dimension will also be neighboring in memory.
@triton.jit
def _block_transpose_kernel(
    in_ptr,
    out_ptr,
    sfc_map_ptr,
    N,
    K,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_k = tl.load(sfc_map_ptr + 2 * pid)
    block_n = tl.load(sfc_map_ptr + 2 * pid + 1)

    in_desc = tl.make_tensor_descriptor(
        base=in_ptr,
        shape=(K, N),
        strides=(N, 1),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
    )
    out_desc = tl.make_tensor_descriptor(
        base=out_ptr,
        shape=(N // BLOCK_SIZE_N, K // BLOCK_SIZE_K, BLOCK_SIZE_K, BLOCK_SIZE_N),
        strides=(BLOCK_SIZE_N * K, BLOCK_SIZE_K * BLOCK_SIZE_N, BLOCK_SIZE_N, 1),
        block_shape=(1, 1, BLOCK_SIZE_K, BLOCK_SIZE_N),
    )

    block = in_desc.load((block_k * BLOCK_SIZE_K, block_n * BLOCK_SIZE_N)).reshape(
        (1, 1, BLOCK_SIZE_K, BLOCK_SIZE_N)
    )
    out_desc.store((block_n, block_k, 0, 0), block)


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
        strides=(BLOCK_SIZE_M * K, BLOCK_SIZE_K, K, 1),
        block_shape=(1, 1, BLOCK_SIZE_M, BLOCK_SIZE_K),
    )

    b_desc = tl.make_tensor_descriptor(
        base=b_ptr,
        shape=(BLOCKS_N, BLOCKS_K, BLOCK_SIZE_K, BLOCK_SIZE_N),
        strides=(BLOCK_SIZE_N * K, BLOCK_SIZE_K * BLOCK_SIZE_N, BLOCK_SIZE_N, 1),
        block_shape=(1, 1, BLOCK_SIZE_K, BLOCK_SIZE_N),
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
        b = b_desc.load([block_n, block_ki, 0, 0]).reshape((BLOCK_SIZE_K, BLOCK_SIZE_N))

        c = tl.dot(a, b, c)

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
    sfc_map_kn = _make_sfc_tensor(BLOCKS_K, BLOCKS_N)

    bp = torch.empty(
        (BLOCKS_N, BLOCKS_K, BLOCK_SIZE_K, BLOCK_SIZE_N), device=b.device, dtype=b.dtype
    )
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    _block_transpose_kernel[(BLOCKS_K * BLOCKS_N,)](
        b,
        bp,
        sfc_map_kn,
        N,
        K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        assume_in_bounds=True,
    )

    for ik in range(blocking_factor_k):
        _sfc_matmul_kernel[(BLOCKS_M * BLOCKS_N,)](
            a,
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
