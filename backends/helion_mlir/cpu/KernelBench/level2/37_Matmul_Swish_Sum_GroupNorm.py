import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
from helion_mlir_backend import output_pool
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear

# K pairs per AMX step.
_PAIRS_STEP = 32
# K pairs per tile of the weight pack.
_PACK_PAIRS = 16


def _pack_groups(b3_t: torch.Tensor, group_size: hl.constexpr) -> torch.Tensor:
    """``[N, K/2, 2]`` K pairs of the transposed weight as one VNNI panel per
    group, ``[N/group, K/2, group, 2]``: a GEMM tile of one group reads one
    panel, a contraction AMX takes (several panels per tile do not)."""
    n, pairs, vnni = b3_t.shape
    group_size = int(group_size)
    out = output_pool.empty(
        (n // group_size, pairs, group_size, vnni), dtype=b3_t.dtype, device=b3_t.device
    )
    for tp, tn in hl.tile([pairs, n]):
        out[tn.id, tp, :, :] = b3_t[tn, tp, :].permute(1, 0, 2)
    return out


def _gemm_swish_group_norm(
    a3: torch.Tensor,
    b4: torch.Tensor,
    params: torch.Tensor,
    eps: hl.constexpr,
) -> torch.Tensor:
    """``GroupNorm(swish(a @ b + bias) + bias2)`` of ``a3``, bf16 ``[M, K/2, 2]``,
    and ``b4``, B in one VNNI panel per group ``[G, K/2, N/G, 2]``; ``params``
    stacks the f32 bias, bias2, GroupNorm weight and bias as ``[4, G, N/G]``. A
    tile's columns are one group, so its statistics are reduced in the
    epilogue."""
    m, pairs, _ = a3.shape
    groups, _, group_size, _ = b4.shape
    out = output_pool.empty((m, groups, group_size), dtype=a3.dtype, device=a3.device)
    for tile_m, tile_g in hl.tile([m, groups]):
        acc = hl.zeros([tile_m, tile_g, group_size], dtype=torch.float32)
        for tile_kp in hl.tile(pairs):
            acc = acc + torch.einsum(
                "mcv,bcnv->mbn", a3[tile_m, tile_kp, :], b4[tile_g, tile_kp, :, :]
            )
        y = acc + params[0, tile_g, :]
        y = torch.sigmoid(y) * y + params[1, tile_g, :]
        mean = y.mean(-1, keepdim=True)
        centered = y - mean
        var = (centered * centered).mean(-1, keepdim=True)
        y = (
            centered * torch.rsqrt(var + eps) * params[2, tile_g, :]
            + params[3, tile_g, :]
        )
        out[tile_m, tile_g, :] = y.to(a3.dtype)
    return out


_KERNELS: dict[tuple, helion.Kernel] = {}


def _swish(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x) * x


def _bias_group_norm(
    y3: torch.Tensor, params: torch.Tensor, eps: hl.constexpr
) -> torch.Tensor:
    """``GroupNorm(y + bias2)`` of ``y3`` ``[M, G, N/G]``, ``params`` as in
    :func:`_gemm_swish_group_norm`."""
    m, groups, group_size = y3.shape
    hl.specialize(group_size)
    out = output_pool.empty_like(y3)
    for tile_m, tile_g in hl.tile([m, groups]):
        y = y3[tile_m, tile_g, :].to(torch.float32) + params[1, tile_g, :]
        mean = y.mean(-1, keepdim=True)
        centered = y - mean
        var = (centered * centered).mean(-1, keepdim=True)
        y = (
            centered * torch.rsqrt(var + eps) * params[2, tile_g, :]
            + params[3, tile_g, :]
        )
        out[tile_m, tile_g, :] = y.to(y3.dtype)
    return out


def _pack_groups_f32(b_t: torch.Tensor, group_size: hl.constexpr) -> torch.Tensor:
    """Transposed weight ``[N, K]`` as one K-major panel per group,
    ``[N/group, K, group]``."""
    n, k = b_t.shape
    group_size = int(group_size)
    out = output_pool.empty(
        (n // group_size, k, group_size), dtype=b_t.dtype, device=b_t.device
    )
    for tk, tn in hl.tile([k, n]):
        out[tn.id, tk, :] = b_t[tn, tk].permute(1, 0)
    return out


def _gemm_swish_group_norm_f32(
    a: torch.Tensor,
    b3: torch.Tensor,
    params: torch.Tensor,
    eps: hl.constexpr,
) -> torch.Tensor:
    """f32 ``_gemm_swish_group_norm`` of ``a`` ``[M, K]`` and ``b3``, one K-major
    panel per group ``[G, K, N/G]``."""
    m, _ = a.shape
    groups, k, group_size = b3.shape
    out = output_pool.empty((m, groups, group_size), dtype=a.dtype, device=a.device)
    for tile_m, tile_g in hl.tile([m, groups]):
        acc = hl.zeros([tile_m, tile_g, group_size], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = acc + torch.einsum(
                "mk,bkn->mbn", a[tile_m, tile_k], b3[tile_g, tile_k, :]
            )
        y = acc + params[0, tile_g, :]
        y = torch.sigmoid(y) * y + params[1, tile_g, :]
        mean = y.mean(-1, keepdim=True)
        centered = y - mean
        var = (centered * centered).mean(-1, keepdim=True)
        y = (
            centered * torch.rsqrt(var + eps) * params[2, tile_g, :]
            + params[3, tile_g, :]
        )
        out[tile_m, tile_g, :] = y
    return out


def _kernel(fn, block_sizes: list[int]) -> helion.Kernel:
    key = (fn.__name__, *block_sizes)
    if key not in _KERNELS:
        _KERNELS[key] = helion.kernel(
            fn,
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=block_sizes),
        )
    return _KERNELS[key]


def _round_up(value: int, block: int) -> int:
    return -(-value // block) * block


def _block_sizes(m: int, pairs: int, groups: int, group_size: int) -> list[int]:
    """Row tiles giving every thread a tile (accumulator within 1 MiB), one group
    per tile, K in one chunk of whole AMX steps."""
    tiles_m = max(1, -(-torch.get_num_threads() // groups))
    tile_m = min(
        _round_up(-(-m // tiles_m), 32), max(32, (1 << 18) // group_size // 32 * 32)
    )
    return [tile_m, 1, pairs]


# K rows per chunk of the f32 kernel: its weight panel chunk stays in L2.
_F32_K_CHUNK = 256


class Model(nn.Module):
    """Matmul, Swish, bias add and GroupNorm as one Helion kernel: GEMM tiles of
    one GroupNorm group each, the rest in the epilogue. The weight is packed
    once (bf16: AMX's VNNI layout; f32: K-major) and reused while it is
    unchanged."""

    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self._cache = None
        self._linear_cache = None

    def _packed(
        self, x: torch.Tensor, fused: bool
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        sources = (
            self.matmul.weight,
            self.matmul.bias,
            self.bias,
            self.group_norm.weight,
            self.group_norm.bias,
        )
        key = (
            tuple((p.data_ptr(), p._version) for p in sources),
            x.dtype,
            x.device,
            fused,
        )
        if self._cache is None or self._cache[0] != key:
            weight = self.matmul.weight.detach().to(dtype=x.dtype, device=x.device)
            n, k = weight.shape
            groups = self.group_norm.num_groups
            group_size = n // groups
            if not fused:
                packed = None
            elif x.dtype == torch.bfloat16:
                packed = _kernel(_pack_groups, [_PACK_PAIRS, group_size])(
                    weight.contiguous().view(n, k // 2, 2), hl.constexpr(group_size)
                )
            else:
                packed = _kernel(_pack_groups_f32, [32, group_size])(
                    weight.contiguous(), hl.constexpr(group_size)
                )
            params = torch.stack(
                [
                    p.detach().to(dtype=torch.float32, device=x.device)
                    for p in sources[1:]
                ]
            ).view(4, groups, group_size)
            self._cache = (key, packed, params.contiguous())
        return self._cache[1], self._cache[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m, k = x.shape
        n = self.matmul.out_features
        groups = self.group_norm.num_groups
        group_size = n // groups
        bf16 = x.dtype == torch.bfloat16
        k_step = 2 * _PAIRS_STEP if bf16 else 32
        if x.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"GEMM+GroupNorm needs bf16 or f32, got {x.dtype}")
        eps = hl.constexpr(self.group_norm.eps)
        # The fused kernel needs whole AMX K steps and 32-channel groups.
        fused = not k % k_step and not group_size % 32
        packed, params = self._packed(x, fused)
        if not fused:
            y, self._linear_cache = linear(x, self.matmul, _swish, self._linear_cache)
            # Rows per tile: at most 32, spread over all threads; one group per tile.
            rows = max(1, min(32, -(-m // torch.get_num_threads())))
            out = _kernel(_bias_group_norm, [rows, 1])(
                y.view(m, groups, group_size), params, eps
            )
            return out.view(m, n)
        if bf16:
            block_sizes = _block_sizes(m, k // 2, groups, group_size)
            out = _kernel(_gemm_swish_group_norm, block_sizes)(
                x.contiguous().view(m, k // 2, 2), packed, params, eps
            )
        else:
            block_sizes = _block_sizes(m, k, groups, group_size)
            block_sizes[2] = min(k, _F32_K_CHUNK)
            out = _kernel(_gemm_swish_group_norm_f32, block_sizes)(
                x.contiguous(), packed, params, eps
            )
        return out.view(m, n)
