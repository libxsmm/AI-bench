import math

import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

# Growth of a row maximum (in scaled logits) below which the stale maximum is
# kept: exponentials stay below e^8, well within f32 and bf16 range.
_RESCALE_THRESHOLD = 8.0

# A partial key tile's padded lanes score 0, so a row whose real scores are all far
# below 0 gives them ``exp(0 - top)``: inf, and the V padding's zeros then make it
# NaN in the dot. Scores of real lanes stay below _RESCALE_THRESHOLD; this bound,
# past it, keeps the padded ones finite, and those V zeros then cancel them. Only
# with such tiles, as it has a cost.
_PADDED_LANE_MAX = 16.0


def _dot_nt(a: torch.Tensor, b: torch.Tensor, vnni: bool) -> torch.Tensor:
    """``a @ b.T`` of ``[M, K]`` and ``[N, K]``; bf16 operands in AMX's VNNI
    layouts, ``[M, K/2, 2]`` and ``[K/2, N, 2]``."""
    if not vnni:
        return torch.einsum("mk,nk->mn", a, b)
    a3 = a.reshape(a.size(0), a.size(1) // 2, 2)
    b3 = b.reshape(b.size(0), b.size(1) // 2, 2).permute(1, 0, 2)
    return torch.einsum("mcv,cnv->mn", a3, b3)


def _dot_nn(a: torch.Tensor, b: torch.Tensor, vnni: bool) -> torch.Tensor:
    """``a @ b`` of ``[M, K]`` and ``[K, N]``, VNNI layouts as in ``_dot_nt``."""
    if not vnni:
        return torch.einsum("mk,kn->mn", a, b)
    a3 = a.reshape(a.size(0), a.size(1) // 2, 2)
    b3 = b.reshape(b.size(0) // 2, 2, b.size(1)).permute(0, 2, 1)
    return torch.einsum("mcv,cnv->mn", a3, b3)


def _exp_bf16(x: torch.Tensor) -> torch.Tensor:
    """``exp(x)`` of f32 ``x <= _RESCALE_THRESHOLD`` at bf16's accuracy: a cubic
    in ``x / 64``, squared six times. The relative error is about
    ``x^4 / (24 * 64^3)``: below 8e-4 for ``|x| <= 8`` (2.5e-6 at 2), under
    bf16's rounding of the weights. It grows further down, 0.22 at ``x = -32``:
    with ``V`` spanning over about 1e6, such a weight can show in the output.
    The clamp is safely past where the cubic's 64th power underflows (denormal
    from about -74, 0 by -84), so ``-inf`` scores weigh nothing, as with
    ``exp``; the cubic is positive down to it. It's cheaper than the f32-accurate
    polynomial and its clamps."""
    y = torch.clamp(x, min=-96.0) * 0.015625
    p = 1.0 + y * (1.0 + y * (0.5 + y * (1.0 / 6.0)))
    p = p * p
    p = p * p
    p = p * p
    p = p * p
    p = p * p
    return p * p


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: hl.constexpr,
    vnni: hl.constexpr,
    rows: hl.constexpr,
    padded_dim: hl.constexpr,
    partial_keys: hl.constexpr,
) -> torch.Tensor:
    """Softmax attention of ``[BH, S, D]`` heads with logits scaled by ``scale``,
    online over key blocks: the running row maxima rescale the sums and the
    output accumulator, only when one of them grows past ``_RESCALE_THRESHOLD``."""
    heads, seq, dim = q.shape
    hl.specialize(seq)
    hl.specialize(dim)
    out = torch.empty_like(q)
    for h in hl.grid(heads):
        # Tiles up to rows and padded_dim, past the operands' ends: a load there
        # reads zeros, which change no dot product and no other row, a store
        # there is dropped.
        for tile_m in hl.tile(int(rows)):
            for tile_d in hl.tile(int(padded_dim)):
                q_tile = q[h, tile_m, tile_d]
                # -FLT_MAX, below any finite score. Finite: Helion's value range
                # analysis of the growth takes -inf - -inf.
                top = hl.full([tile_m], -3.4028234663852886e38, dtype=torch.float32)
                total = hl.zeros([tile_m], dtype=torch.float32)
                acc = hl.zeros([tile_m, tile_d], dtype=torch.float32)
                for tile_n in hl.tile(seq):
                    scores = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                    scores = scores + _dot_nt(q_tile, k[h, tile_n, tile_d], vnni)
                    maxima = torch.amax(scores, dim=-1) * scale
                    # Not "> threshold": a NaN row must open this too, else every
                    # other row of the block keeps its stale top and overflows.
                    if torch.logical_not(
                        torch.amax(maxima - top) <= _RESCALE_THRESHOLD
                    ):
                        new_top = torch.maximum(top, maxima)
                        alpha = torch.exp(top - new_top)
                        total = total * alpha
                        acc = acc * alpha[:, None]
                        top = new_top
                    x = scores * scale - top[:, None]
                    if partial_keys:
                        x = torch.clamp(x, max=_PADDED_LANE_MAX)
                    if vnni:
                        p = _exp_bf16(x)
                    else:
                        p = torch.exp(x)
                    total = total + p.sum(-1)
                    acc = acc + _dot_nn(p.to(q.dtype), v[h, tile_n, tile_d], vnni)
                # A row with no weight at all (all scores -inf) is 0, not 0 / 0;
                # too small to change a sum of weights, whose largest is near 1.
                y = acc / (total[:, None] + 1.0e-37)
                if partial_keys:
                    # Its -inf times a padded key's zeros made a NaN in acc.
                    y = torch.where(total[:, None] == 0.0, torch.zeros_like(y), y)
                out[h, tile_m, tile_d] = y.to(q.dtype)
    return out


_KERNELS: dict[tuple[int, int, int], helion.Kernel] = {}


def _kernel(block_m: int, block_d: int, block_n: int) -> helion.Kernel:
    key = (block_m, block_d, block_n)
    if key not in _KERNELS:
        _KERNELS[key] = helion.kernel(
            _attention,
            backend="mlir",
            config=helion.Config(block_sizes=[block_m, block_d, block_n]),
        )
    return _KERNELS[key]


def _call(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> tuple:
    """The kernel of ``[B, H, S, D]`` operands and its arguments."""
    b, h, s, d = Q.shape
    heads = b * h
    scale = 1.0 / math.sqrt(d)
    # bf16 runs on AMX, its operands in VNNI layout: pairs along D and along
    # the key block. A sequence of one forms no pair of keys, and one padded
    # column aborts LLVM (a masked load of one lane): plain bf16 dots.
    vnni = Q.dtype == torch.bfloat16 and s >= 2 and d >= 2
    # AMX dots whole register tiles: D's one tile and the query tiles are padded
    # to multiples of 32, and key tiles rounded down to one, but a last partial
    # one, which is masked. Partial, a contraction falls off AMX into
    # per-element code.
    block_d = (d + 31) // 32 * 32 if vnni else d
    # Tall query blocks pack each K/V block for many rows; swept on EMR
    # (S2048 D128, ms: 128 rows 13.1, 256 12.3, 512 11.8, 1024 12.1).
    block_m = 512 if block_d <= 128 else 256
    rows = s
    if not vnni:
        block_m = min(s, block_m)
    else:
        # Tiles of equal rows, so that padding costs about 32 rows per tile at
        # most, not most of a tile.
        padded_s = (s + 31) // 32 * 32
        tiles = -(-padded_s // block_m)
        block_m = (-(-padded_s // tiles) + 31) // 32 * 32
        rows = tiles * block_m
    # VNNI pairs along the key block need it even: a block of at least S is
    # all of S, so an odd S gets one less and a masked tail. Below 32 keys no
    # register tile fits: masked, but few.
    block_n = min(s - s % 2 if vnni else s, 128 if block_d <= 64 else 64)
    if vnni and s >= 32:
        block_n = block_n // 32 * 32
    return _kernel(block_m, block_d, block_n), (
        Q.reshape(heads, s, d),
        K.reshape(heads, s, d),
        V.reshape(heads, s, d),
        hl.constexpr(scale),
        hl.constexpr(vnni),
        hl.constexpr(rows),
        hl.constexpr(block_d),
        hl.constexpr(s % block_n != 0),
    )


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        kernel, args = _call(Q, K, V)
        return kernel(*args).view(Q.shape)
