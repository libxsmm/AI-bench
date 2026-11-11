"""Linear helpers with optional deployment-style weight prepacking."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import Callable
from typing import NamedTuple

import torch

from .matmul import identity_epilogue
from .matmul import matmul
from .matmul import matmul_prepacked_b
from .matmul import pack_b_blocked_t
from .matmul import pack_b_vnni_t
from .matmul import prepacked_b_call
from .matmul import vnni_panel
from .matmul import whole_row_tiles

if TYPE_CHECKING:
    from torch import Tensor
    from torch import nn

CACHE_PREPACKED_WEIGHTS_ENV = "HELION_MLIR_CACHE_PREPACKED_WEIGHTS"


def cache_prepacked_weights() -> bool:
    """Whether constant weights are packed once and reused (the default);
    ``HELION_MLIR_CACHE_PREPACKED_WEIGHTS=0`` packs them on every call."""
    value = os.environ.get(CACHE_PREPACKED_WEIGHTS_ENV)
    return value is None or value.strip().lower() not in ("0", "false", "no")


class LinearCache(NamedTuple):
    key: tuple
    packed_weight: Tensor
    bias: Tensor | None
    out_features: int
    call: tuple | None = None
    """``(epilogue, OMP_NUM_THREADS, prepacked_b_call(...))`` of the last call."""


class AffineLinearCache(NamedTuple):
    key: tuple
    packed_weight: Tensor
    bias: Tensor | None
    post_scale: Tensor
    post_bias: Tensor | None
    out_features: int
    call: tuple | None = None
    """``(epilogue, OMP_NUM_THREADS, prepacked_b_call(...))`` of the last call."""


def _parameter_key(parameter: Tensor | None) -> tuple | None:
    if parameter is None:
        return None
    return (
        parameter.data_ptr(),
        parameter._version,
        tuple(parameter.shape),
        parameter.dtype,
        parameter.device,
    )


def _combine_biases(biases: tuple[Tensor, ...], x: Tensor) -> Tensor | None:
    if not biases:
        return None
    bias = biases[0].to(dtype=x.dtype, device=x.device)
    for extra_bias in biases[1:]:
        bias = bias + extra_bias.to(dtype=x.dtype, device=x.device)
    return bias


def _pack_weight(weight: Tensor, rows: int, panel: int | None = None) -> Tensor:
    """``[N, K]`` weight packed for :func:`matmul_prepacked_b` of ``rows`` rows:
    AMX's VNNI panels for bf16 with an even K (``panel`` columns wide, else
    :func:`vnni_panel`'s choice), else 32x32 blocks."""
    if weight.dtype == torch.bfloat16 and not weight.shape[1] % 2:
        if panel is None:
            n, k = map(int, weight.shape)
            panel = vnni_panel(rows, n, k)
        return pack_b_vnni_t(weight, panel)
    return pack_b_blocked_t(weight)


def _whole_row_tiles(
    x: Tensor, layer: nn.Linear, whole_rows: bool, panel: int | None
) -> tuple[int, int] | None:
    """:func:`~.matmul.whole_row_tiles` of a bf16 layer if ``whole_rows``."""
    if not whole_rows or x.dtype != torch.bfloat16 or layer.in_features % 2:
        return None
    return whole_row_tiles(int(x.shape[0]), int(layer.out_features), panel)


def linear(
    x: Tensor,
    layer: nn.Linear,
    epilogue: Callable[[Tensor], Tensor] = identity_epilogue,
    cache: LinearCache | None = None,
    biases: tuple[Tensor, ...] | None = None,
    panel: int | None = None,
    whole_rows: bool = False,
) -> tuple[Tensor, LinearCache | None]:
    """Run a linear layer, by default with its constant weight packed once and
    reused until the source parameters, input dtype, or device change;
    ``HELION_MLIR_CACHE_PREPACKED_WEIGHTS=0`` packs it on every call.
    ``panel`` overrides the columns of a bf16 weight's VNNI panels, which set
    the output tile's width (a multiple of 32; default :func:`vnni_panel`).

    The cache, and the prepared call in it, is keyed by the parameters' data
    pointers and ``_version``: in-place ops and ``load_state_dict`` are
    seen, but a change through ``.data`` (``p.data.mul_(2)``) changes
    neither, so the packed weight goes stale; pass ``cache=None`` after one.

    ``whole_rows``: each thread computes whole rows of the output where
    :func:`~.matmul.whole_row_tiles` allows it (bf16, cached weights), so each
    row of ``x`` is read by one core; for an ``x`` rewritten by a later kernel.
    It sets the tile's rows, and its columns unless ``panel`` does. The panel
    stays as packed if ``OMP_NUM_THREADS`` changes later; tiles it cannot
    serve within the accumulator limit take the default rows.
    """
    if biases is None:
        bias = layer.bias  # (a module attribute lookup: once per call)
        source_biases = () if bias is None else (bias,)
    else:
        source_biases = biases
    if not cache_prepacked_weights():
        weight = layer.weight.to(dtype=x.dtype, device=x.device)
        return (
            matmul(
                x,
                weight,
                trans_b=True,
                bias=_combine_biases(source_biases, x),
                epilogue=epilogue,
            ),
            None,
        )

    key = (
        _parameter_key(layer.weight),
        tuple(map(_parameter_key, source_biases)),
        x.dtype,
        x.device,
        x.shape[0],
        panel,
        whole_rows,
    )
    if cache is None or cache.key != key:
        weight = layer.weight.detach().to(dtype=x.dtype, device=x.device)
        bias = _combine_biases(source_biases, x)
        tiles = _whole_row_tiles(x, layer, whole_rows, panel)
        cache = LinearCache(
            key,
            _pack_weight(weight, int(x.shape[0]), tiles[1] if tiles else panel),
            bias,
            int(layer.out_features),
        )
    # The call's host work (views, tile choice, kernel lookup) is done once:
    # per call it costs about as much as a small layer's kernel.
    threads = os.environ.get("OMP_NUM_THREADS")
    prepared = cache.call
    if prepared is None or prepared[0] is not epilogue or prepared[1] != threads:
        tiles = _whole_row_tiles(
            x, layer, whole_rows, int(cache.packed_weight.shape[2])
        )
        call = prepacked_b_call(
            x,
            cache.packed_weight,
            cache.out_features,
            cache.bias,
            epilogue,
            tile_m=tiles and tiles[0],
        )
        cache = cache._replace(call=(epilogue, threads, call))
    else:
        call = prepared[2]
    return call(x), cache


def linear_affine(
    x: Tensor,
    layer: nn.Linear,
    post_scale: Tensor,
    post_bias: Tensor | None = None,
    epilogue: Callable[[Tensor], Tensor] = identity_epilogue,
    cache: AffineLinearCache | None = None,
) -> tuple[Tensor, AffineLinearCache | None]:
    """Run ``epilogue((linear(x)) * post_scale + post_bias)``; ``post_scale`` and
    ``post_bias`` are ``[1]`` or ``[out_features]``. Cached like :func:`linear`
    (changes through ``.data`` are not seen either)."""
    use_cache = cache_prepacked_weights()
    if x.dtype == torch.bfloat16 and not use_cache:
        bias = None if layer.bias is None else layer.bias.to(dtype=x.dtype)
        result = matmul(
            x,
            layer.weight.to(dtype=x.dtype, device=x.device),
            trans_b=True,
            bias=bias,
            epilogue=epilogue,
            scale=post_scale,
            shift=post_bias,
        )
        return result, None
    key = (
        _parameter_key(layer.weight),
        _parameter_key(layer.bias),
        _parameter_key(post_scale),
        _parameter_key(post_bias),
        x.dtype,
        x.device,
        x.shape[0],
    )
    if not use_cache or cache is None or cache.key != key:
        weight = layer.weight.detach().to(dtype=x.dtype, device=x.device)
        bias = (
            None
            if layer.bias is None
            else layer.bias.detach().to(dtype=x.dtype, device=x.device)
        )
        cache = AffineLinearCache(
            key,
            _pack_weight(weight, int(x.shape[0])),
            bias,
            post_scale.detach().to(dtype=x.dtype, device=x.device),
            None
            if post_bias is None
            else post_bias.detach().to(dtype=x.dtype, device=x.device),
            int(layer.out_features),
        )

    if not use_cache:
        result = matmul_prepacked_b(
            x,
            cache.packed_weight,
            cache.out_features,
            cache.bias,
            epilogue,
            cache.post_scale,
            cache.post_bias,
        )
        return result, None
    # The call's host work is done once, as in linear().
    threads = os.environ.get("OMP_NUM_THREADS")
    prepared = cache.call
    if prepared is None or prepared[0] is not epilogue or prepared[1] != threads:
        call = prepacked_b_call(
            x,
            cache.packed_weight,
            cache.out_features,
            cache.bias,
            epilogue,
            cache.post_scale,
            cache.post_bias,
        )
        cache = cache._replace(call=(epilogue, threads, call))
    else:
        call = prepared[2]
    return call(x), cache
