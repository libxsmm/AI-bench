# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import cuda.tile as ct
from cutile_cpu_utils.sfc_matmul import _make_sfc_tensor
import torch
import torch.nn as nn

ct.set_backend("cpu")

_BLOCK_M = 32
_BLOCK_K = 32


@ct.kernel
def _fused_groupnorm(
    X,
    Weight,
    Params,
    Out,
    SfcMap,
    EPS: ct.Constant[float],
    BLOCK_M: ct.Constant[int],
    GROUP_SIZE: ct.Constant[int],
):
    coordinates = ct.load(SfcMap, index=(ct.bid(0), 0), shape=(1, 2))
    block_m = ct.extract(coordinates, index=(0, 0), shape=()).item()
    group = ct.extract(coordinates, index=(0, 1), shape=()).item()
    accumulator = ct.full((BLOCK_M, GROUP_SIZE), 0, dtype=ct.float32)
    for block_k in range(X.shape[1]):
        activation = ct.load(
            X, index=(block_m, block_k, 0, 0), shape=(1, 1, BLOCK_M, _BLOCK_K)
        )
        activation = ct.reshape(activation, (BLOCK_M, _BLOCK_K))
        weight = ct.load(
            Weight,
            index=(group, block_k, 0, 0, 0),
            shape=(1, 1, _BLOCK_K // 2, GROUP_SIZE, 2),
        )
        weight = ct.reshape(
            ct.permute(weight, (0, 1, 2, 4, 3)), (_BLOCK_K, GROUP_SIZE)
        )
        accumulator = ct.mma(activation, weight, accumulator)

    linear_bias = ct.reshape(
        ct.load(Params, index=(0, group), shape=(1, GROUP_SIZE)), (GROUP_SIZE,)
    )
    extra_bias = ct.reshape(
        ct.load(Params, index=(1, group), shape=(1, GROUP_SIZE)), (GROUP_SIZE,)
    )
    norm_weight = ct.reshape(
        ct.load(Params, index=(2, group), shape=(1, GROUP_SIZE)), (GROUP_SIZE,)
    )
    norm_bias = ct.reshape(
        ct.load(Params, index=(3, group), shape=(1, GROUP_SIZE)), (GROUP_SIZE,)
    )
    values = accumulator + linear_bias[None, :]
    values = values / (1.0 + ct.exp(-values)) + extra_bias[None, :]
    mean = ct.sum(values, axis=1, keepdims=True) / GROUP_SIZE
    centered = values - mean
    variance = ct.sum(centered * centered, axis=1, keepdims=True) / GROUP_SIZE
    inverse_std = 1.0 / ct.sqrt(variance + EPS)
    values = centered * inverse_std
    values = values * norm_weight[None, :] + norm_bias[None, :]
    ct.store(Out, index=(block_m, group), tile=values.astype(Out.dtype))


class Model(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.num_groups = num_groups
        self._cache = None

    def forward(self, x):
        group_size = self.matmul.out_features // self.num_groups
        if (
            x.device.type != "cpu"
            or x.dtype != torch.bfloat16
            or x.shape[0] % _BLOCK_M
            or x.shape[1] % _BLOCK_K
            or group_size < 32
            or group_size > 256
            or group_size & (group_size - 1)
        ):
            return self.group_norm(torch.nn.functional.silu(self.matmul(x)) + self.bias)

        sources = (
            self.matmul.weight,
            self.matmul.bias,
            self.bias,
            self.group_norm.weight,
            self.group_norm.bias,
        )
        key = (
            tuple((param.data_ptr(), param._version) for param in sources),
            x.dtype,
            x.device,
        )
        if self._cache is None or self._cache[0] != key:
            weight = self.matmul.weight.detach().to(dtype=x.dtype, device=x.device)
            packed = (
                weight.reshape(self.num_groups, group_size, x.shape[1] // 2, 2)
                .permute(0, 2, 1, 3)
                .contiguous()
                .view(
                    self.num_groups,
                    x.shape[1] // _BLOCK_K,
                    _BLOCK_K // 2,
                    group_size,
                    2,
                )
            )
            params = torch.stack(
                [
                    param.detach().to(dtype=torch.float32, device=x.device)
                    for param in sources[1:]
                ]
            ).contiguous()
            self._cache = (key, packed, params)

        x = x.contiguous()
        out = torch.empty(
            (x.shape[0], self.matmul.out_features), dtype=x.dtype, device=x.device
        )
        blocks_m = x.shape[0] // _BLOCK_M
        sfc_map = _make_sfc_tensor(blocks_m, self.num_groups)
        with ct.backend_options({"assume_in_bounds": True}):
            ct.launch(
                None,
                (blocks_m * self.num_groups,),
                _fused_groupnorm,
                (
                    x.view(
                        blocks_m,
                        _BLOCK_M,
                        x.shape[1] // _BLOCK_K,
                        _BLOCK_K,
                    ).permute(0, 2, 1, 3),
                    self._cache[1],
                    self._cache[2],
                    out,
                    sfc_map,
                    self.group_norm.eps,
                    _BLOCK_M,
                    group_size,
                ),
            )
        return out
