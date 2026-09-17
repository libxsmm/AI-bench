# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import cuda.tile as ct
from cuda.tile._backend import cpu
from cutile_cpu_utils import groupnorm
from cutile_cpu_utils import pack_weights_for_sfc_matmul
from cutile_cpu_utils import reduce_last_dim
from cutile_cpu_utils import sfc_matmul
import torch
import torch.nn as nn

ct.set_backend("cpu")


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def _min_init_val(dtype, **kwargs):
    return ct.full((), float("inf"), dtype=dtype)


def _min_reduction(value, block, **kwargs):
    return ct.minimum(value, ct.min(block))


@ct.kernel
def _broadcast_bias(
    Bias,
    Reduced,
    Output,
    B: ct.Constant[int],
    R: ct.Constant[int],
    BLOCK_SIZE_R: ct.Constant[int],
):
    batch = ct.bid(0)
    bias = ct.load(Bias, index=(batch,), shape=())
    for block in range(R // BLOCK_SIZE_R):
        reduced = ct.load(
            Reduced,
            index=(block,),
            shape=(BLOCK_SIZE_R,),
        )
        values = reduced + bias
        ct.store(
            Output,
            index=(batch, block),
            tile=ct.reshape(values.astype(Output.dtype), (1, BLOCK_SIZE_R)),
        )


class Model(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.num_groups = num_groups
        self._weight_packed = None
        self._bias = None
        self._bias_extra = None

    def forward(self, x):
        x = x.contiguous()
        if self._weight_packed is None or self._weight_packed.dtype != x.dtype:
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.gemm.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias = self.gemm.bias.data.to(dtype=x.dtype).contiguous()
            self._bias_extra = self.bias.data.to(dtype=x.dtype).contiguous()
            assert self.group_norm.affine, "GroupNorm must have affine=True"
            assert self._bias_extra.shape[-2:] == (1, 1)

        res_mm = sfc_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            trunc_output=False,
            b_is_prepacked=True,
            c_is_owned=True,
            blocking_factor_k=_next_power_of_2(max(1, x.shape[1] // 4096)),
        )
        res_gn = groupnorm(
            res_mm,
            out_dtype=res_mm.dtype,
            num_groups=self.num_groups,
            eps=self.group_norm.eps,
            try_inplace=True,
        )
        res_red = reduce_last_dim(
            res_gn,
            out_dtype=res_gn.dtype,
            init_val=_min_init_val,
            reduction_op=_min_reduction,
            keep_dim=True,
        )
        output_shape = torch.broadcast_shapes(res_red.shape, self._bias_extra.shape)
        output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        n_bias_elements = int(torch.tensor(self._bias_extra.shape[:-2]).prod().item())
        block_size_r = 256
        assert res_red.shape[0] % block_size_r == 0
        with cpu.compile_options({"assume_in_bounds": True}):
            ct.launch(
                None,
                (n_bias_elements,),
                _broadcast_bias,
                (
                    self._bias_extra.view(-1),
                    res_red.view(-1),
                    output.view(n_bias_elements, res_red.shape[0]),
                    n_bias_elements,
                    res_red.shape[0],
                    block_size_r,
                ),
            )
        return output
