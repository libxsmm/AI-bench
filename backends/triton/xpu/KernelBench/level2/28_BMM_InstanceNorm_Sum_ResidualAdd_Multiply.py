# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _get_autotune_configs():
    # Keep the search space focused for this simple 1D elementwise kernel.
    # The workload is large (~8.4M elements), so prioritize medium/large blocks
    # and include at least one 32-warp large-tile config for Intel XPU.
    return [
        # Baseline-compatible configs
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=32, num_stages=2),
        # XPU-oriented extensions, but kept conservative to avoid regressions
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=32, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=32, num_stages=3),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=32, num_stages=2),
    ]


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["n_elements"],
)
@triton.jit
def _mul_kernel_1d(
    Y_ptr,
    OUT_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid.to(tl.int64) * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    y = tl.load(Y_ptr + offs, mask=mask, other=0.0)
    tl.store(OUT_ptr + offs, y * y, mask=mask)


def kernel_function(y):
    assert y.dim() == 2, "Expected y to be a 2D tensor"

    y_xpu = y
    if y_xpu.device.type != "xpu":
        y_xpu = y_xpu.to(device="xpu", dtype=torch.float16)
    elif y_xpu.dtype != torch.float16:
        y_xpu = y_xpu.to(dtype=torch.float16)

    if not y_xpu.is_contiguous():
        y_xpu = y_xpu.contiguous()

    out = torch.empty_like(y_xpu)
    n_elements = y_xpu.numel()

    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)

    _mul_kernel_1d[grid](
        y_xpu,
        out,
        n_elements,
        grf_mode="auto",
    )
    return out


batch_size = 1024
in_features = 8192
out_features = 8192


def get_inputs():
    return [
        torch.rand(batch_size, in_features, dtype=torch.float16),
        torch.rand(batch_size, out_features, dtype=torch.float16),
    ]


def get_init_inputs():
    return [in_features, out_features]


class Model(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.momentum = momentum

    def forward(self, x, y):
        return kernel_function(y)
