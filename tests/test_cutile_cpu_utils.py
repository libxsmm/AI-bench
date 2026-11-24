"""Persistent path of the cuTile CPU SFC matmul.

The checks compile kernels and run only with ``AI_BENCH_CUTILE_KERNEL_TESTS=1``.
"""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

ct = pytest.importorskip("cuda.tile")

from cutile_cpu_utils import pack_weights_for_sfc_matmul  # noqa: E402
from cutile_cpu_utils import sfc_matmul  # noqa: E402
from cutile_cpu_utils.sfc_matmul import _persistent_tiles_per_program  # noqa: E402
from cutile_cpu_utils.sfc_matmul import _persistent_workers  # noqa: E402
from cutile_cpu_utils.sfc_matmul import prepare_sfc_matmul  # noqa: E402

KERNEL_TESTS_ENV = "AI_BENCH_CUTILE_KERNEL_TESTS"
kernel_test = pytest.mark.skipif(
    os.environ.get(KERNEL_TESTS_ENV) != "1",
    reason=f"compiles cuTile kernels; set {KERNEL_TESTS_ENV}=1 to run",
)


def _relu(value):
    return ct.maximum(value, 0.0)


def _scale_columns(value, block_n, post_op_arg_ptr, **kwargs):
    scale = ct.load(post_op_arg_ptr, index=(block_n,), shape=(32,))
    return value * scale.astype(value.dtype)[None, :]


def _scale_rows(value, block_m, post_op_arg_ptr, BLOCK_SIZE_M, **kwargs):
    scale = ct.load(post_op_arg_ptr, index=(block_m,), shape=(BLOCK_SIZE_M,))
    return value * scale.astype(value.dtype)[:, None]


def _subtract_column_sum(value):
    return value - ct.sum(value, axis=0, keepdims=True)


def _openmp(linked, global_runtime):
    return patch(
        "cutile_cpu_utils.sfc_matmul._openmp_libraries",
        return_value=(
            SimpleNamespace(omp_get_max_threads=linked),
            SimpleNamespace(omp_get_max_threads=global_runtime),
        ),
    )


@pytest.mark.parametrize(
    ("tile_count", "workers", "expected"),
    [
        (65536, 64, 256),
        (2048, 64, 8),
        (1024, 64, 4),
        (1000, 64, 2),
        (64, 64, 1),
        (32, 64, 1),
        (4096, 16, 64),
        (4096, 128, 8),
        (65536, None, 16),
        (2048, None, 2),
        (64, None, 1),
    ],
)
def test_persistent_tiles_per_program(tile_count, workers, expected):
    assert _persistent_tiles_per_program(tile_count, workers) == expected


def test_persistent_workers():
    with _openmp(lambda: 48, lambda: 48):
        assert _persistent_workers(None) == 48
        assert _persistent_workers({"num_cpu_threads": 0}) == 48
        assert _persistent_workers({"num_cpu_threads": 64}) == 64


def test_persistent_workers_with_multiple_openmp_runtimes():
    counts = {"linked": 37, "global": 5}
    with _openmp(lambda: counts["linked"], lambda: counts["global"]):
        assert _persistent_workers(None) is None
        assert _persistent_workers({"num_cpu_threads": 48}) == 48
        counts["global"] = 37
        assert _persistent_workers(None) == 37
        counts["linked"] = 256
        assert _persistent_workers(None) is None


@pytest.mark.parametrize(("workers", "tiles", "programs"), [(16, 32, 64), (64, 8, 256)])
def test_persistent_launch_grouping(workers, tiles, programs):
    a = torch.empty((2048, 512), dtype=torch.bfloat16)
    b = torch.empty((512, 1024), dtype=torch.bfloat16)
    prepared = prepare_sfc_matmul(
        a,
        b,
        b_is_prepacked=True,
        persistent=True,
        options={"num_cpu_threads": workers},
    )
    grid, _, args = prepared._launches[0]
    assert grid == (programs, 1, 1)
    assert args[10] == tiles


def test_persistent_launch_grouping_uses_openmp():
    a = torch.empty((2048, 512), dtype=torch.bfloat16)
    b = torch.empty((512, 1024), dtype=torch.bfloat16)
    with _openmp(lambda: 64, lambda: 64):
        prepared = prepare_sfc_matmul(a, b, b_is_prepacked=True, persistent=True)
    grid, _, args = prepared._launches[0]
    assert grid == (256, 1, 1)
    assert args[10] == 8


def test_persistent_disabled_by_default_and_explicitly_enabled():
    a = torch.empty((64, 512), dtype=torch.bfloat16)
    b = torch.empty((512, 64), dtype=torch.bfloat16)
    default = prepare_sfc_matmul(a, b, b_is_prepacked=True)
    enabled = prepare_sfc_matmul(a, b, b_is_prepacked=True, persistent=True)
    disabled = prepare_sfc_matmul(a, b, b_is_prepacked=True, persistent=False)
    assert default._launches[0][1] is disabled._launches[0][1]
    assert default._launches[0][1] is not enabled._launches[0][1]


def test_custom_callbacks_require_rowwise_opt_in():
    a = torch.empty((64, 512), dtype=torch.bfloat16)
    b = torch.empty((512, 64), dtype=torch.bfloat16)
    kwargs = {"b_is_prepacked": True, "post_op": _subtract_column_sum}
    default = prepare_sfc_matmul(a, b, **kwargs)
    unchecked = prepare_sfc_matmul(a, b, persistent=True, **kwargs)
    rowwise = prepare_sfc_matmul(
        a, b, persistent=True, rowwise_callbacks=True, **kwargs
    )
    assert unchecked._launches[0][1] is default._launches[0][1]
    assert rowwise._launches[0][1] is not default._launches[0][1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"b_is_prepacked": False},
        {"b_is_prepacked": True, "reduce_last_dim": True},
        {"b_is_prepacked": True, "softmax_last_dim": True},
        {"b_is_prepacked": True, "c_is_owned": True},
    ],
)
def test_persistent_rejects_unsupported_modes(kwargs):
    a = torch.empty((64, 512), dtype=torch.bfloat16)
    b = torch.empty((512, 64), dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="persistent=True"):
        prepare_sfc_matmul(a, b, persistent=True, **kwargs)


@kernel_test
def test_cross_row_callback_keeps_full_tile_semantics():
    a = torch.arange(64, dtype=torch.float32).reshape(64, 1).repeat(1, 512)
    a = a.to(torch.bfloat16)
    packed = pack_weights_for_sfc_matmul(torch.eye(512, dtype=torch.bfloat16), 32, 32)
    kwargs = {"b_is_prepacked": True, "post_op": _subtract_column_sum}
    expected = sfc_matmul(a, packed, persistent=False, **kwargs)
    result = sfc_matmul(a, packed, persistent=True, **kwargs)
    assert torch.count_nonzero(expected)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@kernel_test
def test_persistent_rowwise_callback_uses_row_location():
    torch.manual_seed(0)
    a = torch.randn(64, 512).to(torch.bfloat16)
    weights = torch.randn(64, 512).to(torch.bfloat16)
    packed = pack_weights_for_sfc_matmul(weights, 32, 32)
    kwargs = {
        "b_is_prepacked": True,
        "post_op": _scale_rows,
        "post_op_arg": torch.arange(64, dtype=torch.float32) / 64,
    }
    expected = sfc_matmul(a, packed, persistent=False, **kwargs)
    result = sfc_matmul(a, packed, persistent=True, rowwise_callbacks=True, **kwargs)
    torch.testing.assert_close(result, expected)


@kernel_test
def test_persistent_long_k_matches_reference_with_blocking():
    torch.manual_seed(0)
    a = torch.randn(64, 4096).to(torch.bfloat16)
    weights = torch.randn(64, 4096).to(torch.bfloat16)
    packed = pack_weights_for_sfc_matmul(weights, 32, 32)
    reference = a.float() @ weights.float().T
    kwargs = {"b_is_prepacked": True, "blocking_factor_k": 4, "trunc_output": False}
    for persistent in (False, True):
        result = sfc_matmul(a, packed, persistent=persistent, **kwargs)
        torch.testing.assert_close(result, reference, rtol=0.005, atol=0.002)


@kernel_test
@pytest.mark.parametrize(
    ("filename", "model_args"),
    [
        ("39_Gemm_Scale_BatchNorm.py", (512, 64, (64,))),
        ("40_Matmul_Scaling_ResidualAdd.py", (512, 64, 0.5)),
        ("41_Gemm_BatchNorm_GELU_ReLU.py", (512, 64)),
    ],
)
def test_persistent_kernel_models_match_original(filename, model_args, monkeypatch):
    path = (
        Path(__file__).resolve().parents[1]
        / "backends/cutile/cpu/KernelBench/level2"
        / filename
    )
    spec = importlib.util.spec_from_file_location("cutile_sfc_test_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(0)
    model = module.Model(*model_args)
    a = torch.randn(64, 512).to(torch.bfloat16)
    calls = []

    def persistent_call(*args, **kwargs):
        calls.append(kwargs)
        return sfc_matmul(*args, **kwargs)

    monkeypatch.setattr(module, "sfc_matmul", persistent_call)
    result = model(a)
    assert calls[0]["persistent"] and calls[0]["rowwise_callbacks"]

    def original_call(*args, **kwargs):
        kwargs["persistent"] = False
        return sfc_matmul(*args, **kwargs)

    monkeypatch.setattr(module, "sfc_matmul", original_call)
    expected = model(a)
    torch.testing.assert_close(result, expected)


@kernel_test
@pytest.mark.parametrize("case", ["bias", "post_op", "post_op_arg"])
@pytest.mark.parametrize(
    ("dtype", "m", "k"),
    [
        (torch.bfloat16, 2048, 512),  # persistent
        (torch.float32, 64, 64),  # falls back to the default path
    ],
)
def test_persistent_matches_default(case, dtype, m, k):
    """Persistent and non-persistent paths agree for each epilogue."""
    n = m // 2
    torch.manual_seed(0)
    a = torch.randn(m, k).to(dtype)
    b = pack_weights_for_sfc_matmul(
        torch.randn(n, k).to(dtype), BLOCK_SIZE_N=32, BLOCK_SIZE_K=32
    )
    bias = torch.randn(n).to(dtype)
    kwargs = {
        "bias": {"bias": bias, "trunc_output": False},
        "post_op": {"bias": bias, "post_op": _relu},
        "post_op_arg": {
            "bias": bias,
            "post_op": _scale_columns,
            "post_op_arg": torch.randn(n).to(dtype),
        },
    }[case]
    expected = sfc_matmul(a, b, b_is_prepacked=True, persistent=False, **kwargs).clone()
    actual = sfc_matmul(
        a, b, b_is_prepacked=True, persistent=True, rowwise_callbacks=True, **kwargs
    )
    torch.testing.assert_close(actual, expected)
