"""Focused tests for the cuTile CPU SFC matmul helper."""

import functools
import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

ct = pytest.importorskip("cuda.tile")

from cutile_cpu_utils import pack_weights_for_sfc_matmul  # noqa: E402
from cutile_cpu_utils import region_matmul  # noqa: E402
from cutile_cpu_utils import sfc_matmul  # noqa: E402
from cutile_cpu_utils.region_matmul import _make_region_tensor  # noqa: E402
from cutile_cpu_utils.sfc_matmul import prepare_sfc_matmul  # noqa: E402

from ai_bench.harness.runner.benchmark_compare import copy_model_weights  # noqa: E402

KERNEL_TESTS_ENV = "AI_BENCH_CUTILE_KERNEL_TESTS"
kernel_test = pytest.mark.skipif(
    os.environ.get(KERNEL_TESTS_ENV) != "1",
    reason=f"compiles cuTile kernels; set {KERNEL_TESTS_ENV}=1 to run",
)


def _force_second_autotune_candidate():
    def make_autotuned(*args, **kwargs):
        options = dict(args[2])

        def autotuned(a, b, **call_kwargs):
            result = None
            for programs in (1, 2):
                result = prepare_sfc_matmul(
                    a,
                    b,
                    blocking_factor_k=1,
                    progs_per_thread=programs,
                    options=options,
                    **call_kwargs,
                )()
            return result

        return autotuned

    return patch(
        "cutile_cpu_utils.sfc_matmul._make_autotuned_sfc_matmul",
        side_effect=make_autotuned,
    )


def _subtract_column_sum(value):
    return value - ct.sum(value, axis=0, keepdims=True)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@kernel_test
@pytest.mark.parametrize(
    "kwargs",
    [
        {"post_op": _subtract_column_sum},
        {"blocking_factor_k": 2, "progs_per_thread": 2},
        {"reduce_last_dim": True, "trunc_output": False},
        {"softmax_last_dim": True},
    ],
)
def test_region_order_matches_sfc_order_bitwise(kwargs):
    a, _, packed = _inputs(m=512, n=256, k=1024)
    common = {
        "bias": torch.randn(256, dtype=torch.bfloat16),
        "b_is_prepacked": True,
        "blocking_factor_k": 1,
        "progs_per_thread": 1,
        "options": {"num_cpu_threads": 4},
        **kwargs,
    }
    assert _make_region_tensor(16, 8, 4 * common["progs_per_thread"], 4) is not None
    expected = sfc_matmul(a, packed, **common).clone()
    actual = region_matmul(a, packed, **common)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@kernel_test
@pytest.mark.parametrize(
    ("problem", "extra_args"),
    [
        ("39_Gemm_Scale_BatchNorm", ((4096,),)),
        ("40_Matmul_Scaling_ResidualAdd", (-1.5,)),
        ("41_Gemm_BatchNorm_GELU_ReLU", ()),
    ],
)
def test_models_use_region_order_and_match_sfc(problem, extra_args):
    root = Path(__file__).resolve().parents[1]
    model_module = _load_module(
        root / "backends/cutile/cpu/KernelBench/level2" / f"{problem}.py",
        f"regions_{problem}",
    )
    torch.manual_seed(0)
    model = model_module.Model(4096, 4096, *extra_args).to(torch.bfloat16).eval()
    if problem.startswith("39"):
        with torch.no_grad():
            model.scale.copy_(torch.linspace(-1, 1, 4096, dtype=torch.bfloat16))
            model.scale[::3] = 0
    inputs = torch.randn(16384, 4096, dtype=torch.bfloat16)
    maps = []
    options = {"num_cpu_threads": 16}

    def recording(*args):
        maps.append(_make_region_tensor(*args))
        return maps[-1]

    with (
        torch.no_grad(),
        patch.object(
            model_module,
            "region_matmul",
            functools.partial(model_module.region_matmul, options=options),
        ),
        patch("cutile_cpu_utils.region_matmul._make_region_tensor", recording),
    ):
        actual = model(inputs).clone()
    assert maps and all(tiles is not None for tiles in maps)
    with torch.no_grad():
        expected = sfc_matmul(
            inputs,
            model._weight_packed,
            bias=model._bias,
            post_op=model._epilogue_fun
            if problem.startswith("40")
            else model_module._epilogue,
            post_op_arg=model._scale if problem.startswith("39") else None,
            b_is_prepacked=True,
            options=options,
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_live_output_view_keeps_pooled_storage():
    a = torch.empty((1024, 64), dtype=torch.bfloat16)
    b = torch.empty((64, 2048), dtype=torch.bfloat16)
    prepared = prepare_sfc_matmul(a, b, b_is_prepacked=True)
    prepared._output.fill_(7)
    address = prepared._output.data_ptr()
    view = prepared._output.view(-1)[1:]
    del prepared
    following = prepare_sfc_matmul(a, b, b_is_prepacked=True)
    assert following._output.data_ptr() != address
    following._output.zero_()
    assert torch.all(view == 7)


@kernel_test
def test_large_pooled_matmul_preserves_live_output():
    a, weights, packed = _inputs(m=1024, n=2048, k=64)
    common = {
        "b_is_prepacked": True,
        "blocking_factor_k": 2,
        "progs_per_thread": 1,
        "options": {"num_cpu_threads": 4},
    }
    prepared = prepare_sfc_matmul(a, packed, **common)
    prepared._launches[0][2][3].fill_(float("nan"))
    output = prepared()
    del prepared
    expected = (a.float() @ weights.float().T).to(torch.bfloat16)
    torch.testing.assert_close(output, expected)
    snapshot = output.clone()
    following = sfc_matmul(-a, packed, **common)
    torch.testing.assert_close(following, -expected)
    torch.testing.assert_close(output, snapshot, rtol=0, atol=0)
    assert output.data_ptr() != following.data_ptr()


def _inputs(m=64, n=64, k=512):
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.bfloat16)
    weights = torch.randn(n, k, dtype=torch.bfloat16)
    packed = pack_weights_for_sfc_matmul(weights, 32, 32)
    return a, weights, packed


@kernel_test
def test_k_blocking_factors_are_bitwise_invariant():
    a, _, packed = _inputs(k=4096)
    outputs = [
        sfc_matmul(
            a,
            packed,
            b_is_prepacked=True,
            blocking_factor_k=factor,
            progs_per_thread=1,
            options={"num_cpu_threads": 4},
        )
        for factor in (1, 2, 4, 8)
    ]
    for output in outputs[1:]:
        torch.testing.assert_close(output, outputs[0], rtol=0, atol=0)


@kernel_test
def test_autotuned_path_matches_reference():
    a, weights, packed = _inputs()
    output = sfc_matmul(a, packed, b_is_prepacked=True, options={"num_cpu_threads": 4})
    reference = (a.float() @ weights.float().T).to(torch.bfloat16)
    torch.testing.assert_close(output, reference)


@kernel_test
@pytest.mark.parametrize("b_is_prepacked", [False, True])
def test_autotuning_candidates_share_buffers(monkeypatch, b_is_prepacked):
    module = importlib.import_module("cutile_cpu_utils.sfc_matmul")
    prepare = module.prepare_sfc_matmul
    allocate = module._make_buffers
    buffers = []

    def recording_prepare(*args, **kwargs):
        prepared = prepare(*args, **kwargs)
        buffers.append(kwargs["_buffers"])
        return prepared

    monkeypatch.setattr(module, "prepare_sfc_matmul", recording_prepare)
    a, weights, packed = _inputs(m=96, n=96, k=2048)
    with patch.object(module, "_make_buffers", wraps=allocate) as allocation:
        output = sfc_matmul(
            a,
            packed if b_is_prepacked else weights.T.contiguous(),
            b_is_prepacked=b_is_prepacked,
            options={"num_cpu_threads": 4},
        )
    assert allocation.call_count == 1
    assert len(buffers) > 1
    assert all(candidate is buffers[0] for candidate in buffers)
    assert buffers[0][-1] is output
    reference = (a.float() @ weights.float().T).to(torch.bfloat16)
    torch.testing.assert_close(output, reference)


@kernel_test
@pytest.mark.parametrize("dtype", [torch.float32, torch.int8])
def test_non_bf16_dtypes(dtype):
    torch.manual_seed(0)
    if dtype == torch.int8:
        a = torch.randint(-3, 4, (32, 64), dtype=dtype)
        weights = torch.randint(-3, 4, (32, 64), dtype=dtype)
    else:
        a = torch.randn(32, 64, dtype=dtype)
        weights = torch.randn(32, 64, dtype=dtype)
    packed = pack_weights_for_sfc_matmul(weights, 32, 32)
    output = sfc_matmul(
        a,
        packed,
        b_is_prepacked=True,
        trunc_output=dtype != torch.int8,
        blocking_factor_k=1,
        progs_per_thread=1,
        options={"num_cpu_threads": 4},
    )
    reference = a.int() @ weights.int().T if dtype == torch.int8 else a @ weights.T
    torch.testing.assert_close(output, reference, rtol=0, atol=0)


@kernel_test
def test_full_tile_callback_semantics_survive_persistence():
    a = torch.arange(64, dtype=torch.float32).reshape(64, 1).repeat(1, 512)
    a = a.to(torch.bfloat16)
    packed = pack_weights_for_sfc_matmul(torch.eye(512, dtype=torch.bfloat16), 32, 32)
    output = sfc_matmul(
        a,
        packed,
        b_is_prepacked=True,
        post_op=_subtract_column_sum,
        blocking_factor_k=2,
        progs_per_thread=1,
        options={"num_cpu_threads": 4},
    )
    expected = a.clone()
    for row in range(0, 64, 32):
        expected[row : row + 32] -= expected[row : row + 32].sum(dim=0, keepdim=True)
    torch.testing.assert_close(output, expected)


@kernel_test
def test_reduction_and_softmax_finish_kernels():
    a, weights, packed = _inputs()
    reduction = sfc_matmul(
        a,
        packed,
        b_is_prepacked=True,
        reduce_last_dim=True,
        trunc_output=False,
        blocking_factor_k=2,
        progs_per_thread=1,
        options={"num_cpu_threads": 4},
    )
    logits = a.float() @ weights.float().T
    torch.testing.assert_close(reduction, logits.sum(dim=1), rtol=0.005, atol=0.01)

    softmax = sfc_matmul(
        a,
        packed,
        b_is_prepacked=True,
        softmax_last_dim=True,
        blocking_factor_k=2,
        progs_per_thread=1,
        options={"num_cpu_threads": 4},
    )
    torch.testing.assert_close(
        softmax.float(), torch.softmax(logits, dim=1), rtol=0.02, atol=0.002
    )


@kernel_test
def test_nonprepacked_b_matches_prepacked_b():
    a, weights, packed = _inputs()
    expected = sfc_matmul(
        a,
        packed,
        b_is_prepacked=True,
        blocking_factor_k=2,
        progs_per_thread=1,
        options={"num_cpu_threads": 4},
    )
    actual = sfc_matmul(
        a,
        weights.T.contiguous(),
        blocking_factor_k=2,
        progs_per_thread=1,
        options={"num_cpu_threads": 4},
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@kernel_test
@pytest.mark.parametrize(
    ("stem", "model_args", "atol"),
    [
        ("14_Gemm_Divide_Sum_Scaling", (512, 64, 1.5), 0.5),
        ("70_Gemm_Sigmoid_Scaling_ResidualAdd", (512, 64, 2.0), 0.04),
        ("95_Matmul_Add_Swish_Tanh_GELU_Hardtanh", (512, 64, (64,)), 0.02),
    ],
)
def test_requested_fused_models_match_reference(stem, model_args, atol):
    root = Path(__file__).resolve().parents[1]
    reference_module = _load_module(
        root / "third_party/KernelBench/KernelBench/level2" / f"{stem}.py",
        f"reference_{stem}",
    )
    cutile_module = _load_module(
        root / "backends/cutile/cpu/KernelBench/level2" / f"{stem}.py",
        f"cutile_{stem}",
    )
    torch.manual_seed(0)
    reference = reference_module.Model(*model_args).to(torch.bfloat16).eval()
    actual_model = cutile_module.Model(*model_args).eval()
    assert copy_model_weights(reference, actual_model)
    inputs = torch.randn(64, 512, dtype=torch.bfloat16)
    expected = reference(inputs)
    actual = actual_model(inputs)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=atol)


@kernel_test
def test_output_does_not_overwrite_next_a():
    a = torch.randn(64, 64, dtype=torch.bfloat16)
    identity = pack_weights_for_sfc_matmul(torch.eye(64, dtype=torch.bfloat16), 32, 32)
    common = {
        "b_is_prepacked": True,
        "options": {"num_cpu_threads": 4},
    }
    first = sfc_matmul(a, identity, blocking_factor_k=1, progs_per_thread=1, **common)
    with _force_second_autotune_candidate():
        second = sfc_matmul(first, identity, **common)
    torch.testing.assert_close(second, a, rtol=0, atol=0)
    torch.testing.assert_close(first, a, rtol=0, atol=0)
    assert first.data_ptr() != second.data_ptr()


@kernel_test
def test_output_does_not_overwrite_b():
    a, _, packed = _inputs(m=512, n=64, k=512)
    common = {
        "b_is_prepacked": True,
        "options": {"num_cpu_threads": 4},
    }
    holder = prepare_sfc_matmul(
        a,
        packed,
        blocking_factor_k=1,
        progs_per_thread=1,
        **common,
    )
    aliased_b = holder._output
    aliased_b.copy_(packed)
    expected = sfc_matmul(
        a,
        aliased_b.clone(),
        blocking_factor_k=1,
        progs_per_thread=1,
        **common,
    )
    with _force_second_autotune_candidate():
        actual = sfc_matmul(a, aliased_b, **common)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(aliased_b, packed, rtol=0, atol=0)


@kernel_test
@pytest.mark.parametrize(
    ("stem", "model_args", "atol"),
    [
        ("30_Gemm_GroupNorm_Hardtanh", (512, 256, 8, -1.0, 1.0), 0.02),
        ("37_Matmul_Swish_Sum_GroupNorm", (512, 256, 8, (256,)), 0.04),
        ("45_Gemm_Sigmoid_LogSumExp", (512, 256, 256), 0.04),
        ("51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd", (512, 256), 0.04),
        ("62_Matmul_GroupNorm_LeakyReLU_Sum", (512, 256, 8), 0.5),
        ("75_Gemm_GroupNorm_Min_BiasAdd", (512, 256, 8, (2, 1, 1)), 0.04),
        ("80_Gemm_Max_Subtract_GELU", (512, 256, 0), 0.04),
        ("80_Gemm_Max_Subtract_GELU", (512, 256, 1), 0.04),
        ("88_Gemm_GroupNorm_Swish_Multiply_Swish", (512, 256, 8, (256,)), 0.04),
        ("94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm", (512, 256, (256,), 8), 0.04),
    ],
)
def test_pooled_models_match_reference(stem, model_args, atol):
    root = Path(__file__).resolve().parents[1]
    reference_module = _load_module(
        root / "third_party/KernelBench/KernelBench/level2" / f"{stem}.py",
        f"reference_{stem}",
    )
    cutile_module = _load_module(
        root / "backends/cutile/cpu/KernelBench/level2" / f"{stem}.py",
        f"cutile_{stem}",
    )
    torch.manual_seed(0)
    reference = reference_module.Model(*model_args).to(torch.bfloat16).eval()
    model = cutile_module.Model(*model_args).eval()
    assert copy_model_weights(reference, model)
    inputs = torch.randn(256, 512, dtype=torch.bfloat16)
    with (
        patch("shared_cpu_utils.output_pool.MIN_BYTES", 1),
        patch("shared_cpu_utils.output_pool._max_bytes", 4 * 1024 * 1024),
        torch.no_grad(),
    ):
        expected = reference(inputs)
        actual = model(inputs)
        # Pool storage comes from torch.frombuffer and cannot be resized.
        assert not actual.untyped_storage().resizable()
        snapshot = actual.clone()
        following = model(-inputs)
        assert not following.untyped_storage().resizable()
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=atol)
    torch.testing.assert_close(actual, snapshot, rtol=0, atol=0)
