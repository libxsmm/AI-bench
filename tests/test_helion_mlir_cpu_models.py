"""Seed-only fidelity of the helion-mlir CPU KernelBench models.

The harness copies the reference weights into a backend model before it
checks correctness, but tools that compare backends in separate processes
(e.g. eager vs helion-mlir timing runs) rely on ``set_all_seeds`` alone. A
backend model must therefore draw its parameters exactly like the
KernelBench reference: same distributions, in the same order, and it must
leave the RNG where the reference does (inputs are drawn after the model).

The model checks that compile kernels run only with
``AI_BENCH_HELION_MLIR_KERNEL_TESTS=1``.
"""

import os
import sys

import pytest
import torch

from ai_bench.harness import core as ai_hc
from ai_bench.harness.runner import KernelBenchRunner
from ai_bench.harness.runner.benchmark_compare import copy_model_weights
from ai_bench.harness.runner.benchmark_compare import set_all_seeds
from ai_bench.utils import import_kernel_from_path

pytest.importorskip("helion_mlir_backend")

CPU = torch.device("cpu")

# Model checks that compile kernels run only on request.
KERNEL_TESTS_ENV = "AI_BENCH_HELION_MLIR_KERNEL_TESTS"
kernel_test = pytest.mark.skipif(
    os.environ.get(KERNEL_TESTS_ENV) != "1",
    reason=f"compiles Helion-MLIR kernels; set {KERNEL_TESTS_ENV}=1 to run",
)


def _runners():
    ref = KernelBenchRunner(device=CPU, backend=ai_hc.Backend.PYTORCH)
    mlir = KernelBenchRunner(device=CPU, backend=ai_hc.Backend.HELION_MLIR)
    return ref, mlir


def _cases():
    _, mlir = _runners()
    for level in ("level1", "level2", "level3"):
        for kernel in sorted((mlir.kernels / level).glob("*.py")):
            if (mlir.specs / level / f"{kernel.stem}.yaml").exists():
                yield f"{level}/{kernel.stem}"


def _build(key, spec_type):
    """Seed-only reference and helion-mlir models plus inputs for ``key``.

    Also returns the RNG states right after building each model."""
    ref_runner, mlir_runner = _runners()
    level, name = key.split("/")
    spec = ref_runner.load_spec(ref_runner.specs / level / f"{name}.yaml")
    if spec_type not in spec:
        pytest.skip(f"no {spec_type} variant")
    variant = ai_hc.expand_variants(spec[spec_type])[0]
    inits = ref_runner.get_spec_inits(spec)
    set_all_seeds(123)
    ref = ref_runner.init_model(
        ref_runner.load_model(ref_runner.kernels / level / f"{name}.py"),
        variant,
        inits,
    )
    ref_rng = torch.get_rng_state()
    args = ai_hc.get_inputs(variant, ref_runner.get_spec_inputs(spec), device=CPU)
    set_all_seeds(123)
    model = mlir_runner.init_model(
        mlir_runner.load_model(mlir_runner.kernels / level / f"{name}.py"),
        variant,
        inits,
    )
    model_rng = torch.get_rng_state()
    return ref, model, args, ref_rng, model_rng


@pytest.mark.parametrize("key", list(_cases()))
def test_seed_only_parameters_match_reference(key):
    """Every parameter, in order, equals the reference's under one seed, and
    building the model leaves the RNG in the reference's state."""
    ref, model, _, ref_rng, model_rng = _build(key, "ci")
    assert torch.equal(model_rng, ref_rng)
    ref_params = [p for _, p in ref.named_parameters()]
    params = [p for _, p in model.named_parameters()]
    assert len(params) == len(ref_params)
    for index, (p, r) in enumerate(zip(params, ref_params)):
        assert p.shape == r.shape, index
        assert torch.equal(p, r), index


@pytest.mark.parametrize("key", list(_cases()))
def test_harness_weight_copy_succeeds(key):
    """``copy_model_weights`` takes the reference state; derived constants
    kept as buffers must be non-persistent (97's ``_bn_scale`` made the copy
    fail, so the harness fell back to seed-only weights)."""
    ref, model, *_ = _build(key, "ci")
    assert copy_model_weights(ref, model)


@pytest.mark.parametrize(
    "key", ["level2/33_Gemm_Scale_BatchNorm", "level2/39_Gemm_Scale_BatchNorm"]
)
@kernel_test
def test_gemm_scale_batchnorm_seed_only_output(key):
    """The bf16 mlir-cpu-bench case (128x4096x4096) without a weight copy:
    the models drew ``scale`` from torch.rand where the reference uses
    torch.randn, so the outputs differed by 1.3 x max|ref|."""
    ref, model, args, *_ = _build(key, "mlir-cpu-bench")
    with torch.no_grad():
        expected = ref(*args).float()
        out = model(*args).float()
    error = ((out - expected).abs().max() / expected.abs().max()).item()
    assert error < 2e-2


def _pair(key, inits, dtype):
    """Reference and helion-mlir CPU models of ``key`` built from ``inits``, in
    eval mode and ``dtype``, with the reference weights copied in. Built right
    after loading each (see the module docstring)."""
    ref_runner, mlir_runner = _runners()
    level, name = key.split("/")
    set_all_seeds(123)
    ref = ref_runner.load_model(ref_runner.kernels / level / f"{name}.py")(*inits)
    ref = ref.to(dtype).eval()
    model = mlir_runner.load_model(mlir_runner.kernels / level / f"{name}.py")(*inits)
    model = model.to(dtype).eval()
    assert copy_model_weights(ref, model)
    return ref, model


def _error(out, ref):
    out, ref = out.double(), ref.double()
    return ((out - ref).abs().max() / ref.abs().max()).item()


@pytest.mark.parametrize(
    "key",
    [
        "level2/66_Matmul_Dropout_Softmax",
        "level2/84_Gemm_BatchNorm_Scaling_Softmax",
        "level2/99_Matmul_GELU_Softmax",
    ],
)
@kernel_test
def test_softmax_over_partial_last_chunk(key):
    """130x1001x515 f32, where 66 first failed: the softmax over 515 columns
    runs in 128-column chunks, and the zero padding of the partial last one
    entered the row maxima and sums (0.18 x max|ref| off)."""
    inits = (1001, 515, 0.2) if key.startswith("level2/66_") else (1001, 515)
    ref, model = _pair(key, inits, torch.float32)
    torch.manual_seed(0)
    x = torch.rand(130, 1001)
    with torch.no_grad():
        assert _error(model(x), ref(x)) < 1e-4


@pytest.mark.parametrize(
    "key",
    [
        "level2/66_Matmul_Dropout_Softmax",
        "level2/84_Gemm_BatchNorm_Scaling_Softmax",
        "level2/99_Matmul_GELU_Softmax",
    ],
)
@kernel_test
def test_softmax_over_one_column(key):
    """OUT_FEATURES = 1: the chunked softmax kernels did not compile
    (ControlFlowTensorMismatch); a softmax over one column is 1."""
    inits = (32, 1, 0.2) if key.startswith("level2/66_") else (32, 1)
    ref, model = _pair(key, inits, torch.float32)
    x = torch.rand(5, 32)
    with torch.no_grad():
        assert torch.equal(model(x), ref(x))


def test_matmul_dropout_softmax_needs_eval_mode():
    """AI-bench runs models in eval mode, where dropout is the identity;
    random training masks cannot match the reference, so they are refused."""
    _, model = _pair("level2/66_Matmul_Dropout_Softmax", (32, 32, 0.2), torch.float32)
    with pytest.raises(ValueError, match="eval mode"):
        model.train()(torch.rand(2, 32))


@pytest.mark.parametrize(
    ("m", "k", "n", "pool", "scale", "dtype"),
    [
        (130, 1001, 515, 16, 2.0, torch.float32),  # partial last window dropped
        (130, 1001, 515, 16, 2.0, torch.bfloat16),  # odd K: blocked bf16 GEMM
        (100, 512, 301, 3, -1.5, torch.float32),  # window of 3, negative scale
        (64, 512, 512, 16, 2.0, torch.bfloat16),  # AMX VNNI path
    ],
)
@kernel_test
def test_matmul_avgpool_pooled_weights(m, k, n, pool, scale, dtype):
    """98 folds the average pool into the weights (window means of W's rows
    and of b, W' rounded once to the input dtype): checked against the
    reference and against its fp64 result."""
    ref, model = _pair(
        "level2/98_Matmul_AvgPool_GELU_Scale_Max", (k, n, pool, scale), dtype
    )
    torch.manual_seed(0)
    x = torch.rand(m, k).to(dtype)
    with torch.no_grad():
        out = model(x)
        expected = ref(x)
        exact = ref.double()(x.double())
    if dtype is torch.float32:
        assert _error(out, expected) < 1e-5
    else:
        # bf16: no further from the fp64 result than the reference is.
        assert _error(out, exact) <= max(_error(expected, exact), 4e-3)


def _load_case(key, spec_type, dtype):
    _, runner = _runners()
    level, name = key.split("/")
    spec = runner.load_spec(runner.specs / level / f"{name}.yaml")
    variant = next(
        v for v in ai_hc.expand_variants(spec[spec_type]) if v["dtype"] == dtype
    )
    set_all_seeds(123)
    model = runner.init_model(
        runner.load_model(runner.kernels / level / f"{name}.py"),
        variant,
        runner.get_spec_inits(spec),
    )
    return model, ai_hc.get_inputs(variant, runner.get_spec_inputs(spec), device=CPU)


# A kernel reading a module global, directly (84) or in an epilogue (12): it
# compiles on its first call, after the other model was loaded. Same shapes as
# the run that found this: bench-cpu, bf16, 84 first.
@pytest.mark.parametrize(
    "key,spec_type,dtype",
    [
        pytest.param(
            "level2/84_Gemm_BatchNorm_Scaling_Softmax",
            "bench-cpu",
            "bfloat16",
            marks=kernel_test,
        ),
        ("level2/12_Gemm_Multiply_LeakyReLU", "ci", "float32"),
    ],
)
def test_kernel_compiles_after_another_model_was_loaded(key, spec_type, dtype):
    other = "level1/7_Matmul_with_small_K_dimension_"
    alone, inputs = _load_case(key, spec_type, dtype)
    with torch.no_grad():
        expected = alone(*inputs)
    assert expected.count_nonzero() > 0
    model, inputs = _load_case(key, spec_type, dtype)
    later, later_inputs = _load_case(other, "bench-cpu", "bfloat16")
    with torch.no_grad():
        assert torch.equal(model(*inputs), expected)
        later(*later_inputs)


def test_each_kernel_load_gets_its_own_module(tmp_path):
    for parent, value in (("a", 1), ("b", 2)):
        (tmp_path / parent).mkdir()
        (tmp_path / parent / "1_kernel-x.py").write_text(f"VALUE = {value}\n")
    loaded = [
        import_kernel_from_path(tmp_path / parent / "1_kernel-x.py")
        for parent in ("a", "b", "a")
    ]
    try:
        assert [m.VALUE for m in loaded] == [1, 2, 1]
        assert len({m.__name__ for m in loaded}) == 3
        for module in loaded:
            assert module.__name__.isidentifier()
            assert sys.modules[module.__name__] is module
    finally:
        for module in loaded:
            del sys.modules[module.__name__]


@kernel_test
def test_matmul_avgpool_pooled_weights_follow_updates():
    """The pooled weights are rebuilt when the parameters change, in place or
    through ``load_state_dict``."""
    key = "level2/98_Matmul_AvgPool_GELU_Scale_Max"
    ref, model = _pair(key, (64, 64, 8, 2.0), torch.float32)
    x = torch.rand(4, 64)
    with torch.no_grad():
        model(x)
        model.matmul.weight.mul_(-0.5)
        ref.matmul.weight.mul_(-0.5)
        assert _error(model(x), ref(x)) < 1e-5
        ref.matmul.bias.add_(0.25)
        model.load_state_dict(ref.state_dict())
        assert _error(model(x), ref(x)) < 1e-5


def _sdpa_module():
    _, mlir = _runners()
    path = mlir.kernels / "level1" / "97_ScaledDotProductAttention.py"
    return import_kernel_from_path(path)


def test_sdpa_cheap_exp_accuracy():
    x = torch.linspace(-40.0, 8.0, 200001, dtype=torch.float32)
    out = _sdpa_module()._exp_bf16(x).double()
    ref = x.double().exp()
    near = x >= -8.0
    assert ((out - ref) / ref)[near].abs().max() < 8e-4
    assert (out - ref)[~near].abs().max() < 3e-7
    assert out.min() > 0.0


@pytest.mark.parametrize(
    "shape",
    [
        (1, 4, 64, 64),
        (1, 4, 129, 65),
        (1, 2, 33, 17),
        (1, 4, 1024, 128),
        (1, 2, 65, 7),
        (1, 2, 64, 100),
        (1, 2, 129, 161),
    ],
)
@pytest.mark.parametrize("gain", [1.0e-2, 1.0, 16.0])
@kernel_test
def test_sdpa_bf16_matches_fp64(shape, gain):
    """The cheap exponential at bf16: as close to fp64 as torch's bf16 SDPA,
    from nearly uniform weights to rows of few large ones; ragged shapes."""
    model = _sdpa_module().Model()
    torch.manual_seed(11)
    q, k, v = (torch.rand(shape) for _ in range(3))
    q, k, v = (q * gain).bfloat16(), (k * gain).bfloat16(), v.bfloat16()
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.double(), k.double(), v.double()
    )
    torch_error = _error(torch.nn.functional.scaled_dot_product_attention(q, k, v), ref)
    assert _error(model(q, k, v), ref) < max(2.0 * torch_error, 6e-3)


# Below 32 keys, one column and odd shapes stay off AMX or masked: right values.
@pytest.mark.parametrize(
    "shape",
    [(1, 2, 31, 64), (1, 2, 2, 1), (1, 2, 16, 33), (1, 2, 33, 7), (1, 2, 48, 7)],
)
@kernel_test
def test_sdpa_bf16_small_shapes_match_fp64(shape):
    model = _sdpa_module().Model()
    torch.manual_seed(3)
    q, k, v = (torch.randn(shape).bfloat16() for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.double(), k.double(), v.double()
    )
    assert _error(model(q, k, v), ref) < 8e-3


# D not a multiple of 32 (e.g. (4, 32, 2048, 66)): partial tiles of D left every
# contraction off AMX, in masked per-element code. Results were right, so the operands' padding is what's
# checked here, AMX below, then the results at the same shapes.
@pytest.mark.parametrize(
    ("shape", "padded"), [((4, 32, 2048, 66), 96), ((32, 32, 512, 100), 128)]
)
@kernel_test
def test_sdpa_bf16_unaligned_head_dim_is_padded_for_amx(shape, padded):
    from helion_mlir_backend import generate_mlir

    module = _sdpa_module()
    torch.manual_seed(5)
    q, k, v = (torch.randn(shape).bfloat16() for _ in range(3))
    kernel, args = module._call(q, k, v)
    ir = str(generate_mlir(kernel, list(args)))
    pads = [line for line in ir.splitlines() if "} : tensor<" in line]
    assert len(pads) == 3 and all(f"x{padded}xbf16>" in line for line in pads)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.double(), k.double(), v.double()
    )
    torch_error = _error(torch.nn.functional.scaled_dot_product_attention(q, k, v), ref)
    assert _error(kernel(*args).view(shape), ref) < max(2.0 * torch_error, 6e-3)


def _amx_contractions(kernel, args):
    """The ``x86.amx.tile_mulf`` and leftover ``vector.contract`` ops after the
    x86 vectorization stage of ``kernel``'s CPU pipeline, AMX forced on."""
    from helion_mlir_backend import generate_mlir
    from helion_mlir_backend._compiler.execution import inline_module
    from helion_mlir_backend._compiler.targets.cpu import pipeline_descriptor
    from lighthouse.execution.target import TargetInfo
    from lighthouse.pipeline.driver import BackendDriver
    from mlir import ir

    module = generate_mlir(kernel, list(args))
    host = [f for f in TargetInfo.host().features if not f.startswith("amx")]
    features = [*host, "amx_tile", "amx_int8", "amx_bf16"]
    with (
        TargetInfo.override(features=features),
        module.context,
        ir.Location.unknown(),
    ):
        inline_module(module)
        driver = BackendDriver(
            module, kernel.fn.__name__, result_to_args=False, benchmark=False
        )
        driver.add_stage(pipeline_descriptor("opt"))
        for stage in driver.stages:
            module = stage.apply(module)
            text = str(module)
            if "x86.amx.tile_mulf" in text:
                return text.count("x86.amx.tile_mulf"), text.count("vector.contract")
    return 0, str(module).count("vector.contract")


# The AMX gate of the shapes that ran masked: every contraction reaches AMX,
# with no masked key tile (D padded, S dividing its tiles) or some for the tail.
@pytest.mark.parametrize(
    ("shape", "all_on_amx"),
    [
        ((4, 32, 2048, 66), True),
        ((32, 32, 512, 100), True),
        ((2, 2, 96, 64), True),
        ((32, 32, 100, 64), False),
        ((2, 2, 100, 64), False),
        ((1, 2, 129, 161), False),
    ],
)
@kernel_test
def test_sdpa_bf16_contractions_reach_amx(shape, all_on_amx):
    q, k, v = (torch.zeros(shape, dtype=torch.bfloat16) for _ in range(3))
    kernel, args = _sdpa_module()._call(q, k, v)
    dots, left = _amx_contractions(kernel, args)
    assert dots > 0
    assert left == 0 or not all_on_amx


# Scaled scores all below about -88 with a partial key tile: a padded lane's
# score of 0 gave exp(0 - top) = inf, and inf * 0 in the V dot made every such row
# NaN, on finite inputs, with torch and fp64 finite. Found on S 65, 600, 3; f32 too.
@pytest.mark.parametrize(
    ("shape", "dtype", "limit"),
    [
        ((1, 2, 65, 64), torch.bfloat16, 6e-3),
        ((1, 2, 3, 64), torch.bfloat16, 6e-3),
        ((1, 2, 33, 100), torch.bfloat16, 6e-3),
        ((1, 2, 600, 66), torch.bfloat16, 6e-3),
        ((1, 2, 129, 64), torch.float32, 1e-4),
    ],
)
@kernel_test
def test_sdpa_all_negative_scores_with_a_partial_key_tile(shape, dtype, limit):
    torch.manual_seed(0)
    q = (torch.randn(shape).abs() * 8).to(dtype)
    k = (-torch.randn(shape).abs() * 8).to(dtype)
    v = torch.randn(shape).to(dtype)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.double(), k.double(), v.double()
    )
    out = _sdpa_module().Model()(q, k, v)
    assert not out.double().isnan().any()
    assert _error(out, ref) < limit


# The padded columns of D are computed, not stored: NaNs and -inf in the last
# real column act as in fp64.
@kernel_test
def test_sdpa_bf16_unaligned_head_dim_nonfinite():
    model = _sdpa_module().Model()
    q, k, v = _sdpa_nonfinite_inputs((1, 2, 64, 100))
    q[0, 0, 3, 99] = float("nan")
    out = model(q, k, v)
    assert _nan_rows(out) == 1
    assert out[0, 0, 3].double().isnan().any()
    q, k, v = _sdpa_nonfinite_inputs((1, 2, 64, 100))
    q[0, 0, 3, 99] = float("-inf")
    out = model(q, k, v)
    assert _nan_rows(out) == 0
    assert out[0, 0, 3].double().abs().max() == 0


@pytest.mark.parametrize("operand", ["q", "k", "v"])
@kernel_test
def test_sdpa_bf16_propagates_nan(operand):
    model = _sdpa_module().Model()
    torch.manual_seed(1)
    tensors = {n: torch.rand(1, 2, 64, 64).bfloat16() for n in "qkv"}
    tensors[operand][0, 0, 3, 5] = float("nan")
    out = model(tensors["q"], tensors["k"], tensors["v"])
    assert out[0, 0, 3].isnan().any()


def _sdpa_nonfinite_inputs(shape):
    torch.manual_seed(1)
    return [torch.rand(shape).bfloat16() for _ in range(3)]


def _nan_rows(out):
    return int(out.double().isnan().any(-1).sum())


# One NaN in Q is one NaN row, as in fp64: found at both block sizes (a NaN
# row closed the rescale test for all rows of its block, 64 and 512 rows).
@pytest.mark.parametrize("shape", [(1, 2, 64, 64), (1, 1, 1024, 128)])
@kernel_test
def test_sdpa_bf16_one_nan_query_is_one_nan_row(shape):
    model = _sdpa_module().Model()
    q, k, v = _sdpa_nonfinite_inputs(shape)
    q[0, 0, 3, 5] = float("nan")
    out = model(q, k, v)
    assert _nan_rows(out) == 1
    assert out[0, 0, 3].double().isnan().any()


# A row of all -inf scores has no weights: 0 as in fp64, not the plain mean of
# V, finite and wrong; with a partial key tile (S not dividing its tiles), not NaN.
@pytest.mark.parametrize("s", [64, 33, 40, 48, 66, 100, 120, 5, 65, 129])
@kernel_test
def test_sdpa_bf16_minus_inf_row_is_not_the_mean_of_v(s):
    model = _sdpa_module().Model()
    q, k, v = _sdpa_nonfinite_inputs((1, 2, s, 64))
    q[0, 0, 3, 5] = float("-inf")
    out = model(q, k, v)
    assert _nan_rows(out) == 0
    assert out[0, 0, 3].double().abs().max() == 0


@kernel_test
def test_sdpa_bf16_minus_inf_key_weighs_nothing():
    model = _sdpa_module().Model()
    q, k, v = _sdpa_nonfinite_inputs((1, 2, 64, 64))
    k[0, 0, 7, 5] = float("-inf")
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.double(), k.double(), v.double()
    )
    out = model(q, k, v)
    assert _nan_rows(out) == 0
    assert _error(out, ref) < 6e-3


@kernel_test
def test_sdpa_bf16_plus_inf_row_leaves_other_rows():
    model = _sdpa_module().Model()
    q, k, v = _sdpa_nonfinite_inputs((1, 1, 64, 64))
    clean = model(q, k, v).double()
    q[0, 0, 3, 5] = float("inf")
    out = model(q, k, v).double()
    others = [r for r in range(64) if r != 3]
    assert not out[0, 0, others].isnan().any()
    assert _error(out[0, 0, others], clean[0, 0, others]) < 6e-3


@kernel_test
def test_sdpa_bf16_scores_below_any_finite_start():
    model = _sdpa_module().Model()
    q, k, v = _sdpa_nonfinite_inputs((1, 1, 64, 64))
    q, k = (q * -1.0e17).bfloat16(), (k * 1.0e17).bfloat16()
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.double(), k.double(), v.double()
    )
    assert ref.abs().max() > 0
    assert _error(model(q, k, v), ref) < 6e-3


# +inf in row 0 opens the rescale test once, then inf - inf is NaN there; a NaN
# row must not close it for row 1, whose scores keep growing past the threshold:
# with the test closed, row 1 overflows (128 NaNs instead of fp64's 64).
@kernel_test
def test_sdpa_bf16_nan_row_keeps_rescale_open_for_growing_row():
    s, d, block = 512, 64, 128
    logits = torch.tensor(
        [
            [
                1.0
                if r == 0
                else 120.0 - ((j * 5) % 17) / 2
                if r == 1 and j >= block
                else ((j * 3) % 5) / 5
                for j in range(s)
            ]
            for r in range(4)
        ],
        dtype=torch.float32,
    )
    q, k = torch.zeros(1, 1, s, d), torch.zeros(1, 1, s, d)
    for r in range(4):
        q[0, 0, r, r] = 1.0
        k[0, 0, :, r] = d**0.5 * logits[r]
    torch.manual_seed(5)
    v = torch.rand(1, 1, s, d).bfloat16()
    q, k = q.bfloat16(), k.bfloat16()
    q[0, 0, 0, 0] = float("inf")
    out = _sdpa_module().Model()(q, k, v).double()
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.double(), k.double(), v.double()
    )
    assert out.isnan().equal(ref.isnan())
    assert int(out.isnan().sum()) == 64
    finite = ~ref.isnan()
    assert (out[finite] - ref[finite]).abs().max() < 2e-2 * ref[finite].abs().max()


_TENSOR_MATMUL = "level1/11_4D_tensor_matrix_multiplication"


@pytest.mark.parametrize(
    ("b", "i", "j", "inner", "outer"),
    [
        (2, 128, 64, 64, 96),  # whole-row tiles on 8 threads
        (2, 129, 65, 33, 25),  # the default tiles: ragged everywhere
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@kernel_test
def test_tensor_matmul_tiles(b, i, j, inner, outer, dtype, monkeypatch):
    """11's tiles change no value: bit for bit the default tiles' result, and
    close to the fp64 result."""
    from helion_mlir_cpu_utils import matmul

    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    ref, model = _pair(_TENSOR_MATMUL, (), dtype)
    torch.manual_seed(0)
    x = torch.rand(b, i, j, inner).to(dtype)
    y = torch.rand(inner, outer).to(dtype)
    with torch.no_grad():
        out = model(x, y)
        default = matmul(x.reshape(b * i * j, inner), y).view(out.shape)
        exact = ref.double()(x.double(), y.double())
    assert torch.equal(out, default)
    assert _error(out, exact) < (2.0**-7 if dtype is torch.bfloat16 else 1e-5)


_GEMM_SUM = "level2/14_Gemm_Divide_Sum_Scaling"
_GEMM_SUM_ENV = "HELION_MLIR_L2_14_REWRITE"


@pytest.fixture
def gemm_sum_rewrite(monkeypatch):
    """14's models built in the test use the opt-in rewrite."""
    monkeypatch.setenv(_GEMM_SUM_ENV, "1")


@pytest.mark.parametrize(
    ("value", "rewrite"),
    [
        (None, False),
        ("0", False),
        ("false", False),
        ("", False),
        ("1", True),
        ("true", True),
    ],
)
@kernel_test
def test_gemm_sum_flag_picks_the_kernels(value, rewrite, monkeypatch):
    """14 runs the GEMM and the row sum unless the flag asks for the rewrite:
    told by the helpers called, one call per form."""
    if value is None:
        monkeypatch.delenv(_GEMM_SUM_ENV, raising=False)
    else:
        monkeypatch.setenv(_GEMM_SUM_ENV, value)
    ref, model = _pair(_GEMM_SUM, (64, 64, 1.5), torch.float32)
    called = []
    namespace = type(model).forward.__globals__
    for name in ("linear", "row_sum", "column_sum", "row_dot"):

        def record(*args, _name=name, _helper=namespace[name], **kwargs):
            called.append(_name)
            return _helper(*args, **kwargs)

        monkeypatch.setitem(namespace, name, record)
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(torch.rand(4, 64))
    assert called == (["column_sum", "row_dot"] if rewrite else ["linear", "row_sum"])
    assert out.shape == (4, 1)


@pytest.mark.parametrize(
    ("m", "k", "n", "scale"),
    [
        (64, 256, 64, 1.5),  # whole tiles
        (33, 65, 47, -1.5),  # ragged everywhere, negative scale
        (100, 129, 301, 2.0),  # more outputs than inputs
        (5, 31, 3, 1.5),  # below every block
        (1, 64, 1, 1.5),  # one row, one output
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@kernel_test
def test_gemm_sum_forms_agree_with_fp64(m, k, n, scale, dtype, monkeypatch):
    """The default GEMM form and the rewrite: both right against fp64, on the
    first call and a second one on a fresh input."""
    tol = 0.02 if dtype is torch.bfloat16 else 1e-4
    monkeypatch.delenv(_GEMM_SUM_ENV, raising=False)
    ref, default = _pair(_GEMM_SUM, (k, n, scale), dtype)
    monkeypatch.setenv(_GEMM_SUM_ENV, "1")
    _, rewrite = _pair(_GEMM_SUM, (k, n, scale), dtype)
    ref = ref.double()
    torch.manual_seed(0)
    for _ in range(2):
        x = torch.rand(m, k).to(dtype)
        with torch.no_grad():
            out, dot = default(x), rewrite(x)
            exact = ref(x.double())
        assert out.shape == (m, 1) and out.dtype == dtype
        assert _error(out, exact) < tol
        assert _error(dot, exact) < tol


@kernel_test
def test_gemm_sum_default_follows_weight_changes(monkeypatch):
    """The packed weight of 14's default form follows in-place ops and
    ``load_state_dict``, as the other kernels' do through ``linear``."""
    monkeypatch.delenv(_GEMM_SUM_ENV, raising=False)
    ref, model = _pair(_GEMM_SUM, (64, 64, 1.5), torch.float32)
    x = torch.rand(4, 64)
    with torch.no_grad():
        model(x)
        model.weight.mul_(-0.5)
        ref.weight.mul_(-0.5)
        assert _error(model(x), ref(x)) < 1e-4
        ref.weight.add_(0.25)
        model.load_state_dict(ref.state_dict())
        assert _error(model(x), ref(x)) < 1e-4


@pytest.mark.parametrize(
    ("m", "k", "n", "scale"),
    [
        (64, 256, 64, 1.5),  # whole tiles
        (33, 65, 47, -1.5),  # ragged everywhere, negative scale
        (100, 129, 301, 2.0),  # more outputs than inputs
        (5, 31, 3, 1.5),  # below every block
        (1, 64, 1, 1.5),  # one row, one output
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@kernel_test
def test_gemm_sum_as_dot_product(m, k, n, scale, dtype, gemm_sum_rewrite):
    """14's rewrite computes ``x · sum_n(W)``, not the GEMM and the row sum: checked
    against its fp64 result, against the reference and against the GEMM form."""
    ref, model = _pair(_GEMM_SUM, (k, n, scale), dtype)
    torch.manual_seed(0)
    x = torch.rand(m, k).to(dtype)
    with torch.no_grad():
        out = model(x)
        expected = ref(x)
        gemm = model._gemm_sum(x)
        exact = ref.double()(x.double())
    assert out.shape == expected.shape == (m, 1) and out.dtype == dtype
    if dtype is torch.float32:
        assert _error(out, expected) < 1e-5
        assert _error(out, gemm) < 1e-5
    else:
        # bf16: one rounding, so half an ulp of the largest result at most.
        assert _error(out, exact) <= 2.0**-8
        assert _error(out, exact) <= max(_error(expected, exact), _error(gemm, exact))


@kernel_test
def test_gemm_sum_follows_weight_changes(gemm_sum_rewrite):
    """The summed weight is never stale: not after in-place ops or
    ``load_state_dict``, and not after ``.data.copy_()``, which leaves
    ``_version`` unchanged."""
    ref, model = _pair(_GEMM_SUM, (64, 64, 1.5), torch.float32)
    x = torch.rand(4, 64)
    with torch.no_grad():
        model(x)
        model.weight.mul_(-0.5)
        ref.weight.mul_(-0.5)
        assert _error(model(x), ref(x)) < 1e-5
        ref.weight.add_(0.25)
        model.load_state_dict(ref.state_dict())
        assert _error(model(x), ref(x)) < 1e-5
        version = model.weight._version
        new = torch.randn_like(model.weight)
        model.weight.data.copy_(new)
        ref.weight.data.copy_(new)
        assert model.weight._version == version
        assert _error(model(x), ref(x)) < 1e-5


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@kernel_test
def test_gemm_sum_guard_path_follows_weight_changes(dtype, gemm_sum_rewrite):
    """The recomputation of an overflowing result does not reuse a packed weight
    either: its sign follows ``.data``, which leaves ``_version`` unchanged."""
    ref, model = _pair(_GEMM_SUM, (64, 64, 1.5), dtype)
    x = torch.full((8, 64), 1e-5).to(dtype)
    with torch.no_grad():
        for value in (1e37, -1e37, 1e37):
            model.weight.data.fill_(value)
            ref.weight.data.fill_(value)
            out, expected = model(x), ref(x)
            assert torch.isfinite(out).all() and torch.isfinite(expected).all()
            assert _error(out, expected) < 2.0**-6


@kernel_test
def test_gemm_sum_falls_back_for_other_dtypes(gemm_sum_rewrite):
    """A weight of another dtype than the input takes the GEMM and the row sum,
    which round the weight to the input's dtype."""
    ref, model = _pair(_GEMM_SUM, (64, 64, 1.5), torch.float32)
    x = torch.rand(4, 64).to(torch.bfloat16)
    called = []
    gemm_sum = model._gemm_sum
    model._gemm_sum = lambda v: called.append(1) or gemm_sum(v)
    with torch.no_grad():
        out = model(x)
        expected = ref.to(torch.bfloat16)(x)
    assert called and out.dtype == torch.bfloat16
    assert _error(out, expected) < 0.02


@pytest.mark.parametrize(
    ("values", "weights"),
    [
        pytest.param(1e-5, 1e37, id="summed-weight-overflows"),
        pytest.param((1e20, -1e20), 1e17, id="partial-dot-overflows"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@kernel_test
def test_gemm_sum_finite_inputs_stay_finite(values, weights, dtype, gemm_sum_rewrite):
    """The reordered sums overflow where the GEMM does not: 14's rewrite
    recomputes."""
    ref, model = _pair(_GEMM_SUM, (64, 64, 1.0), dtype)
    row = torch.tensor(values).reshape(-1)
    x = row.repeat(64, 64 // row.numel()).to(dtype)
    with torch.no_grad():
        model.weight.fill_(weights)
        ref.weight.fill_(weights)
        out = model(x)
        expected = ref(x)
    assert torch.isfinite(out).all()
    assert torch.equal(out, model._gemm_sum(x))
    assert torch.isfinite(expected).all()


@kernel_test
def test_gemm_sum_infinite_input_gives_the_references_nan(gemm_sum_rewrite):
    ref, model = _pair(_GEMM_SUM, (64, 64, 1.5), torch.float32)
    torch.manual_seed(0)
    x = torch.rand(64, 64)
    x[3, 5] = torch.inf
    with torch.no_grad():
        out = model(x)
        expected = ref(x)
    assert torch.equal(out.isnan(), expected.isnan()) and out[3].isnan()


@pytest.mark.parametrize(
    ("x_shape", "n"),
    [
        pytest.param((16,), 8, id="1-D"),
        pytest.param((2, 3, 16), 8, id="3-D"),
        pytest.param((2, 3, 4, 16), 8, id="4-D"),
        pytest.param((8, 16), 0, id="no-outputs"),
        pytest.param((0, 16), 8, id="no-rows"),
    ],
)
def test_gemm_sum_other_shapes_run_eagerly(x_shape, n, gemm_sum_rewrite):
    ref, model = _pair(_GEMM_SUM, (16, n, 1.5), torch.float32)
    torch.manual_seed(0)
    x = torch.rand(*x_shape)
    with torch.no_grad():
        if x.dim() == 1:
            # The reference's sum has no dim 1 for a 1-D product: the same error.
            with pytest.raises(IndexError):
                ref(x)
            with pytest.raises(IndexError):
                model(x)
            return
        out = model(x)
        expected = ref(x)
    assert torch.equal(out, expected)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float16])
def test_gemm_sum_other_float_dtypes_run_eagerly(dtype, gemm_sum_rewrite):
    ref, model = _pair(_GEMM_SUM, (16, 8, 1.5), dtype)
    torch.manual_seed(0)
    x = torch.rand(4, 16).to(dtype)
    with torch.no_grad():
        out = model(x)
        expected = ref(x)
    assert out.dtype == dtype and torch.equal(out, expected)


@pytest.mark.parametrize("dtype", [torch.int32, torch.uint8])
def test_gemm_sum_integer_inputs_raise_as_the_reference(dtype, gemm_sum_rewrite):
    ref, model = _pair(_GEMM_SUM, (16, 8, 1.5), torch.float32)
    x = torch.ones(4, 16, dtype=dtype)
    with torch.no_grad(), pytest.raises(RuntimeError):
        ref(x)
    with torch.no_grad(), pytest.raises(RuntimeError):
        model(x)
