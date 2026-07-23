"""Tests for the ai-bench-compare CLI (ai_bench.cli_compare).

The heavy ``benchmark_problem`` call and the print helpers are mocked for most
tests so that ``main()`` never runs real kernels; a single integration test
exercises the full path using a trivial mock kernel and problem spec to keep
it self-contained and fast.
"""

import types
from unittest import mock

import pytest
import torch

from ai_bench import cli_compare
from ai_bench.harness import core as ai_hc
from ai_bench.harness import runner as ai_hr
from ai_bench.harness.runner import benchmark_compare
from ai_bench.utils import finder

# Minimal, trivial kernel/spec content reused by the integration test. The
# spec carries explicit flop/mem_bytes so benchmarking never estimates them.
_SPEC_CI = """
inputs:
  X:
    shape: [N]
    dtype: float32
ci:
  - params: [X]
    dims:
      N: 8
    flop: 8*N
    mem_bytes: (N + N) * 4
"""

_KERNEL_DOUBLE = """
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 2
"""


@pytest.fixture(autouse=True)
def _reset_finder():
    """Ensure finder global path config never leaks between CLI tests."""
    finder.reset_configuration()
    yield
    finder.reset_configuration()


@pytest.fixture
def mock_backend(monkeypatch):
    """Patch heavy execution and output so main() never runs real kernels."""
    monkeypatch.setattr(
        cli_compare,
        "get_problem_choices",
        lambda: ["level1/double", "level2/other"],
    )
    benchmark = mock.MagicMock(
        return_value={"problem": "level1/double", "variants": []}
    )
    full = mock.MagicMock()
    brief = mock.MagicMock()
    monkeypatch.setattr(cli_compare, "benchmark_problem", benchmark)
    monkeypatch.setattr(cli_compare, "print_comparison", full)
    monkeypatch.setattr(cli_compare, "print_comparison_brief", brief)
    return types.SimpleNamespace(benchmark=benchmark, full=full, brief=brief)


class TestGetProblemChoices:
    """Tests for the KernelBench problem-choice generator."""

    def test_lists_sorted_choices(self, tmp_path):
        """Test choices are 'level/name' entries sorted by level then name."""
        specs = tmp_path / "specs"
        kb = specs / "KernelBench"
        (kb / "level1").mkdir(parents=True)
        (kb / "level2").mkdir(parents=True)
        (kb / "level1" / "b.yaml").write_text(_SPEC_CI)
        (kb / "level1" / "a.yaml").write_text(_SPEC_CI)
        (kb / "level2" / "c.yaml").write_text(_SPEC_CI)

        finder.configure(specs_dir=specs)

        assert cli_compare.get_problem_choices() == [
            "level1/a",
            "level1/b",
            "level2/c",
        ]


class TestMainDeviceSelection:
    """Tests for device selection in main()."""

    @pytest.mark.parametrize(
        "argv, expected",
        [
            ([], "cpu"),
            (["--xpu"], "xpu"),
            (["--cuda"], "cuda"),
        ],
    )
    def test_device_selection(self, mock_backend, argv, expected):
        """Test that the correct torch device is passed to benchmark_problem."""
        rc = cli_compare.main(["--problem", "level1/double", *argv])

        assert rc == 0
        assert mock_backend.benchmark.call_args.kwargs["device"] == torch.device(
            expected
        )


class TestMainBackends:
    """Tests for backend selection in main()."""

    def test_default_backends(self, mock_backend):
        """Test default backends are PyTorch and PyTorch-compile."""
        cli_compare.main(["--problem", "level1/double"])

        assert mock_backend.benchmark.call_args.kwargs["backends"] == [
            ai_hc.Backend.PYTORCH,
            ai_hc.Backend.PYTORCH_COMPILE,
        ]

    def test_explicit_backends(self, mock_backend):
        """Test that --backends selects and orders the requested backends."""
        cli_compare.main(
            [
                "--problem",
                "level1/double",
                "--backends",
                "pytorch",
                "triton",
            ]
        )

        assert mock_backend.benchmark.call_args.kwargs["backends"] == [
            ai_hc.Backend.PYTORCH,
            ai_hc.Backend.TRITON,
        ]

    def test_invalid_backend_exits(self, mock_backend):
        """Test that an unknown backend is rejected by argparse."""
        with pytest.raises(SystemExit):
            cli_compare.main(
                [
                    "--problem",
                    "level1/double",
                    "--backends",
                    "nonsense",
                ]
            )


class TestMainSpecType:
    """Tests for spec-type selection in main()."""

    @pytest.mark.parametrize(
        "argv, expected",
        [
            ([], ai_hc.SpecKey.V_BENCH_CPU),
            (["--xpu"], ai_hc.SpecKey.V_BENCH_GPU),
            (["--cuda"], ai_hc.SpecKey.V_BENCH_GPU),
            (["--ci"], ai_hc.SpecKey.V_CI),
        ],
    )
    def test_spec_type(self, mock_backend, argv, expected):
        """Test device and --ci select the right default spec category."""
        cli_compare.main(["--problem", "level1/double", *argv])

        assert mock_backend.benchmark.call_args.kwargs["spec_type"] == expected

    def test_variant_overrides_ci(self, mock_backend):
        """Test that --variant takes priority over --ci and device defaults."""
        cli_compare.main(
            [
                "--problem",
                "level1/double",
                "--variant",
                "bench-gpu-1",
                "--ci",
            ]
        )

        assert mock_backend.benchmark.call_args.kwargs["spec_type"] == "bench-gpu-1"


class TestMainTolerances:
    """Tests for tolerance overrides in main()."""

    def test_default_tolerances(self, mock_backend):
        """Test tolerances default to None (resolved per-spec downstream)."""
        cli_compare.main(["--problem", "level1/double"])

        kwargs = mock_backend.benchmark.call_args.kwargs
        assert kwargs["rtol"] is None
        assert kwargs["atol"] is None

    def test_explicit_tolerances(self, mock_backend):
        """Test that --rtol and --atol are forwarded as floats."""
        cli_compare.main(
            [
                "--problem",
                "level1/double",
                "--rtol",
                "1e-3",
                "--atol",
                "1e-6",
            ]
        )

        kwargs = mock_backend.benchmark.call_args.kwargs
        assert kwargs["rtol"] == 1e-3
        assert kwargs["atol"] == 1e-6


class TestMainDtype:
    """Tests for the --dtype variant filter."""

    def test_dtype_forwards_to_benchmark(self, mock_backend):
        """Test dtype is forwarded to comparison benchmarking."""
        rc = cli_compare.main(["--problem", "level1/double", "--dtype", "bfloat16"])

        assert rc == 0
        assert mock_backend.benchmark.call_args.kwargs["dtype"] == "bfloat16"


class TestMainProblem:
    """Tests for problem argument handling in main()."""

    def test_problem_forwarded(self, mock_backend):
        """Test that the selected problem is forwarded to benchmark_problem."""
        cli_compare.main(["--problem", "level2/other"])

        assert mock_backend.benchmark.call_args.kwargs["problem"] == "level2/other"

    def test_missing_problem_exits(self, mock_backend):
        """Test that omitting the required --problem argument exits."""
        with pytest.raises(SystemExit):
            cli_compare.main([])

    def test_invalid_problem_choice_exits(self, mock_backend):
        """Test that a problem outside the known choices is rejected."""
        with pytest.raises(SystemExit):
            cli_compare.main(["--problem", "level9/nope"])


class TestMainOutputFormat:
    """Tests for brief vs. full output selection in main()."""

    def test_full_output_default(self, mock_backend):
        """Test the full comparison printer is used by default."""
        cli_compare.main(["--problem", "level1/double"])

        mock_backend.full.assert_called_once()
        mock_backend.brief.assert_not_called()

    def test_brief_output(self, mock_backend):
        """Test --brief switches to the brief comparison printer."""
        cli_compare.main(["--problem", "level1/double", "--brief"])

        mock_backend.brief.assert_called_once()
        mock_backend.full.assert_not_called()


class TestMainUnits:
    """Tests for FLOPS and memory-bandwidth unit selection."""

    def test_default_units(self, mock_backend):
        """Test default units are TFLOPS and GB/s."""
        cli_compare.main(["--problem", "level1/double"])

        kwargs = mock_backend.benchmark.call_args.kwargs
        assert kwargs["flops_unit"] == ai_hr.FlopsUnit.TFLOPS
        assert kwargs["mem_bw_unit"] == ai_hr.MemBwUnit.GBS

    def test_gflops_and_mbs(self, mock_backend):
        """Test output flags select GFLOPS and MB/s."""
        cli_compare.main(["--problem", "level1/double", "--gflops", "--mbs"])

        kwargs = mock_backend.benchmark.call_args.kwargs
        assert kwargs["flops_unit"] == ai_hr.FlopsUnit.GFLOPS
        assert kwargs["mem_bw_unit"] == ai_hr.MemBwUnit.MBS

    def test_full_output_uses_result_units(self, monkeypatch):
        """Test full output labels performance columns with result units."""
        result = types.SimpleNamespace(
            meas_us=1.0,
            flops=2.0,
            flops_unit=ai_hr.FlopsUnit.GFLOPS,
            mem_bw=3.0,
            mem_bw_unit=ai_hr.MemBwUnit.MBS,
        )
        variant = benchmark_compare.VariantResult(
            backends={"pytorch": result},
            speedups={"pytorch": 1.0},
        )
        info = mock.MagicMock()
        monkeypatch.setattr(benchmark_compare.logger, "info", info)

        benchmark_compare.print_variant_results(variant)

        output = "\n".join(str(call.args[0]) for call in info.call_args_list)
        assert "GFLOPS" in output
        assert "MB/s" in output


class TestMainErrorHandling:
    """Tests for main() error handling and exit codes."""

    def test_file_not_found(self, mock_backend, capsys):
        """Test FileNotFoundError yields exit code 1."""
        mock_backend.benchmark.side_effect = FileNotFoundError("no spec")

        rc = cli_compare.main(["--problem", "level1/double"])

        assert rc == 1
        assert "Error" in capsys.readouterr().err

    def test_value_error(self, mock_backend, capsys):
        """Test ValueError yields exit code 1."""
        mock_backend.benchmark.side_effect = ValueError("bad problem")

        rc = cli_compare.main(["--problem", "level1/double"])

        assert rc == 1
        assert "Error" in capsys.readouterr().err

    def test_keyboard_interrupt(self, mock_backend, capsys):
        """Test KeyboardInterrupt yields exit code 130."""
        mock_backend.benchmark.side_effect = KeyboardInterrupt()

        rc = cli_compare.main(["--problem", "level1/double"])

        assert rc == 130
        assert "Interrupted" in capsys.readouterr().err


class TestMainIntegration:
    """End-to-end test using a trivial mock kernel and spec."""

    def test_ci_comparison_run(self, tmp_path):
        """Test a full single-backend CI comparison run returns 0."""
        specs_root = tmp_path / "specs"
        spec_dir = specs_root / "KernelBench" / "level1"
        spec_dir.mkdir(parents=True)
        (spec_dir / "double.yaml").write_text(_SPEC_CI)

        kernels_root = tmp_path / "kernels"
        kernel_dir = kernels_root / "KernelBench" / "level1"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "double.py").write_text(_KERNEL_DOUBLE)

        finder.configure(specs_dir=specs_root, kernels_dir=kernels_root)

        rc = cli_compare.main(
            [
                "--problem",
                "level1/double",
                "--ci",
                "--backends",
                "pytorch",
                "--brief",
            ]
        )

        assert rc == 0
