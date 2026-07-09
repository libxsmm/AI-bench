"""Tests for the ai-bench command-line interface (ai_bench.cli).

The runner classes are mocked for most tests so that ``main()`` never
executes real kernels; a couple of integration tests exercise the full path
using trivial mock kernels and problem specs to keep them self-contained and
fast.
"""

from unittest import mock

import pytest
import torch

from ai_bench import cli
from ai_bench.harness import core as ai_hc
from ai_bench.harness import runner as ai_hr
from ai_bench.utils import finder

# Minimal, trivial kernel/spec content reused by the integration tests.
_SPEC_CI = """
inputs:
  X:
    shape: [N]
    dtype: float32
ci:
  - params: [X]
    dims:
      N: 8
"""

_KERNEL_DOUBLE = """
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 2
"""

# Spec exposing a custom, non-default variant key used to exercise --variant.
_SPEC_CUSTOM_VARIANT = """
inputs:
  X:
    shape: [N]
    dtype: float32
my-variant:
  - params: [X]
    dims:
      N: 8
    flop: N
    mem_bytes: N * 4
"""


@pytest.fixture(autouse=True)
def _reset_finder():
    """Ensure finder global path config never leaks between CLI tests."""
    finder.reset_configuration()
    yield
    finder.reset_configuration()


@pytest.fixture
def mock_kb_runner():
    """Patch KernelBenchRunner so main() never runs real kernels."""
    with mock.patch("ai_bench.cli.runner.KernelBenchRunner") as m:
        yield m


@pytest.fixture
def mock_kernel_runner():
    """Patch KernelRunner so --kernel mode never runs real kernels."""
    with mock.patch("ai_bench.cli.runner.KernelRunner") as m:
        yield m


class TestCreateParser:
    """Tests for argument parsing."""

    def test_defaults(self):
        """Test that parser defaults match CPU / PyTorch / CI run."""
        args = cli.create_parser().parse_args([])

        assert args.xpu is False
        assert args.cuda is False
        assert args.triton is False
        assert args.helion is False
        assert args.torch_compile is False
        assert args.mlir is False
        assert args.gluon is False
        assert args.sycl is False
        assert args.bench is False
        assert args.gflops is False
        assert args.mbs is False
        assert args.kernel is None
        assert args.no_env is False
        assert args.csv is None
        assert args.note == ""
        assert args.variant is None

    def test_device_group_mutually_exclusive(self):
        """Test that --xpu and --cuda cannot be combined."""
        with pytest.raises(SystemExit):
            cli.create_parser().parse_args(["--xpu", "--cuda"])

    def test_backend_group_mutually_exclusive(self):
        """Test that two backends cannot be combined."""
        with pytest.raises(SystemExit):
            cli.create_parser().parse_args(["--triton", "--helion"])

    def test_env_group_mutually_exclusive(self):
        """Test that --no-env and --env-file cannot be combined."""
        with pytest.raises(SystemExit):
            cli.create_parser().parse_args(["--no-env", "--env-file", "x"])

    def test_kernel_requires_two_paths(self):
        """Test that --kernel requires exactly two arguments."""
        with pytest.raises(SystemExit):
            cli.create_parser().parse_args(["--kernel", "only_one.py"])

    def test_version(self, capsys):
        """Test that --version exits cleanly and prints the program name."""
        with pytest.raises(SystemExit) as exc:
            cli.create_parser().parse_args(["--version"])

        assert exc.value.code == 0
        assert "ai-bench" in capsys.readouterr().out


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
    def test_device_selection(self, mock_kb_runner, argv, expected):
        """Test that the correct torch device is passed to the runner."""
        rc = cli.main([*argv, "--no-env"])

        assert rc == 0
        assert mock_kb_runner.call_args.kwargs["device"] == torch.device(expected)
        mock_kb_runner.return_value.run_kernels.assert_called_once_with()


class TestMainBackendSelection:
    """Tests for backend selection in main()."""

    @pytest.mark.parametrize(
        "flag, expected",
        [
            ([], ai_hc.Backend.PYTORCH),
            (["--triton"], ai_hc.Backend.TRITON),
            (["--helion"], ai_hc.Backend.HELION),
            (["--torch-compile"], ai_hc.Backend.PYTORCH_COMPILE),
            (["--mlir"], ai_hc.Backend.MLIR),
            (["--gluon"], ai_hc.Backend.GLUON),
            (["--sycl"], ai_hc.Backend.SYCL),
        ],
    )
    def test_backend_selection(self, mock_kb_runner, flag, expected):
        """Test that the correct backend is passed to the runner."""
        rc = cli.main([*flag, "--no-env"])

        assert rc == 0
        assert mock_kb_runner.call_args.kwargs["backend"] == expected


class TestMainSpecType:
    """Tests for spec-type selection in main()."""

    @pytest.mark.parametrize(
        "argv, expected",
        [
            (["--no-env"], ai_hc.SpecKey.V_CI),
            (["--no-env", "--bench"], ai_hc.SpecKey.V_BENCH_CPU),
            (["--no-env", "--bench", "--xpu"], ai_hc.SpecKey.V_BENCH_GPU),
            (["--no-env", "--bench", "--cuda"], ai_hc.SpecKey.V_BENCH_GPU),
        ],
    )
    def test_spec_type(self, mock_kb_runner, argv, expected):
        """Test that --bench and device select the right spec category."""
        rc = cli.main(argv)

        assert rc == 0
        assert mock_kb_runner.call_args.kwargs["spec_type"] == expected

    @pytest.mark.parametrize(
        "argv, expected",
        [
            (["--no-env", "--variant", "bench-gpu-1"], "bench-gpu-1"),
            (["--no-env", "--bench", "--variant", "custom"], "custom"),
            (
                ["--no-env", "--bench", "--xpu", "--variant", "my-variant"],
                "my-variant",
            ),
        ],
    )
    def test_variant_overrides_spec_type(self, mock_kb_runner, argv, expected):
        """Test that --variant overrides the default spec-type selection."""
        rc = cli.main(argv)

        assert rc == 0
        assert mock_kb_runner.call_args.kwargs["spec_type"] == expected


class TestMainUnits:
    """Tests for FLOPS / memory-bandwidth unit selection in main()."""

    def test_default_units(self, mock_kb_runner):
        """Test default units are TFLOPS and GB/s."""
        cli.main(["--no-env"])

        kwargs = mock_kb_runner.call_args.kwargs
        assert kwargs["flops_unit"] == ai_hr.FlopsUnit.TFLOPS
        assert kwargs["mem_bw_unit"] == ai_hr.MemBwUnit.GBS

    def test_gflops_and_mbs(self, mock_kb_runner):
        """Test --gflops and --mbs switch the reported units."""
        cli.main(["--no-env", "--gflops", "--mbs"])

        kwargs = mock_kb_runner.call_args.kwargs
        assert kwargs["flops_unit"] == ai_hr.FlopsUnit.GFLOPS
        assert kwargs["mem_bw_unit"] == ai_hr.MemBwUnit.MBS


class TestMainCsv:
    """Tests for CSV logging options in main()."""

    def test_csv_and_note(self, mock_kb_runner):
        """Test that --csv and --note are forwarded to the runner."""
        cli.main(["--no-env", "--csv", "out.csv", "--note", "hello"])

        kwargs = mock_kb_runner.call_args.kwargs
        assert kwargs["csv_path"] == "out.csv"
        assert kwargs["note"] == "hello"

    def test_csv_defaults(self, mock_kb_runner):
        """Test CSV defaults (no path, empty note)."""
        cli.main(["--no-env"])

        kwargs = mock_kb_runner.call_args.kwargs
        assert kwargs["csv_path"] is None
        assert kwargs["note"] == ""


class TestMainKernelMode:
    """Tests for single-kernel mode (--kernel) in main()."""

    def test_valid_kernel_and_spec(self, mock_kernel_runner, tmp_path):
        """Test that valid paths dispatch to KernelRunner.run_kernel_spec."""
        kernel = tmp_path / "double.py"
        kernel.write_text(_KERNEL_DOUBLE)
        spec = tmp_path / "double.yaml"
        spec.write_text(_SPEC_CI)

        rc = cli.main(["--no-env", "--kernel", str(kernel), str(spec)])

        assert rc == 0
        mock_kernel_runner.return_value.run_kernel_spec.assert_called_once_with(
            kernel, spec
        )

    def test_kernel_bad_suffix(self, mock_kernel_runner, tmp_path, capsys):
        """Test non-.py kernel path is rejected before running."""
        kernel = tmp_path / "double.txt"
        kernel.write_text(_KERNEL_DOUBLE)
        spec = tmp_path / "double.yaml"
        spec.write_text(_SPEC_CI)

        rc = cli.main(["--no-env", "--kernel", str(kernel), str(spec)])

        assert rc == 1
        assert "Expected .py kernel file" in capsys.readouterr().err
        mock_kernel_runner.assert_not_called()

    def test_kernel_missing_file(self, mock_kernel_runner, tmp_path, capsys):
        """Test missing kernel file is rejected before running."""
        kernel = tmp_path / "missing.py"
        spec = tmp_path / "double.yaml"
        spec.write_text(_SPEC_CI)

        rc = cli.main(["--no-env", "--kernel", str(kernel), str(spec)])

        assert rc == 1
        assert "Kernel path not found" in capsys.readouterr().err
        mock_kernel_runner.assert_not_called()

    def test_spec_bad_suffix(self, mock_kernel_runner, tmp_path, capsys):
        """Test non-.yaml spec path is rejected before running."""
        kernel = tmp_path / "double.py"
        kernel.write_text(_KERNEL_DOUBLE)
        spec = tmp_path / "double.txt"
        spec.write_text(_SPEC_CI)

        rc = cli.main(["--no-env", "--kernel", str(kernel), str(spec)])

        assert rc == 1
        assert "Expected .yaml spec file" in capsys.readouterr().err
        mock_kernel_runner.assert_not_called()

    def test_spec_missing_file(self, mock_kernel_runner, tmp_path, capsys):
        """Test missing spec file is rejected before running."""
        kernel = tmp_path / "double.py"
        kernel.write_text(_KERNEL_DOUBLE)
        spec = tmp_path / "missing.yaml"

        rc = cli.main(["--no-env", "--kernel", str(kernel), str(spec)])

        assert rc == 1
        assert "Spec path not found" in capsys.readouterr().err
        mock_kernel_runner.assert_not_called()


class TestMainErrorHandling:
    """Tests for main() error handling and exit codes."""

    def test_configuration_error(self, mock_kb_runner, capsys):
        """Test ConfigurationError yields exit code 1."""
        mock_kb_runner.side_effect = finder.ConfigurationError("boom")

        rc = cli.main(["--no-env"])

        assert rc == 1
        assert "Configuration error" in capsys.readouterr().err

    def test_value_error(self, mock_kb_runner, capsys):
        """Test ValueError yields exit code 1."""
        mock_kb_runner.side_effect = ValueError("bad backend")

        rc = cli.main(["--no-env"])

        assert rc == 1
        assert "Error" in capsys.readouterr().err

    def test_keyboard_interrupt(self, mock_kb_runner, capsys):
        """Test KeyboardInterrupt yields exit code 130."""
        mock_kb_runner.side_effect = KeyboardInterrupt()

        rc = cli.main(["--no-env"])

        assert rc == 130
        assert "Interrupted" in capsys.readouterr().err


class TestMainPathConfiguration:
    """Tests for path-configuration wiring in main()."""

    def test_configure_called_with_dirs(self, mock_kb_runner, tmp_path):
        """Test that provided dirs are forwarded to finder.configure."""
        specs = tmp_path / "specs"
        specs.mkdir()
        kernels = tmp_path / "kernels"
        kernels.mkdir()

        with mock.patch("ai_bench.cli.finder.configure") as mock_configure:
            cli.main(
                [
                    "--no-env",
                    "--specs-dir",
                    str(specs),
                    "--kernels-dir",
                    str(kernels),
                ]
            )

        mock_configure.assert_called_once()
        kwargs = mock_configure.call_args.kwargs
        assert kwargs["specs_dir"] == specs
        assert kwargs["kernels_dir"] == kernels

    def test_configure_not_called_without_dirs(self, mock_kb_runner):
        """Test that finder.configure is skipped when no dirs are given."""
        with mock.patch("ai_bench.cli.finder.configure") as mock_configure:
            cli.main(["--no-env"])

        mock_configure.assert_not_called()


class TestMainEnvFile:
    """Tests for .env handling in main()."""

    def test_no_env_skips_load(self, mock_kb_runner):
        """Test that --no-env prevents any .env loading."""
        with mock.patch("ai_bench.cli.finder.load_env") as mock_load:
            cli.main(["--no-env"])

        mock_load.assert_not_called()

    def test_auto_load_env(self, mock_kb_runner):
        """Test that .env auto-detection runs by default."""
        with mock.patch("ai_bench.cli.finder.load_env") as mock_load:
            cli.main([])

        mock_load.assert_called_once_with()

    def test_env_file_explicit_path(self, mock_kb_runner, tmp_path):
        """Test that an explicit --env-file path is passed to load_env."""
        env_file = tmp_path / ".env"
        env_file.write_text("AIBENCH_CARD=BMG\n")

        with mock.patch("ai_bench.cli.finder.load_env", return_value=True) as mock_load:
            cli.main(["--env-file", str(env_file)])

        mock_load.assert_called_once_with(env_file)

    def test_env_file_not_found_warns(self, mock_kb_runner, tmp_path, capsys):
        """Test that a missing --env-file prints a warning but still runs."""
        missing = tmp_path / "missing.env"

        with mock.patch("ai_bench.cli.finder.load_env", return_value=False):
            rc = cli.main(["--env-file", str(missing)])

        assert rc == 0
        assert ".env file not found" in capsys.readouterr().err


class TestMainIntegration:
    """End-to-end tests using trivial mock kernels and specs."""

    def _build_kernelbench_tree(self, tmp_path):
        """Create a minimal KernelBench specs/kernels layout."""
        specs_root = tmp_path / "specs"
        spec_dir = specs_root / "KernelBench" / "level1"
        spec_dir.mkdir(parents=True)
        (spec_dir / "double.yaml").write_text(_SPEC_CI)

        kernels_root = tmp_path / "kernels"
        kernel_dir = kernels_root / "KernelBench" / "level1"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "double.py").write_text(_KERNEL_DOUBLE)

        return specs_root, kernels_root

    def test_kernelbench_ci_run(self, tmp_path):
        """Test a full CI run over a mocked KernelBench tree returns 0."""
        specs_root, kernels_root = self._build_kernelbench_tree(tmp_path)

        rc = cli.main(
            [
                "--no-env",
                "--specs-dir",
                str(specs_root),
                "--kernels-dir",
                str(kernels_root),
            ]
        )

        assert rc == 0

    def test_single_kernel_ci_run(self, tmp_path):
        """Test a full --kernel CI run on a trivial kernel/spec returns 0."""
        kernel = tmp_path / "double.py"
        kernel.write_text(_KERNEL_DOUBLE)
        spec = tmp_path / "double.yaml"
        spec.write_text(_SPEC_CI)

        rc = cli.main(["--no-env", "--kernel", str(kernel), str(spec)])

        assert rc == 0

    def test_single_kernel_custom_variant_run(self, tmp_path):
        """Test that --variant selects and benchmarks a custom spec variant."""
        kernel = tmp_path / "double.py"
        kernel.write_text(_KERNEL_DOUBLE)
        spec = tmp_path / "double.yaml"
        spec.write_text(_SPEC_CUSTOM_VARIANT)

        rc = cli.main(
            [
                "--no-env",
                "--variant",
                "my-variant",
                "--kernel",
                str(kernel),
                str(spec),
            ]
        )

        assert rc == 0
