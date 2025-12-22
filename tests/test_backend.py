"""Tests for AI Bench harness with focus on backend selection and mocked execution."""

from pathlib import Path
import tempfile
from unittest import mock

import pytest
import torch

from ai_bench.harness import core as ai_hc
from ai_bench.harness import runner
from ai_bench.harness import testing


class TestBackendEnum:
    """Tests for Backend enum."""

    def test_backend_values(self):
        """Test that backend enum has expected values."""
        assert ai_hc.Backend.PYTORCH == "pytorch"
        assert ai_hc.Backend.PYTORCH_COMPILE == "pytorch-compile"
        assert ai_hc.Backend.TRITON == "triton"

    def test_backend_iteration(self):
        """Test that all backends can be iterated."""
        backends = list(ai_hc.Backend)
        assert len(backends) == 3
        assert ai_hc.Backend.PYTORCH in backends
        assert ai_hc.Backend.PYTORCH_COMPILE in backends
        assert ai_hc.Backend.TRITON in backends

    def test_backend_from_string(self):
        """Test creating backend from string."""
        assert ai_hc.Backend("pytorch") == ai_hc.Backend.PYTORCH
        assert ai_hc.Backend("pytorch-compile") == ai_hc.Backend.PYTORCH_COMPILE
        assert ai_hc.Backend("triton") == ai_hc.Backend.TRITON

    def test_backend_invalid_string(self):
        """Test that invalid string raises error."""
        with pytest.raises(ValueError):
            ai_hc.Backend("invalid")


class TestSpecFunctions:
    """Tests for spec parsing functions."""

    def test_get_inputs(self):
        """Test input tensor generation."""
        variant = {
            ai_hc.VKey.PARAMS: ["A", "B"],
            ai_hc.VKey.DIMS: {"N": 64},
        }
        inputs = {
            "A": {ai_hc.InKey.SHAPE: ["N", "N"], ai_hc.InKey.TYPE: "float32"},
            "B": {ai_hc.InKey.SHAPE: ["N", "N"], ai_hc.InKey.TYPE: "float32"},
        }

        tensors = ai_hc.get_inputs(variant, inputs, device=torch.device("cpu"))

        assert len(tensors) == 2
        assert tensors[0].shape == (64, 64)
        assert tensors[1].shape == (64, 64)
        assert tensors[0].dtype == torch.float32

    def test_get_inputs_float16(self):
        """Test input tensor generation with float16."""
        variant = {
            ai_hc.VKey.PARAMS: ["X"],
            ai_hc.VKey.DIMS: {"B": 4, "S": 128, "H": 256},
        }
        inputs = {
            "X": {ai_hc.InKey.SHAPE: ["B", "S", "H"], ai_hc.InKey.TYPE: "float16"},
        }

        tensors = ai_hc.get_inputs(variant, inputs, device=torch.device("cpu"))

        assert len(tensors) == 1
        assert tensors[0].shape == (4, 128, 256)
        assert tensors[0].dtype == torch.float16

    def test_get_inits(self):
        """Test initialization value extraction."""
        variant = {
            ai_hc.VKey.DIMS: {"N": 256, "H": 512},
        }
        inits = [
            {ai_hc.InitKey.DIM: "N"},
            {ai_hc.InitKey.DIM: "H"},
        ]

        init_vals = ai_hc.get_inits(variant, inits)

        assert init_vals == [256, 512]

    def test_get_inits_empty(self):
        """Test initialization with empty inits."""
        variant = {ai_hc.VKey.DIMS: {"N": 256}}
        inits = []

        init_vals = ai_hc.get_inits(variant, inits)

        assert init_vals == []

    def test_get_variant_torch_dtype(self):
        """Test variant dtype extraction."""
        variant_with_dtype = {ai_hc.VKey.TYPE: "float16"}
        variant_without_dtype = {ai_hc.VKey.DIMS: {"N": 64}}

        assert ai_hc.get_variant_torch_dtype(variant_with_dtype) == torch.float16
        assert ai_hc.get_variant_torch_dtype(variant_without_dtype) is None

    def test_get_flop_numeric(self):
        """Test FLOP extraction with numeric value."""
        variant = {ai_hc.VKey.FLOP: 1000000}

        assert ai_hc.get_flop(variant) == 1000000

    def test_get_flop_formula(self):
        """Test FLOP extraction with formula."""
        variant = {
            ai_hc.VKey.DIMS: {"M": 64, "N": 64, "K": 64},
            ai_hc.VKey.FLOP: "2*M*N*K",
        }

        flop = ai_hc.get_flop(variant)

        assert flop == 2 * 64 * 64 * 64

    def test_get_flop_missing(self):
        """Test FLOP extraction when missing."""
        variant = {ai_hc.VKey.DIMS: {"N": 64}}

        assert ai_hc.get_flop(variant) is None

    def test_get_mem_bytes_numeric(self):
        """Test memory bytes extraction with numeric value."""
        variant = {ai_hc.VKey.MEM_BYTES: 16384}

        assert ai_hc.get_mem_bytes(variant) == 16384

    def test_get_mem_bytes_formula(self):
        """Test memory bytes extraction with formula."""
        variant = {
            ai_hc.VKey.DIMS: {"M": 16, "N": 32, "K": 64},
            ai_hc.VKey.MEM_BYTES: "(M*K + K*N + M*N) * 2",
        }

        mem_bytes = ai_hc.get_mem_bytes(variant)

        assert mem_bytes == (16 * 64 + 64 * 32 + 16 * 32) * 2

    def test_get_mem_bytes_missing(self):
        """Test memory bytes extraction when missing."""
        variant = {ai_hc.VKey.DIMS: {"N": 64}}

        assert ai_hc.get_mem_bytes(variant) is None


class TestKernelBenchRunnerInit:
    """Tests for KernelBenchRunner initialization."""

    @mock.patch("os.path.isdir")
    def test_init_pytorch_backend(self, mock_isdir):
        """Test runner initialization with PyTorch backend."""
        mock_isdir.return_value = True

        kb_runner = runner.KernelBenchRunner(
            spec_type=ai_hc.SpecKey.V_CI,
            device=torch.device("cpu"),
            backend=ai_hc.Backend.PYTORCH,
        )

        assert kb_runner.backend == ai_hc.Backend.PYTORCH
        assert "KernelBench" in str(kb_runner.kernels)
        assert "triton" not in str(kb_runner.kernels)

    @mock.patch("os.path.isdir")
    def test_init_pytorch_compile_backend(self, mock_isdir):
        """Test runner initialization with PyTorch compile backend."""
        mock_isdir.return_value = True

        kb_runner = runner.KernelBenchRunner(
            spec_type=ai_hc.SpecKey.V_CI,
            device=torch.device("cpu"),
            backend=ai_hc.Backend.PYTORCH_COMPILE,
        )

        assert kb_runner.backend == ai_hc.Backend.PYTORCH_COMPILE
        assert "KernelBench" in str(kb_runner.kernels)
        assert "triton" not in str(kb_runner.kernels)

    @mock.patch("os.path.isdir")
    def test_init_triton_backend(self, mock_isdir):
        """Test runner initialization with Triton backend."""
        mock_isdir.return_value = True

        kb_runner = runner.KernelBenchRunner(
            spec_type=ai_hc.SpecKey.V_CI,
            device=torch.device("cpu"),
            backend=ai_hc.Backend.TRITON,
        )

        assert kb_runner.backend == ai_hc.Backend.TRITON
        assert "triton" in str(kb_runner.kernels)

    @mock.patch("os.path.isdir")
    def test_init_missing_kernels_dir(self, mock_isdir):
        """Test runner raises error when kernels directory is missing."""
        mock_isdir.return_value = False

        with pytest.raises(ValueError, match="Missing kernels directory"):
            runner.KernelBenchRunner(
                backend=ai_hc.Backend.PYTORCH,
            )

    @mock.patch("os.path.isdir")
    def test_init_cpu_warmup_rep(self, mock_isdir):
        """Test CPU warmup and rep settings."""
        mock_isdir.return_value = True

        kb_runner = runner.KernelBenchRunner(
            device=torch.device("cpu"),
            backend=ai_hc.Backend.PYTORCH,
        )

        assert kb_runner.warmup == 5
        assert kb_runner.rep == 20

    @mock.patch("os.path.isdir")
    def test_init_xpu_warmup_rep(self, mock_isdir):
        """Test XPU warmup and rep settings."""
        mock_isdir.return_value = True

        kb_runner = runner.KernelBenchRunner(
            device=torch.device("xpu"),
            backend=ai_hc.Backend.PYTORCH,
        )

        assert kb_runner.warmup == 200
        assert kb_runner.rep == 100


class TestKernelBenchRunnerExecution:
    """Tests for KernelBenchRunner execution with mocked kernels."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for specs and kernels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create directory structure.
            specs_dir = tmpdir / "problems" / "specs" / "KernelBench" / "level1"
            pytorch_kernels_dir = (
                tmpdir / "third_party" / "KernelBench" / "KernelBench" / "level1"
            )
            triton_kernels_dir = (
                tmpdir / "backends" / "triton" / "KernelBench" / "level1"
            )

            specs_dir.mkdir(parents=True)
            pytorch_kernels_dir.mkdir(parents=True)
            triton_kernels_dir.mkdir(parents=True)

            yield {
                "root": tmpdir,
                "specs": specs_dir,
                "pytorch_kernels": pytorch_kernels_dir,
                "triton_kernels": triton_kernels_dir,
            }

    def _create_spec_file(self, specs_dir: Path, filename: str, content: str):
        """Helper to create a spec file."""
        (specs_dir / filename).write_text(content)

    def _create_kernel_file(self, kernels_dir: Path, filename: str, content: str):
        """Helper to create a kernel file."""
        (kernels_dir / filename).write_text(content)

    def test_load_model_success(self, temp_dirs):
        """Test successful model loading."""
        kernel_content = """
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * x
"""
        self._create_kernel_file(
            temp_dirs["pytorch_kernels"], "test_kernel.py", kernel_content
        )

        with mock.patch(
            "ai_bench.utils.finder.project_root", return_value=temp_dirs["root"]
        ):
            kb_runner = runner.KernelBenchRunner(
                backend=ai_hc.Backend.PYTORCH,
            )
            model_cls = kb_runner.load_model(
                temp_dirs["pytorch_kernels"] / "test_kernel.py"
            )

        assert model_cls is not None
        model = model_cls()
        x = torch.tensor([2.0])
        result = model(x)
        assert torch.allclose(result, torch.tensor([4.0]))

    def test_load_model_missing_file(self, temp_dirs):
        """Test loading non-existent model file."""
        with mock.patch(
            "ai_bench.utils.finder.project_root", return_value=temp_dirs["root"]
        ):
            kb_runner = runner.KernelBenchRunner(
                backend=ai_hc.Backend.PYTORCH,
            )
            model = kb_runner.load_model(
                temp_dirs["pytorch_kernels"] / "nonexistent.py"
            )

        assert model is None

    def test_load_model_no_model_class(self, temp_dirs):
        """Test loading file without Model class."""
        kernel_content = """
def some_function():
    pass
"""
        self._create_kernel_file(
            temp_dirs["pytorch_kernels"], "no_model.py", kernel_content
        )

        with mock.patch(
            "ai_bench.utils.finder.project_root", return_value=temp_dirs["root"]
        ):
            kb_runner = runner.KernelBenchRunner(
                backend=ai_hc.Backend.PYTORCH,
            )
            model = kb_runner.load_model(temp_dirs["pytorch_kernels"] / "no_model.py")

        assert model is None

    def test_run_kernels_ci_validation(self, temp_dirs):
        """Test CI validation run with mocked kernel."""
        spec_content = """
inputs:
  X:
    shape: [N]
    dtype: float32
ci:
  - params: [X]
    dims:
      N: 8
"""
        kernel_content = """
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.called = False

    def forward(self, x):
        return x * 2
"""
        self._create_spec_file(temp_dirs["specs"], "square.yaml", spec_content)
        self._create_kernel_file(
            temp_dirs["pytorch_kernels"], "square.py", kernel_content
        )

        with mock.patch(
            "ai_bench.utils.finder.project_root", return_value=temp_dirs["root"]
        ):
            kb_runner = runner.KernelBenchRunner(
                spec_type=ai_hc.SpecKey.V_CI,
                device=torch.device("cpu"),
                backend=ai_hc.Backend.PYTORCH,
            )
            # Should not raise any exceptions.
            kb_runner.run_kernels()

    def test_run_kernels_different_backends(self, temp_dirs):
        """Test running same spec with different backends."""
        spec_content = """
inputs:
  A:
    shape: [N, N]
    dtype: float32
ci:
  - params: [A]
    dims:
      N: 4
"""
        pytorch_kernel = """
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backend = "pytorch"

    def forward(self, x):
        return x @ x  # PyTorch matmul
"""
        triton_kernel = """
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backend = "triton"

    def forward(self, x):
        # Simulated Triton kernel (same result for testing).
        return x @ x
"""
        self._create_spec_file(temp_dirs["specs"], "matmul.yaml", spec_content)
        self._create_kernel_file(
            temp_dirs["pytorch_kernels"], "matmul.py", pytorch_kernel
        )
        self._create_kernel_file(
            temp_dirs["triton_kernels"], "matmul.py", triton_kernel
        )

        with mock.patch(
            "ai_bench.utils.finder.project_root", return_value=temp_dirs["root"]
        ):
            # Run with PyTorch backend.
            pytorch_runner = runner.KernelBenchRunner(
                spec_type=ai_hc.SpecKey.V_CI,
                device=torch.device("cpu"),
                backend=ai_hc.Backend.PYTORCH,
            )
            assert pytorch_runner.backend == ai_hc.Backend.PYTORCH

            # Run with PyTorch compile backend.
            pytorch_compile_runner = runner.KernelBenchRunner(
                spec_type=ai_hc.SpecKey.V_CI,
                device=torch.device("cpu"),
                backend=ai_hc.Backend.PYTORCH_COMPILE,
            )
            assert pytorch_compile_runner.backend == ai_hc.Backend.PYTORCH_COMPILE

            # Run with Triton backend.
            triton_runner = runner.KernelBenchRunner(
                spec_type=ai_hc.SpecKey.V_CI,
                device=torch.device("cpu"),
                backend=ai_hc.Backend.TRITON,
            )
            assert triton_runner.backend == ai_hc.Backend.TRITON

            # Verify they use different kernel directories.
            assert "KernelBench" in str(pytorch_runner.kernels)
            assert "triton" not in str(pytorch_runner.kernels)
            assert "triton" in str(triton_runner.kernels)

    def test_run_kernels_with_inits(self, temp_dirs):
        """Test running kernel that requires initialization parameters."""
        spec_content = """
inputs:
  X:
    shape: [B, H]
    dtype: float32
inits:
  - dim: H
ci:
  - params: [X]
    dims:
      B: 2
      H: 16
"""
        kernel_content = """
import torch

class Model(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        return x * self.weight
"""
        self._create_spec_file(temp_dirs["specs"], "weighted.yaml", spec_content)
        self._create_kernel_file(
            temp_dirs["pytorch_kernels"], "weighted.py", kernel_content
        )

        with mock.patch(
            "ai_bench.utils.finder.project_root", return_value=temp_dirs["root"]
        ):
            kb_runner = runner.KernelBenchRunner(
                spec_type=ai_hc.SpecKey.V_CI,
                device=torch.device("cpu"),
                backend=ai_hc.Backend.PYTORCH,
            )
            # Should initialize model with hidden_size=16.
            kb_runner.run_kernels()


class TestTimerFunctions:
    """Tests for timing functions."""

    def test_time_cpu_basic(self):
        """Test CPU timing with simple function."""

        def simple_fn(x):
            return x * 2

        x = torch.randn(100)
        time_us = testing.time_cpu(simple_fn, (x,), warmup=2, rep=5)

        assert time_us > 0
        assert isinstance(time_us, float)

    def test_time_cpu_consistency(self):
        """Test that CPU timing is reasonably consistent."""

        def sleep_fn(x):
            # Do some work.
            for _ in range(100):
                x = x * 1.0001
            return x

        x = torch.randn(1000)
        times = [testing.time_cpu(sleep_fn, (x,), warmup=2, rep=10) for _ in range(3)]

        # Times should be within 10x of each other (very loose bound).
        assert max(times) / min(times) < 10

    def test_time_dispatcher(self):
        """Test time() dispatches to correct function based on device."""

        def simple_fn(x):
            return x + 1

        x = torch.randn(100)

        # CPU should work.
        time_us = testing.time(
            simple_fn, (x,), warmup=2, rep=5, device=torch.device("cpu")
        )
        assert time_us > 0

        # None device should default to CPU.
        time_us = testing.time(simple_fn, (x,), warmup=2, rep=5, device=None)
        assert time_us > 0

    def test_time_unsupported_device(self):
        """Test that unsupported device raises error."""

        def simple_fn(x):
            return x

        x = torch.randn(100)

        with pytest.raises(ValueError, match="Unsupported device"):
            testing.time(simple_fn, (x,), device=torch.device("cuda"))


class TestEquations:
    """Tests for equation evaluation."""

    def test_eval_eq_simple(self):
        """Test simple arithmetic."""
        from ai_bench.utils import eval_eq

        assert eval_eq("2 + 3") == 5
        assert eval_eq("10 - 4") == 6
        assert eval_eq("6 * 7") == 42
        assert eval_eq("20 / 4") == 5

    def test_eval_eq_complex(self):
        """Test complex expressions."""
        from ai_bench.utils import eval_eq

        assert eval_eq("2 * 64 * 64 * 64") == 2 * 64 * 64 * 64
        assert eval_eq("(10 + 5) * 2") == 30
        assert eval_eq("2 ** 10") == 1024

    def test_eval_eq_negative(self):
        """Test negative numbers."""
        from ai_bench.utils import eval_eq

        assert eval_eq("-5") == -5
        assert eval_eq("-5 + 10") == 5

    def test_eval_eq_invalid(self):
        """Test invalid expressions raise errors."""
        from ai_bench.utils import eval_eq

        # SyntaxError for invalid Python syntax.
        with pytest.raises(SyntaxError):
            eval_eq("import os")

        # TypeError for unsupported operations (e.g., function calls).
        with pytest.raises(TypeError):
            eval_eq("abs(-5)")


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.fixture
    def integration_setup(self):
        """Set up a complete test environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create full directory structure.
            specs_dir = tmpdir / "problems" / "specs" / "KernelBench" / "level1"
            pytorch_dir = (
                tmpdir / "third_party" / "KernelBench" / "KernelBench" / "level1"
            )
            triton_dir = tmpdir / "backends" / "triton" / "KernelBench" / "level1"

            specs_dir.mkdir(parents=True)
            pytorch_dir.mkdir(parents=True)
            triton_dir.mkdir(parents=True)

            # Create spec.
            spec = """
inputs:
  A:
    shape: [M, K]
    dtype: float32
  B:
    shape: [K, N]
    dtype: float32
ci:
  - params: [A, B]
    dims:
      M: 8
      K: 8
      N: 8
bench-cpu:
  - params: [A, B]
    dims:
      M: 32
      K: 32
      N: 32
    flop: 2*M*N*K
    mem_bytes: "(M*K + K*N + M*N) * 4"
"""
            (specs_dir / "matmul.yaml").write_text(spec)

            # Create PyTorch kernel.
            pytorch_kernel = """
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return torch.matmul(A, B)
"""
            (pytorch_dir / "matmul.py").write_text(pytorch_kernel)

            # Create Triton kernel (mocked - same behavior).
            triton_kernel = '''
import torch

class Model(torch.nn.Module):
    """Mocked Triton kernel with same interface."""
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        # In real usage, this would be a Triton kernel.
        return torch.matmul(A, B)
'''
            (triton_dir / "matmul.py").write_text(triton_kernel)

            yield tmpdir

    def test_full_ci_pipeline_pytorch(self, integration_setup):
        """Test complete CI pipeline with PyTorch backend."""
        with mock.patch(
            "ai_bench.utils.finder.project_root", return_value=integration_setup
        ):
            kb_runner = runner.KernelBenchRunner(
                spec_type=ai_hc.SpecKey.V_CI,
                device=torch.device("cpu"),
                backend=ai_hc.Backend.PYTORCH,
            )
            kb_runner.run_kernels()

    def test_full_ci_pipeline_pytorch_compile(self, integration_setup):
        """Test complete CI pipeline with PyTorch compile backend."""
        with mock.patch(
            "ai_bench.utils.finder.project_root", return_value=integration_setup
        ):
            kb_runner = runner.KernelBenchRunner(
                spec_type=ai_hc.SpecKey.V_CI,
                device=torch.device("cpu"),
                backend=ai_hc.Backend.PYTORCH_COMPILE,
            )
            kb_runner.run_kernels()

    def test_full_ci_pipeline_triton(self, integration_setup):
        """Test complete CI pipeline with Triton backend."""
        with mock.patch(
            "ai_bench.utils.finder.project_root", return_value=integration_setup
        ):
            kb_runner = runner.KernelBenchRunner(
                spec_type=ai_hc.SpecKey.V_CI,
                device=torch.device("cpu"),
                backend=ai_hc.Backend.TRITON,
            )
            kb_runner.run_kernels()

    def test_full_bench_pipeline(self, integration_setup):
        """Test complete benchmark pipeline."""
        with mock.patch(
            "ai_bench.utils.finder.project_root", return_value=integration_setup
        ):
            kb_runner = runner.KernelBenchRunner(
                spec_type=ai_hc.SpecKey.V_BENCH_CPU,
                device=torch.device("cpu"),
                backend=ai_hc.Backend.PYTORCH,
            )
            kb_runner.run_kernels()

    def test_backend_produces_same_results(self, integration_setup):
        """Test that both backends produce same numerical results."""
        with mock.patch(
            "ai_bench.utils.finder.project_root", return_value=integration_setup
        ):
            # Load both kernels.
            pytorch_runner = runner.KernelBenchRunner(
                spec_type=ai_hc.SpecKey.V_CI,
                device=torch.device("cpu"),
                backend=ai_hc.Backend.PYTORCH,
            )
            triton_runner = runner.KernelBenchRunner(
                spec_type=ai_hc.SpecKey.V_CI,
                device=torch.device("cpu"),
                backend=ai_hc.Backend.TRITON,
            )

            pytorch_model_cls = pytorch_runner.load_model(
                integration_setup
                / "third_party"
                / "KernelBench"
                / "KernelBench"
                / "level1"
                / "matmul.py"
            )
            triton_model_cls = triton_runner.load_model(
                integration_setup
                / "backends"
                / "triton"
                / "KernelBench"
                / "level1"
                / "matmul.py"
            )

            pytorch_model = pytorch_model_cls()
            triton_model = triton_model_cls()

            # Create deterministic inputs.
            torch.manual_seed(42)
            A = torch.randn(8, 8)
            B = torch.randn(8, 8)

            pytorch_result = pytorch_model(A, B)
            triton_result = triton_model(A, B)

            assert torch.allclose(pytorch_result, triton_result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
