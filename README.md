# AI-bench: Unified AI Benchmarking Suite

[![Tests](https://github.com/libxsmm/AI-bench/actions/workflows/test.yml/badge.svg)](https://github.com/libxsmm/AI-bench/actions/workflows/test.yml)
[![Lint](https://github.com/libxsmm/AI-bench/actions/workflows/lint.yml/badge.svg)](https://github.com/libxsmm/AI-bench/actions/workflows/lint.yml)
[![KernelBench Perf](https://github.com/libxsmm/AI-bench/actions/workflows/kernel_bench.yml/badge.svg)](https://github.com/libxsmm/AI-bench/actions/workflows/kernel_bench.yml)
![Status](https://img.shields.io/badge/status-beta-yellow)

A benchmarking framework for evaluating AI kernel implementations across multiple backends (PyTorch, Triton) and devices (CPU, XPU).

## Installation

The project is using [uv](https://docs.astral.sh/uv/) package manager.

`uv` can be [installed](https://docs.astral.sh/uv/getting-started/installation/) locally using:

```bash
pip install uv
```

The project can be installed with appropriate device-specific extensions using:

```bash
# CPU only
uv sync --extra cpu

# CPU + XPU
uv sync --extra xpu
```

## Usage

### Command Line Interface

After installation, the `ai-bench` command is available:

```bash
# Show help
ai-bench --help

# PyTorch on CPU (default)
ai-bench

# PyTorch on XPU
ai-bench --xpu

# PyTorch compile on XPU
ai-bench --xpu --torch-compile

# Triton on XPU
ai-bench --xpu --triton

# Benchmark mode (with timing)
ai-bench --xpu --bench

# Log results to CSV
ai-bench --xpu --bench --csv results.csv --note "baseline run"
```

### Using Scripts Directly

Run KernelBench problems with different backends and devices:

```bash
# PyTorch on CPU (default)
python infra/scripts/run_kernel_bench.py

# PyTorch on XPU
python infra/scripts/run_kernel_bench.py --xpu

# PyTorch compile on XPU
python infra/scripts/run_kernel_bench.py --xpu --torch-compile

# Triton on XPU
python infra/scripts/run_kernel_bench.py --xpu --triton

# Benchmark mode (with timing)
python infra/scripts/run_kernel_bench.py --xpu --bench

# Triton benchmark on XPU
python infra/scripts/run_kernel_bench.py --xpu --triton --bench
```

### As a Library

```python
import ai_bench
import torch

# Configure paths if running outside project root
ai_bench.configure(
    specs_dir="/path/to/specs",
    kernels_dir="/path/to/kernels",
)

# Create benchmark runner
runner = ai_bench.KernelBenchRunner(
    spec_type=ai_bench.SpecKey.V_BENCH_GPU,
    device=torch.device("xpu"),
    backend=ai_bench.Backend.PYTORCH,
    flops_unit=ai_bench.FlopsUnit.TFLOPS,
    mem_bw_unit=ai_bench.MemBwUnit.GBS,
    csv_path="results.csv",
)
runner.run_kernels()
```

### CSV Logging

Benchmark results can be logged to a CSV file using the `--csv` option:

```bash
# Log results to CSV
python infra/scripts/run_kernel_bench.py --xpu --triton --bench --csv results.csv

# Add a note to identify the run
python infra/scripts/run_kernel_bench.py --xpu --triton --bench --csv results.csv --note "BMG card test"
```

The CSV file includes the following columns:

- `kernel_name`: Name of the kernel
- `kernel_type`: Backend used (pytorch/triton)
- `problem_level`: KernelBench problem level
- `flops`: Number of floating-point operations
- `flops_val`: Computed FLOPS value
- `flops_unit`: FLOPS unit (GFLOPS/TFLOPS)
- `mem_bytes`: Number of bytes transferred - input reads + output writes
- `mem_bw_val`: Computed memory bandwidth value
- `mem_bw_unit`: Memory bandwidth unit (MB/s or GB/s)
- `time_us`: Execution time in microseconds
- `input_values`: Input dimensions as JSON
- `note`: User-provided note

Additionally, any environment variables prefixed with `AIBENCH_` are automatically captured and included in the CSV output. This is useful for recording system configuration:

```bash
# Set environment variables for tracking
export AIBENCH_CARD="BMG"
export AIBENCH_SYSTEM="TestRig1"
python infra/scripts/run_kernel_bench.py --xpu --triton --bench --csv results.csv
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--xpu` | Run on Intel XPU (default: CPU) |
| `--triton` | Use Triton backend (default: PyTorch) |
| `--torch-compile` | Use PyTorch compile mode |
| `--bench` | Run benchmarks with timing (default: CI validation) |
| `--gflops` | Report GFLOPS (default: TFLOPS) |
| `--mbs` | Report MB/s (default: GB/s) |
| `--csv PATH` | Log results to specified CSV file |
| `--note TEXT` | Add a note to CSV output for identifying runs |
| `--sweep PATH` | Use sweep config for parameter sweeps |
| `--specs-dir PATH` | Path to specs directory (CLI only) |
| `--kernels-dir PATH` | Path to kernels directory (CLI only) |

## Parameter Sweeps

For benchmarking across multiple input shapes, use sweep configs:

```bash
# Run with parameter sweep
ai-bench --xpu --bench --sweep sweeps/level1.yaml --csv results.csv
```

### Sweep Config Format

```yaml
# sweeps/level1.yaml
19_ReLU:
  batch_size: [1024, 2048, 4096]
  dim: [49152, 98304, 196608, 294912]
  # Generates: 3 × 4 = 12 variants

1_Conv2D_ReLU_BiasAdd:
  batch_size: [64, 128, 256]
  in_channels: [32, 64]
  height: [64, 128]
  _linked:
    width: height    # Square images (width = height)
  # Generates: 3 × 2 × 2 = 12 variants

2_Standard_matrix_multiplication_:
  m: [1024, 2048, 4096]
  k: [4096, 8192, 12288]
  _fixed:
    n: 3072          # Fixed parameter
```

### Special Keys

| Key | Description |
|-----|-------------|
| `_linked` | Set parameter equal to another (e.g., `width: height`) |
| `_fixed` | Set constant value (not swept) |

### Preview Sweep Config

```bash
# Show summary
python -m ai_bench.utils.sweep_preview sweeps/level1.yaml

# Show specific kernel variants
python -m ai_bench.utils.sweep_preview sweeps/level1.yaml --kernel 19_ReLU

# Export all variants to YAML
python -m ai_bench.utils.sweep_preview sweeps/level1.yaml --output expanded.yaml
```

## Testing

Run tests with pytest:

```bash
pytest -v
```

## Linting

The project uses `pre-commit` to run various checks automatically.

All checks can be run using:

```bash
pre-commit run -a
```

## Config variables

Environment variables used for project configuration:

| Variable | Description |
|----------|-------------|
| `AIBENCH_LOG=INFO\|DEBUG\|...` | Globally overrides logging level |
| `AIBENCH_SPECS_DIR` | Path to specs directory |
| `AIBENCH_KERNELS_DIR` | Path to PyTorch kernels directory |
| `AIBENCH_TRITON_KERNELS_DIR` | Path to Triton kernels directory |

## License

MIT License - see [LICENSE](LICENSE) for details.