# AI-bench: Unified AI Benchmarking Suite

[![Tests](https://github.com/libxsmm/AI-bench/actions/workflows/test.yml/badge.svg)](https://github.com/libxsmm/AI-bench/actions/workflows/test.yml)
[![Lint](https://github.com/libxsmm/AI-bench/actions/workflows/lint.yml/badge.svg)](https://github.com/libxsmm/AI-bench/actions/workflows/lint.yml)
[![KernelBench Perf](https://github.com/libxsmm/AI-bench/actions/workflows/kernel_bench.yml/badge.svg)](https://github.com/libxsmm/AI-bench/actions/workflows/kernel_bench.yml)
![Status](https://img.shields.io/badge/status-beta-yellow)

A benchmarking framework for evaluating AI kernel implementations across multiple backends (PyTorch, Triton, Helion, MLIR) and devices (CPU, XPU, CUDA).

| | PyTorch | Triton | Helion | MLIR |
|:---:|:---:|:---:|:---:|:---:|
| **CPU** | ✅ | ❌ | ❌ | ✅ |
| **XPU** | ✅ | ✅ | ✅ | ❌ |
| **CUDA** | ✅ | ⚠️ | ⚠️ | ❌ |

✅ - Supported ⚠️ - Partially implemented ❌ - Unsupported

## Installation

The project is using [uv](https://docs.astral.sh/uv/) package manager.

`uv` can be [installed](https://docs.astral.sh/uv/getting-started/installation/) locally using:

```bash
pip install uv
```

The project can be installed with appropriate device and backend extensions using:

```bash
# CPU only
uv sync --extra cpu

# CPU + XPU
uv sync --extra xpu

# CPU + CUDA
uv sync --extra cuda

# CPU + MLIR backend
uv sync --extra cpu --extra mlir
```

## Usage

### Command Line Interface

After installation, the `ai-bench` command is available:

```bash
# Show help
ai-bench --help

# PyTorch on CPU (default)
ai-bench

# MLIR on CPU
ai-bench --mlir

# PyTorch on XPU
ai-bench --xpu

# PyTorch on CUDA GPU
ai-bench --cuda

# PyTorch compile on XPU
ai-bench --xpu --torch-compile

# Triton on XPU
ai-bench --xpu --triton

# Helion on XPU
ai-bench --xpu --helion

# Benchmark mode (with timing)
ai-bench --xpu --bench

# Log results to CSV
ai-bench --xpu --bench --csv results.csv --note "baseline run"

# Run a single kernel with a problem specification
ai-bench --kernel /path/to/kernel.py /path/to/spec.yaml

# Run a single kernel with a problem specification on XPU
ai-bench --kernel /path/to/kernel.py /path/to/spec.yaml --xpu
```

Using `ai-bench-compare` command, KernelBench performance can be compared across multiple backends:

```bash
# Show help
ai-bench-compare --help

# Comparison for the given problem on CPU (default)
ai-bench-compare --problem level1/1_Square_matrix_multiplication_

# Compare PyTorch and Triton backends on XPU
ai-bench-compare --problem level2/99_Matmul_GELU_Softmax --backend pytorch triton --xpu
```

Optionally, custom CLI autocompletion is available for certain scripts. It can be activated using:
```bash
activate-global-python-argcomplete --user
```

### As a Library

```python
import ai_bench
import torch

# Create a single kernel benchmark
kernel_runner = ai_bench.KernelRunner(
    spec_type=ai_bench.SpecKey.V_BENCH_CPU,
    device=torch.device("cpu"),
    backend=ai_bench.Backend.PYTORCH,
    flops_unit=ai_bench.FlopsUnit.TFLOPS,
    mem_bw_unit=ai_bench.MemBwUnit.GBS,
)
kernel_runner.run_kernel_spec("path/to/kernel.py", "path/to/spec.yaml")

# Configure paths if running outside project root
ai_bench.configure(
    specs_dir="/path/to/specs",
    kernels_dir="/path/to/kernels",
)

# Create KernelBench benchmark runner
kb_runner = ai_bench.KernelBenchRunner(
    spec_type=ai_bench.SpecKey.V_BENCH_GPU,
    device=torch.device("xpu"),
    backend=ai_bench.Backend.PYTORCH,
    flops_unit=ai_bench.FlopsUnit.TFLOPS,
    mem_bw_unit=ai_bench.MemBwUnit.GBS,
    csv_path="results.csv",
)
kb_runner.run_kernels()
```

### CSV Logging

Benchmark results can be logged to a CSV file using the `--csv` option:

```bash
# Log results to CSV
ai-bench --xpu --triton --bench --csv results.csv

# Add a note to identify the run
ai-bench --xpu --triton --bench --csv results.csv --note "BMG card test"
```

The CSV file includes the following columns:

- `kernel_name`: Name of the kernel
- `kernel_type`: Backend used (pytorch/triton)
- `problem_level`: KernelBench problem level
- `flops`: Number of floating-point operations
- `flops_val`: Computed FLOPS value
- `flops_unit`: FLOPS unit (GFLOPS/TFLOPS)
- `flops_note`: FLOPS measurement annotation (see 'Notes legend')
- `mem_bytes`: Number of memory bytes transferred - input reads + output writes
- `mem_bw_val`: Computed memory bandwidth value
- `mem_bw_unit`: Memory bandwidth unit (MB/s or GB/s)
- `mem_note`: Memory measurement annotation (see 'Notes legend')
- `time_us`: Execution time in microseconds
- `input_values`: Input dimensions as JSON
- `note`: User-provided note

Additionally, any environment variables prefixed with `AIBENCH_` are automatically captured and included in the CSV output. This is useful for recording system configuration:

```bash
# Set environment variables for tracking
export AIBENCH_CARD="BMG"
export AIBENCH_SYSTEM="TestRig1"
ai-bench --xpu --triton --bench --csv results.csv
```

Notes legend:

- `⚠️`: estimated value, use with caution

### Command Line Options

| Option | Description |
|--------|-------------|
| `--kernel KERNEL_PATH SPEC_PATH` | Run a kernel with a spec (default: KernelBench) |
| `--xpu` | Run on Intel XPU (default: CPU) |
| `--cuda` | Run on Nvidia GPU (default: CPU) |
| `--triton` | Use Triton backend (default: PyTorch eager) |
| `--torch-compile` | Use PyTorch compile mode (default: PyTorch eager) |
| `--helion` | Use Helion backend (default: PyTorch eager) |
| `--mlir` | Use MLIR backend (default: PyTorch eager) |
| `--bench` | Run benchmarks with timing (default: CI validation) |
| `--gflops` | Report GFLOPS (default: TFLOPS) |
| `--mbs` | Report MB/s (default: GB/s) |
| `--csv PATH` | Log results to specified CSV file |
| `--note TEXT` | Add a note to CSV output for identifying runs |
| `--specs-dir PATH` | Path to specs directory (CLI only) |
| `--kernels-dir PATH` | Path to kernels directory (CLI only) |
| `--triton-kernels-dir PATH` | Path to Triton kernels directory (CLI only) |
| `--helion-kernels-dir PATH` | Path to Helion kernels directory (CLI only) |
| `--mlir-kernels-dir PATH` | Path to MLIR kernels directory (CLI only) |
| `--env-file PATH` | Path to .env file (default: auto-detect) |
| `--no-env` | Disable loading .env config |

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
| `AIBENCH_HELION_KERNELS_DIR` | Path to Helion kernels directory |
| `AIBENCH_MLIR_KERNELS_DIR` | Path to MLIR kernels directory |
| `AIBENCH_MLIR_LIB_PATH` | Paths to MLIR shared libraries (colon separated) |
| `AIBENCH_MLIR_DUMP` | Dump imported MLIR IR |
| `AIBENCH_MLIR_DUMP_OBJ` | Dump jitted MLIR to an object file |

## License

MIT License - see [LICENSE](LICENSE) for details.
