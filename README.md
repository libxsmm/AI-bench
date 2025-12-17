# AI-bench: Unified AI Benchmarking Suite

[![Tests](https://github.com/libxsmm/AI-bench/actions/workflows/test.yml/badge.svg)](https://github.com/libxsmm/AI-bench/actions/workflows/test.yml)
[![Lint](https://github.com/libxsmm/AI-bench/actions/workflows/lint.yml/badge.svg)](https://github.com/libxsmm/AI-bench/actions/workflows/lint.yml)

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
| `--bench` | Run benchmarks with timing (default: CI validation) |
| `--csv PATH` | Log results to specified CSV file |
| `--note TEXT` | Add a note to CSV output for identifying runs |

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
* `AIBENCH_LOG=INFO|DEBUG|...` - globally overrides logging level, available levels directly correspond to Python's `logging` module
