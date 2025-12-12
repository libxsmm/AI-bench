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

# Triton on XPU
python infra/scripts/run_kernel_bench.py --xpu --triton

# Benchmark mode (with timing)
python infra/scripts/run_kernel_bench.py --xpu --bench

# Triton benchmark on XPU
python infra/scripts/run_kernel_bench.py --xpu --triton --bench
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