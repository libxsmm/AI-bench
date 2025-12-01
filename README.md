# AI-bench: Unified AI Benchmarking Suite
[![Tests](https://github.com/libxsmm/AI-bench/actions/workflows/test.yml/badge.svg)](https://github.com/libxsmm/AI-bench/actions/workflows/test.yml)
[![Lint](https://github.com/libxsmm/AI-bench/actions/workflows/lint.yml/badge.svg)](https://github.com/libxsmm/AI-bench/actions/workflows/lint.yml)

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

## Linting
The project uses `pre-commit` to run various checks automatically.
All checks can be run using:
```bash
pre-commit run -a
```
