"""AI kernel benchmarking harness.

Example usage as library:
    >>> import ai_bench
    >>> import torch
    >>>
    >>> # Load configuration from .env file (optional)
    >>> ai_bench.load_env()
    >>>
    >>> # Or configure paths explicitly
    >>> ai_bench.configure(
    ...     specs_dir="/path/to/specs",
    ...     kernels_dir="/path/to/kernels",
    ... )
    >>>
    >>> # Create and run benchmark
    >>> runner = ai_bench.KernelBenchRunner(
    ...     spec_type=ai_bench.SpecKey.V_BENCH_GPU,
    ...     device=torch.device("xpu"),
    ...     backend=ai_bench.Backend.PYTORCH,
    ... )
    >>> runner.run_kernels()

Example usage as CLI:
    $ ai-bench --xpu --bench --csv results.csv

Example .env file:
    AIBENCH_SPECS_DIR=/path/to/specs
    AIBENCH_KERNELS_DIR=/path/to/kernels
    AIBENCH_CARD=BMG
"""

__version__ = "0.1.0b1"

# Core types and enums
from ai_bench.harness.core import Backend
from ai_bench.harness.core import InitKey
from ai_bench.harness.core import InKey
from ai_bench.harness.core import SpecKey
from ai_bench.harness.core import VKey

# Core functions
from ai_bench.harness.core import get_flop
from ai_bench.harness.core import get_inits
from ai_bench.harness.core import get_inputs
from ai_bench.harness.core import get_mem_bytes
from ai_bench.harness.core import get_torch_dtype
from ai_bench.harness.core import get_variant_torch_dtype
from ai_bench.harness.core import input_shape
from ai_bench.harness.core import input_torch_dtype

# Runner
from ai_bench.harness.runner import FlopsUnit
from ai_bench.harness.runner import KernelBenchRunner
from ai_bench.harness.runner import MemBwUnit

# Timing utilities
from ai_bench.harness.testing import time
from ai_bench.harness.testing import time_cpu
from ai_bench.harness.testing import time_xpu

# Configuration
from ai_bench.utils.finder import ConfigurationError
from ai_bench.utils.finder import configure
from ai_bench.utils.finder import is_env_loaded
from ai_bench.utils.finder import load_env
from ai_bench.utils.finder import reset_configuration

__all__ = [
    "Backend",
    "ConfigurationError",
    "FlopsUnit",
    "InKey",
    "InitKey",
    "KernelBenchRunner",
    "MemBwUnit",
    "SpecKey",
    "VKey",
    "__version__",
    "configure",
    "get_flop",
    "get_inits",
    "get_inputs",
    "get_mem_bytes",
    "get_torch_dtype",
    "get_variant_torch_dtype",
    "input_shape",
    "input_torch_dtype",
    "is_env_loaded",
    "load_env",
    "reset_configuration",
    "time",
    "time_cpu",
    "time_xpu",
]
