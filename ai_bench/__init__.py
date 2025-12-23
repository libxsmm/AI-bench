"""AI kernel benchmarking harness for PyTorch and Triton.

Example usage as library:
    >>> import ai_bench
    >>> import torch
    >>>
    >>> # Configure paths (required when used as library)
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
"""

__version__ = "0.1.0"

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

# Configuration
from ai_bench.utils.finder import ConfigurationError
from ai_bench.utils.finder import configure
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
    "reset_configuration",
    "time",
    "time_cpu",
]
