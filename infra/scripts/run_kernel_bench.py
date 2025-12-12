import argparse

import torch

from ai_bench.harness import core
from ai_bench.harness import runner


def main(args):
    # Determine device.
    if args.xpu:
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")

    # Determine backend.
    if args.triton:
        backend = core.Backend.TRITON
    else:
        backend = core.Backend.PYTORCH

    # Determine spec type.
    spec_type = core.SpecKey.V_CI
    if args.bench:
        if device.type == "xpu":
            spec_type = core.SpecKey.V_BENCH_GPU
        else:
            spec_type = core.SpecKey.V_BENCH_CPU

    kb_runner = runner.KernelBenchRunner(
        spec_type=spec_type,
        device=device,
        backend=backend,
    )
    kb_runner.run_kernels()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KernelBench problems")

    # Device options.
    parser.add_argument(
        "--xpu",
        action="store_true",
        default=False,
        help="Run on Intel GPU",
    )

    # Backend options.
    parser.add_argument(
        "--triton",
        action="store_true",
        default=False,
        help="Use Triton backend (default: PyTorch)",
    )

    # Run mode.
    parser.add_argument(
        "--bench",
        action="store_true",
        default=False,
        help="Benchmark execution (default: CI validation)",
    )

    args = parser.parse_args()
    main(args)
