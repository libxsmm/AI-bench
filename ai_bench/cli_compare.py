#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

"""CLI tool to compare kernel backends."""

import argparse
import os
from pathlib import Path
import sys

import argcomplete
import torch

from ai_bench.harness import core as ai_hc
from ai_bench.harness.runner.benchmark_compare import benchmark_problem
from ai_bench.harness.runner.benchmark_compare import print_comparison
from ai_bench.harness.runner.benchmark_compare import print_comparison_brief
from ai_bench.utils import finder


def get_problem_choices() -> list[str]:
    """
    Generate available choices for KernelBench problems.
    Returns:
        List of CLI problem choices.
    """
    kb_specs = finder.specs() / "KernelBench"
    choices = []
    for level_dir in sorted(
        Path(entry) for entry in os.scandir(kb_specs) if entry.is_dir()
    ):
        for file in sorted(Path(file) for file in os.listdir(level_dir)):
            spec = f"{level_dir.name}/{file.stem}"
            choices.append(spec)
    return choices


def main():
    parser = argparse.ArgumentParser(
        description="Compare kernel performance across backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comparison on CPU
  ai-bench-compare --problem level1/19_ReLU

  # Run comparison on XPU
  ai-bench-compare --problem level1/19_ReLU --xpu

  # Run comparison on CUDA GPU
  ai-bench-compare --problem level1/19_ReLU --cuda

  # Run comparison between PyTorch and Triton
  ai-bench-compare --problem level2/99_Matmul_GELU_Softmax --xpu --backends pytorch triton

  # Override tolerances from command line (takes priority over spec values)
  ai-bench-compare --problem level1/19_ReLU --xpu --rtol 1e-3 --atol 1e-6
        """,
    )

    problem_choices = get_problem_choices()
    parser.add_argument(
        "--problem",
        metavar="PROBLEM",
        required=True,
        help="KernelBench problem: level[1-4]/kernel_name",
        choices=problem_choices,
    )
    parser.add_argument(
        "--xpu", action="store_true", help="Run on Intel XPU (default: CPU)"
    )
    parser.add_argument(
        "--cuda", action="store_true", help="Run on cuda device (default: CPU)"
    )
    parser.add_argument("--ci", action="store_true", help="Run with CI spec")
    backends_choices = [str(val) for val in ai_hc.Backend]
    parser.add_argument(
        "--backends",
        metavar="BACKENDS",
        nargs="+",
        help=f"Available backends: {backends_choices}",
        choices=backends_choices,
    )

    bench_group = parser.add_argument_group("kernel validation options")
    bench_group.add_argument(
        "--rtol",
        default=None,
        type=float,
        help="Override relative tolerance (default: per-problem spec value, fallback 1e-2)",
    )
    bench_group.add_argument(
        "--atol",
        default=None,
        type=float,
        help="Override absolute tolerance (default: per-problem spec value, fallback 1e-5)",
    )

    # TODO: Enable and propagate config.
    # bench_group = parser.add_argument_group("benchmarking options")
    # bench_group.add_argument("--time-warmup", action="store_true", default=True)
    # bench_group.add_argument("--no-time-warmup", action="store_true")
    # bench_group.add_argument("--warmup", type=int)
    # bench_group.add_argument("--rep", type=int)
    # bench_group.add_argument("--no-clear-l2", action="store_true")

    parser.add_argument("--brief", action="store_true", help="Brief output")

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.backends:
        backends = [ai_hc.Backend(b) for b in args.backends]
    else:
        backends = [ai_hc.Backend.PYTORCH, ai_hc.Backend.PYTORCH_COMPILE]

    # Determine device
    if args.xpu:
        device = torch.device("xpu")
    elif args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Determine spec type
    if args.ci:
        spec_type = ai_hc.SpecKey.V_CI
    else:
        spec_type = (
            ai_hc.SpecKey.V_BENCH_GPU
            if args.xpu or args.cuda
            else ai_hc.SpecKey.V_BENCH_CPU
        )

    try:
        results = benchmark_problem(
            problem=args.problem,
            device=device,
            spec_type=spec_type,
            rtol=args.rtol,
            atol=args.atol,
            backends=backends,
        )
        (print_comparison_brief if args.brief else print_comparison)(results)
        return 0
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
