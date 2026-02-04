#!/usr/bin/env python3
"""CLI tool to compare kernel backends."""

import argparse
import sys

import torch

from ai_bench.harness import core as ai_hc
from ai_bench.harness import testing
from ai_bench.harness.runner.benchmark_compare import (
    benchmark_problem, print_comparison, print_comparison_brief,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compare kernel performance across backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare.py --problem level1/softmax --device xpu
  python compare.py --problem level1/softmax --device xpu --backends pytorch triton
  python compare.py --problem level1/softmax --device xpu --benchmark-method elapsed_time

CV Stability: ★★★ (<1%) | ★★ (1-5%) | ★ (5-10%) | ⚠ (>10%)
        """,
    )
    
    parser.add_argument("--problem", required=True, help="level/kernel_name")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "xpu"])
    parser.add_argument("--spec-type", default="bench-gpu", choices=["ci", "bench-cpu", "bench-gpu"])
    parser.add_argument("--backends", nargs="+", choices=["pytorch", "pytorch-compile", "triton"])
    parser.add_argument("--no-triton", action="store_true", help="Exclude Triton")
    
    bench_group = parser.add_argument_group("benchmarking options")
    bench_group.add_argument("--time-warmup", action="store_true", default=True)
    bench_group.add_argument("--no-time-warmup", action="store_true")
    bench_group.add_argument("--warmup", type=int)
    bench_group.add_argument("--rep", type=int)
    bench_group.add_argument("--no-clear-l2", action="store_true")
    
    parser.add_argument("--brief", action="store_true", help="Brief output")
    
    args = parser.parse_args()

    spec_map = {"ci": ai_hc.SpecKey.V_CI, "bench-cpu": ai_hc.SpecKey.V_BENCH_CPU, "bench-gpu": ai_hc.SpecKey.V_BENCH_GPU}
    backend_map = {"pytorch": ai_hc.Backend.PYTORCH, "pytorch-compile": ai_hc.Backend.PYTORCH_COMPILE, "triton": ai_hc.Backend.TRITON}

    if args.backends:
        backends = [backend_map[b] for b in args.backends]
    elif args.no_triton:
        backends = [ai_hc.Backend.PYTORCH, ai_hc.Backend.PYTORCH_COMPILE]
    else:
        backends = [ai_hc.Backend.PYTORCH, ai_hc.Backend.PYTORCH_COMPILE, ai_hc.Backend.TRITON]

    device = torch.device(args.device)

    try:
        results = benchmark_problem(
            problem=args.problem, device=device, spec_type=spec_map[args.spec_type],
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