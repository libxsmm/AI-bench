import argparse

import torch

from ai_bench.harness.core import SpecKey
from ai_bench.harness.runner.benchmark_compare import benchmark_problem
from ai_bench.harness.runner.benchmark_compare import print_comparison


def main():
    parser = argparse.ArgumentParser(description="Compare kernel backends")
    parser.add_argument(
        "--problem",
        required=True,
        help="Problem path: level/kernel_name (e.g., level1/softmax)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use (cpu, cuda, xpu)",
    )
    parser.add_argument(
        "--spec-type",
        default="bench-gpu",
        choices=["ci", "bench-cpu", "bench-gpu"],
        help="Spec variant to use",
    )

    args = parser.parse_args()

    spec_map = {
        "ci": SpecKey.V_CI,
        "bench-cpu": SpecKey.V_BENCH_CPU,
        "bench-gpu": SpecKey.V_BENCH_GPU,
    }

    device = torch.device(args.device)
    results = benchmark_problem(
        problem=args.problem,
        device=device,
        spec_type=spec_map[args.spec_type],
    )
    print_comparison(results)


if __name__ == "__main__":
    main()
