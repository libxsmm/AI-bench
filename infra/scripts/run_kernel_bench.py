#!/usr/bin/env python
"""Run KernelBench problems."""

import argparse
from pathlib import Path

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
        if args.torch_compile:
            backend = core.Backend.PYTORCH_COMPILE
        else:
            backend = core.Backend.PYTORCH

    # Determine spec type.
    spec_type = core.SpecKey.V_CI
    if args.bench:
        if device.type == "xpu":
            spec_type = core.SpecKey.V_BENCH_GPU
        else:
            spec_type = core.SpecKey.V_BENCH_CPU

    if args.gflops:
        flops_unit = runner.FlopsUnit.GFLOPS
    else:
        flops_unit = runner.FlopsUnit.TFLOPS

    if args.mbs:
        mem_bw_unit = runner.MemBwUnit.MBS
    else:
        mem_bw_unit = runner.MemBwUnit.GBS

    kb_runner = runner.KernelBenchRunner(
        spec_type=spec_type,
        device=device,
        backend=backend,
        flops_unit=flops_unit,
        mem_bw_unit=mem_bw_unit,
        csv_path=args.csv,
        note=args.note,
        sweep_config_path=args.sweep,
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
    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument(
        "--triton",
        action="store_true",
        default=False,
        help="Use Triton backend",
    )
    backend_group.add_argument(
        "--torch-compile",
        action="store_true",
        default=False,
        help="Use PyTorch compile mode",
    )

    # Run mode.
    parser.add_argument(
        "--bench",
        action="store_true",
        default=False,
        help="Benchmark execution (default: CI validation)",
    )

    # Stats options.
    parser.add_argument(
        "--gflops",
        action="store_true",
        default=False,
        help="Report GFLOPS (default: TFLOPS)",
    )
    parser.add_argument(
        "--mbs",
        action="store_true",
        default=False,
        help="Report MB/s (default: GB/s)",
    )

    # Sweep config.
    parser.add_argument(
        "--sweep",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to sweep config YAML for parameter sweeps",
    )

    # CSV logging options.
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to CSV file for logging results",
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Optional note to include in CSV output",
    )

    args = parser.parse_args()
    main(args)
