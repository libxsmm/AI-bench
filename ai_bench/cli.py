"""Command-line interface for ai_bench."""

import argparse
from pathlib import Path
import sys

import torch

from ai_bench import __version__
from ai_bench.harness import core
from ai_bench.harness import runner
from ai_bench.utils import finder


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog="ai-bench",
        description="AI kernel benchmarking harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run CI validation on CPU
  ai-bench

  # Run benchmarks on XPU with PyTorch
  ai-bench --xpu --bench

  # Run with PyTorch compile
  ai-bench --xpu --bench --torch-compile

  # Run with Triton backend
  ai-bench --xpu --bench --triton

  # Save results to CSV
  ai-bench --xpu --bench --csv results.csv --note "baseline run"

  # Use custom paths (for library-style usage)
  ai-bench --specs-dir /path/to/specs --kernels-dir /path/to/kernels

  # Use specific .env file
  ai-bench --env-file /path/to/.env --xpu --bench

  # Disable .env loading
  ai-bench --no-env --xpu --bench

Environment file (.env) example:
  AIBENCH_SPECS_DIR=/path/to/specs
  AIBENCH_KERNELS_DIR=/path/to/kernels
  AIBENCH_CARD=BMG
  AIBENCH_SYSTEM=TestRig1
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # Environment file options
    env_group = parser.add_argument_group("environment file options")
    env_exclusive = env_group.add_mutually_exclusive_group()
    env_exclusive.add_argument(
        "--env-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to .env file (default: auto-detect in cwd or project root)",
    )
    env_exclusive.add_argument(
        "--no-env",
        action="store_true",
        default=False,
        help="Disable loading .env file",
    )

    # Path configuration
    path_group = parser.add_argument_group("path configuration")
    path_group.add_argument(
        "--specs-dir",
        type=Path,
        default=None,
        help="Path to specs directory (default: auto-detect or AIBENCH_SPECS_DIR)",
    )
    path_group.add_argument(
        "--kernels-dir",
        type=Path,
        default=None,
        help="Path to PyTorch kernels directory (default: auto-detect or AIBENCH_KERNELS_DIR)",
    )
    path_group.add_argument(
        "--triton-kernels-dir",
        type=Path,
        default=None,
        help="Path to Triton kernels directory (default: auto-detect or AIBENCH_TRITON_KERNELS_DIR)",
    )

    # Device options
    device_group = parser.add_argument_group("device options")
    device_group.add_argument(
        "--xpu",
        action="store_true",
        default=False,
        help="Run on Intel XPU (default: CPU)",
    )
    device_group.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Run on CUDA GPU (default: CPU)",
    )

    # Backend options
    backend_group = parser.add_argument_group("backend options")
    backend_exclusive = backend_group.add_mutually_exclusive_group()
    backend_exclusive.add_argument(
        "--triton",
        action="store_true",
        default=False,
        help="Use Triton backend",
    )
    backend_exclusive.add_argument(
        "--torch-compile",
        action="store_true",
        default=False,
        help="Use PyTorch compile mode",
    )

    # Run mode
    mode_group = parser.add_argument_group("run mode")
    mode_group.add_argument(
        "--bench",
        action="store_true",
        default=False,
        help="Run full benchmarks (default: CI validation only)",
    )

    # Output options
    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "--gflops",
        action="store_true",
        default=False,
        help="Report GFLOPS (default: TFLOPS)",
    )
    output_group.add_argument(
        "--mbs",
        action="store_true",
        default=False,
        help="Report MB/s (default: GB/s)",
    )

    # CSV logging
    csv_group = parser.add_argument_group("CSV logging")
    csv_group.add_argument(
        "--csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to CSV file for logging results",
    )
    csv_group.add_argument(
        "--note",
        type=str,
        default="",
        help="Note to include in CSV output",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for CLI.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Load .env file (unless disabled)
    if not args.no_env:
        if args.env_file:
            if not finder.load_env(args.env_file):
                print(f"Warning: .env file not found: {args.env_file}", file=sys.stderr)
        else:
            finder.load_env()  # Auto-detect

    # Configure paths if provided
    if args.specs_dir or args.kernels_dir or args.triton_kernels_dir:
        finder.configure(
            specs_dir=args.specs_dir,
            kernels_dir=args.kernels_dir,
            triton_kernels_dir=args.triton_kernels_dir,
        )

    # Determine device
    if args.xpu:
        device = torch.device("xpu")
    elif args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Determine backend
    if args.triton:
        backend = core.Backend.TRITON
    elif args.torch_compile:
        backend = core.Backend.PYTORCH_COMPILE
    else:
        backend = core.Backend.PYTORCH

    # Determine spec type
    if args.bench:
        if device.type == "cpu":
            spec_type = core.SpecKey.V_BENCH_CPU
        else:
            spec_type = core.SpecKey.V_BENCH_GPU
    else:
        spec_type = core.SpecKey.V_CI

    # Determine units
    flops_unit = runner.FlopsUnit.GFLOPS if args.gflops else runner.FlopsUnit.TFLOPS
    mem_bw_unit = runner.MemBwUnit.MBS if args.mbs else runner.MemBwUnit.GBS

    try:
        kb_runner = runner.KernelBenchRunner(
            spec_type=spec_type,
            device=device,
            backend=backend,
            flops_unit=flops_unit,
            mem_bw_unit=mem_bw_unit,
            csv_path=args.csv,
            note=args.note,
        )
        kb_runner.run_kernels()
        return 0

    except finder.ConfigurationError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
