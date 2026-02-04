"""Benchmark comparison utilities for ai_bench."""

import torch
from dataclasses import dataclass
from typing import Optional, List

from ai_bench.harness import core as ai_hc
from ai_bench.harness import testing
from ai_bench.harness.runner import KernelBenchRunner
from .kernel_runner import KernelStats



def benchmark_problem(
    problem: str,
    device: torch.device,
    spec_type: ai_hc.SpecKey = ai_hc.SpecKey.V_BENCH_GPU,
    backends: Optional[List[ai_hc.Backend]] = None,
) -> dict:
    """Benchmark a specific problem across multiple backends."""
    if backends is None:
        backends = [ai_hc.Backend.PYTORCH, ai_hc.Backend.PYTORCH_COMPILE, ai_hc.Backend.TRITON]
    print(f"backends: {backends}")
    parts = problem.strip("/").split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid problem format: '{problem}'. Expected 'level/kernel_name'")

    level, kernel_name = parts
    
    results = {
        "problem": problem, "device": str(device),
        "backends": {}, "spec_flop": None, "spec_mem_bytes": None,
    }

    for backend in backends:
        print(f"backend: {backend}")
        try:
            runner = KernelBenchRunner(
                spec_type=spec_type, device=device, backend=backend
            )

            spec_path = runner.specs / level / f"{kernel_name}.yaml"
            if not spec_path.exists():
                raise FileNotFoundError(f"Spec not found: {spec_path}")

            kernel_path = runner.kernels / level / f"{kernel_name}.py"
            print(f"kernel path: {kernel_path}")
            # Run the kernel in all compatible variants.
            run_stats: list[KernelStats] | None = runner.run_kernel_spec(
                kernel_path, spec_path
            )

            # Continue if desired configuration is not available or
            # if there is nothing extra to report.
            if not run_stats:
                print(f"Warnning: recieved no results for {backend} backend.")
                continue

            results["backends"][str(backend)] = run_stats[0] # assuming a single variant


        except FileNotFoundError as e:
            print(f"error: {e}")
            results["backends"][str(backend)] = None

    pytorch_res = results["backends"].get(str(ai_hc.Backend.PYTORCH))

    if pytorch_res:
        baseline_time = pytorch_res.meas_us
        results["speedups"] = {b: baseline_time / r.meas_us for b, r in results["backends"].items() 
                              if r}

    return results


def _fmt_sci(x):
    if x is None:
        return "N/A"
    if x == 0:
        return "0"
    return f"{x:.3e}" if abs(x) >= 1e9 or abs(x) < 1e-2 else f"{x:.3f}"


def _fmt_cv(cv):
    if cv is None:
        return "N/A"
    cv_pct = cv * 100
    indicator = "★★★" if cv_pct < 1 else "★★" if cv_pct < 5 else "★" if cv_pct < 10 else "⚠"
    return f"{cv_pct:.2f}% {indicator}"


def print_comparison(results: dict):
    """Pretty print comparison results."""
    print(f"\n{'='*80}")
    print(f"Problem: {results['problem']}")
    print(f"Device:  {results['device']}")
    print(f"{'='*80}")

    spec_flop, spec_mem = results.get("spec_flop"), results.get("spec_mem_bytes")
    if spec_flop or spec_mem:
        parts = []
        if spec_flop:
            parts.append(f"FLOPs={_fmt_sci(spec_flop)}")
        if spec_mem:
            parts.append(f"Bytes={_fmt_sci(spec_mem)}")
        print("Spec: " + "  ".join(parts))

    speedups = results.get("speedups", {})
    have_bw = any(r and r.mem_bw for r in results["backends"].values())

    flops_unit_list = [r.flops_unit for r in results["backends"].values()]
    flops_unit = next( f for f in flops_unit_list if f is not None)
    if have_bw:
        # print(f"\n{'Backend':<18} {'Time (μs)':>12} {'TFLOPS':>8} {'GB/s':>8} {'CV':>12} {'Speedup':>10}")
        print(f"\n{'Backend':<18} {'Time (μs)':>12} {flops_unit:>8} {'GB/s':>8} {'Speedup':>10}")
        print("-" * 80)
    else:
        # print(f"\n{'Backend':<18} {'Time (μs)':>12} {'TFLOPS':>10} {'CV':>12} {'Speedup':>10}")
        print(f"\n{'Backend':<18} {'Time (μs)':>12} {flops_unit:>10} {'Speedup':>10}")
        print("-" * 70)

    for backend, res in results["backends"].items():
        if not res:
            print(f"{backend:<18} {'ERROR: no results'}")
            continue
        speedup_str = f"{speedups.get(backend, 1.0):.2f}x"
        flops_str = f"{res.flops:.2f}" if res.flops else "N/A"
        # cv_str = _fmt_cv(res.cv)
        if have_bw:
            gbs_str = f"{res.mem_bw:.1f}" if res.mem_bw else "N/A"
            # print(f"{backend:<18} {res.meas_us:>12.2f} {flops_str:>8} {gbs_str:>8} {cv_str:>12} {speedup_str:>10}")
            print(f"{backend:<18} {res.meas_us:>12.2f} {flops_str:>8} {gbs_str:>8} {speedup_str:>10}")
        else:
            # print(f"{backend:<18} {res.meas_us:>12.2f} {flops_str:>10} {cv_str:>12} {speedup_str:>10}")
            print(f"{backend:<18} {res.meas_us:>12.2f} {flops_str:>10} {speedup_str:>10}")

    print("-" * (80 if have_bw else 70))


    valid = [(b, r) for b, r in results["backends"].items() if r]
    if valid:
        fastest = min(valid, key=lambda x: x[1].meas_us)
        slowest = max(valid, key=lambda x: x[1].meas_us)
        print(f"\nFastest: {fastest[0]} ({fastest[1].meas_us:.2f} μs)")
        if len(valid) > 1:
            print(f"Max speedup: {slowest[1].meas_us / fastest[1].meas_us:.2f}x ({fastest[0]} vs {slowest[0]})")
        # most_stable = min(valid, key=lambda x: x[1].cv or float('inf'))
        # if most_stable[1].cv is not None:
        #     print(f"\nMost stable: {most_stable[0]} (CV: {_fmt_cv(most_stable[1].cv)})")

    print(f"{'='*(80 if have_bw else 70)}\n")


def print_comparison_brief(results: dict):
    """Print brief one-line comparison."""
    fastest, fastest_time = None, float('inf')
    for backend, res in results["backends"].items():
        if not res.error and res.meas_us > 0 and res.meas_us < fastest_time:
            fastest, fastest_time = backend, res.meas_us
    speedups = results.get("speedups", {})
    speedup_strs = [f"{b}:{s:.2f}x" for b, s in speedups.items()]
    print(f"{results['problem']}: fastest={fastest} ({fastest_time:.2f}μs) | {' '.join(speedup_strs)}")