from dataclasses import dataclass
from typing import Optional

import torch

from ai_bench.harness import core as ai_hc
from ai_bench.harness.runner import KernelBenchRunner


@dataclass
class BenchmarkResult:
    backend: str
    time_us: float
    flop: Optional[float] = None
    tflops: Optional[float] = None
    error: Optional[str] = None


def benchmark_problem(
    problem: str,
    device: torch.device,
    spec_type: ai_hc.SpecKey = ai_hc.SpecKey.V_BENCH_GPU,
) -> dict:
    """
    Benchmark a specific problem across PyTorch and Triton backends.

    Args:
        problem: Problem path in format "level/kernel_name" (e.g., "level1/softmax")
        device: Device to run on
        spec_type: Spec variant to use

    Returns:
        Dict with comparison results
    """
    parts = problem.strip("/").split("/")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid problem format: '{problem}'. Expected 'level/kernel_name'"
        )

    level, kernel_name = parts
    results = {"problem": problem, "device": str(device), "backends": {}}

    for backend in [ai_hc.Backend.PYTORCH, ai_hc.Backend.TRITON]:
        try:
            runner = KernelBenchRunner(
                spec_type=spec_type,
                device=device,
                backend=backend,
            )
            result = _run_single_kernel(runner, level, kernel_name)
            results["backends"][str(backend)] = result
        except FileNotFoundError as e:
            results["backends"][str(backend)] = BenchmarkResult(
                backend=str(backend), time_us=0, error=str(e)
            )

    # Add speedup comparison if both succeeded
    pytorch_res = results["backends"].get("pytorch")
    triton_res = results["backends"].get("triton")

    if pytorch_res and triton_res and not pytorch_res.error and not triton_res.error:
        results["speedup"] = pytorch_res.time_us / triton_res.time_us

    return results


def _run_single_kernel(
    runner: KernelBenchRunner, level: str, kernel_name: str
) -> BenchmarkResult:
    """Run a single kernel and return results."""
    import yaml

    from ai_bench.harness import testing

    spec_path = runner.specs / level / f"{kernel_name}.yaml"
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec not found: {spec_path}")

    with open(spec_path) as f:
        spec = yaml.safe_load(f)

    if runner.spec_type not in spec:
        raise FileNotFoundError(f"Spec type '{runner.spec_type}' not available")

    kernel_path = runner.kernels / level / f"{kernel_name}.py"
    model_obj = runner.load_model(kernel_path)
    if not model_obj:
        raise FileNotFoundError(f"Kernel not found: {kernel_path}")

    # Use first variant for comparison
    variant = spec[runner.spec_type][0]
    inputs = spec[ai_hc.SpecKey.INS]
    inits = spec.get(ai_hc.SpecKey.INITS, [])

    model_inits = ai_hc.get_inits(variant, inits)
    model_dtype = ai_hc.get_variant_torch_dtype(variant)
    model = model_obj(*model_inits).to(runner.device, dtype=model_dtype)
    args = ai_hc.get_inputs(variant, inputs, device=runner.device)

    time_us = testing.time(
        fn=model.forward,
        args=args,
        warmup=runner.warmup,
        rep=runner.rep,
        device=runner.device,
    )

    flop = ai_hc.get_flop(variant)
    tflops = flop / time_us / 1e6 if flop else None

    return BenchmarkResult(
        backend=str(runner.backend),
        time_us=time_us,
        flop=flop,
        tflops=tflops,
    )


def print_comparison(results: dict):
    """Pretty print comparison results."""
    print(f"\n{'=' * 60}")
    print(f"Problem: {results['problem']}")
    print(f"Device:  {results['device']}")
    print(f"{'=' * 60}")

    for backend, res in results["backends"].items():
        if res.error:
            print(f"{backend:>10}: ERROR - {res.error}")
        else:
            tflops_str = f"{res.tflops:.2f} TFLOPS" if res.tflops else "N/A"
            print(f"{backend:>10}: {res.time_us:>10.2f} μs | {tflops_str}")

    if "speedup" in results:
        speedup = results["speedup"]
        winner = "Triton" if speedup > 1 else "PyTorch"
        print(f"{'-' * 60}")
        print(f"{'Speedup':>10}: {speedup:.2f}x ({winner} faster)")

    print(f"{'=' * 60}\n")
