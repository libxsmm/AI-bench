"""Benchmark comparison utilities for ai_bench."""

from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Optional

import torch

from ai_bench.harness import core as ai_hc
from ai_bench.harness.runner import KernelBenchRunner
from ai_bench.utils.logger import setup_logger

logger = setup_logger()


def copy_model_weights(source_model, target_model) -> bool:
    """
    Copy weights from source model to target model.

    Handles cases where parameter names might differ slightly.

    Args:
        source_model: Model to copy weights from
        target_model: Model to copy weights to

    Returns:
        True if weights were successfully copied
    """

    try:
        source_state = source_model.state_dict()
        target_state = target_model.state_dict()

        # Try direct load first
        try:
            target_model.load_state_dict(source_state)
            logger.debug("Copied weights using direct state_dict load")
            return True
        except Exception as e:
            logger.debug(f"Direct state_dict load failed: {e}")

        # Try matching by shape if names differ
        source_params = list(source_state.items())
        target_params = list(target_state.keys())

        if len(source_params) != len(target_params):
            logger.warning(
                f"Parameter count mismatch: original={len(source_params)}, "
                f"optimized={len(target_params)}"
            )
            return False

        new_state = {}
        for (src_name, src_tensor), tgt_name in zip(source_params, target_params):
            tgt_shape = target_state[tgt_name].shape
            if src_tensor.shape != tgt_shape:
                logger.warning(
                    f"Shape mismatch for {src_name}->{tgt_name}: "
                    f"{src_tensor.shape} vs {tgt_shape}"
                )
                return False
            new_state[tgt_name] = src_tensor

        target_model.load_state_dict(new_state)
        logger.debug("Copied weights using shape-matched state_dict")
        return True

    except Exception as e:
        logger.warning(f"Failed to copy model weights: {e}")
        return False


def set_all_seeds(seed: int) -> None:
    """Set random seeds for all backends."""

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)

    # Also set Python random and numpy if available
    import random

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def check_correctness(original_output, optimized_output, rtol, atol) -> bool:
    """
    Compare two output tensors and check correctness.

    Args:
        original_output: Output from original kernel
        optimized_output: Output from optimized kernel
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if outputs match within tolerance
    """

    # Handle tuple outputs (some models return multiple tensors)
    if isinstance(original_output, tuple):
        original_output = original_output[0]
    if isinstance(optimized_output, tuple):
        optimized_output = optimized_output[0]

    # Check shapes
    if original_output.shape != optimized_output.shape:
        logger.warning(
            f"Shape mismatch: original={original_output.shape}, "
            f"optimized={optimized_output.shape}"
        )
        return False

    # Check for NaN/Inf
    if torch.isnan(original_output).any():
        logger.warning("Original output contains NaN values")
    if torch.isnan(optimized_output).any():
        logger.warning(
            "Optimized output contains NaN values - likely a bug in optimization"
        )
        return False
    if torch.isinf(optimized_output).any() and not torch.isinf(original_output).any():
        logger.warning("Optimized output contains Inf values not present in original")
        return False

    # Compare values
    is_close = torch.allclose(original_output, optimized_output, rtol=rtol, atol=atol)

    if not is_close:
        diff = torch.abs(original_output - optimized_output)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()

        # Calculate relative error
        rel_diff = diff / (torch.abs(original_output) + 1e-8)
        max_rel_diff = torch.max(rel_diff).item()

        logger.warning(
            f"Output mismatch: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, "
            f"max_rel_diff={max_rel_diff:.6e} (rtol={rtol}, atol={atol})"
        )

        # Log some debug info about where differences occur
        if diff.numel() > 0:
            num_mismatched = (
                (diff > atol + rtol * torch.abs(original_output)).sum().item()
            )
            total_elements = diff.numel()
            logger.debug(
                f"Mismatched elements: {num_mismatched}/{total_elements} "
                f"({100 * num_mismatched / total_elements:.2f}%)"
            )

    return is_close


@dataclass
class VariantResult:
    """
    Variant benchmark results.

    Args:
        variant: Specs' variant entry
        backends: Performance stats of each measured backend
        spec_flop: Number of floating point operations (FLOP)
        spec_mem_bytes: Number of memory access bytes
        speedups: Speedup of each measured backend
    """

    variant: defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))
    backends: defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))
    spec_flop: float | None = None
    spec_mem_bytes: float | None = None
    speedups: defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))


def benchmark_problem(
    problem: str,
    device: torch.device,
    spec_type: ai_hc.SpecKey = ai_hc.SpecKey.V_BENCH_GPU,
    rtol: float | None = None,
    atol: float | None = None,
    backends: Optional[List[ai_hc.Backend]] = None,
) -> dict:
    """Benchmark a specific problem across multiple backends.

    Tolerance resolution order (highest priority first):
        1. Explicit rtol/atol arguments (CLI override)
        2. Per-variant spec values (rtol/atol in YAML)
        3. Default values (1e-2 / 1e-5)

    Args:
        problem: Problem identifier in 'level/kernel_name' format
        device: Target device
        spec_type: Type of problem spec to use
        rtol: Override relative tolerance. If None, uses per-variant spec value.
        atol: Override absolute tolerance. If None, uses per-variant spec value.
        backends: List of backends to benchmark
    Returns:
        Dictionary with benchmark results
    """
    if backends is None:
        backends = [
            ai_hc.Backend.PYTORCH,
            ai_hc.Backend.PYTORCH_COMPILE,
        ]
    parts = problem.strip("/").split("/")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid problem format: '{problem}'. Expected 'level/kernel_name'"
        )

    logger.info(f"backends: {[str(backend) for backend in backends]}")
    logger.info(f"Device: {device}")
    logger.info(f"Problem spec: {spec_type}")

    level, kernel_name = parts

    pytorch_model = None

    # Set seed for reproducible inputs
    rand_seed = 123
    set_all_seeds(rand_seed)
    logger.info(f"Using seed: {rand_seed}")

    completed = False
    variant_idx = 0
    variant_results = []
    while not completed:
        logger.info(f"{'=' * 80}")
        logger.info(f"Running variant index: {variant_idx}")

        variant_result = VariantResult()
        variant_results.append(variant_result)
        variants = []

        for backend in backends:
            logger.info(f"backend: {backend}")
            try:
                runner = KernelBenchRunner(
                    spec_type=spec_type, device=device, backend=backend
                )

                spec_path = runner.specs / level / f"{kernel_name}.yaml"
                if not spec_path.exists():
                    raise FileNotFoundError(f"Spec not found: {spec_path}")

                kernel_path = runner.kernels / level / f"{kernel_name}.py"
                if not kernel_path.exists():
                    raise FileNotFoundError(f"Kernel not found: {kernel_path}")
                logger.info(f"kernel path: {kernel_path}")

                # Run the kernel in all compatible variants.
                model_obj = runner.load_model(kernel_path)
                if not model_obj:
                    raise ValueError("Missing kernel's entry model")

                spec = runner.load_spec(spec_path)
                variants = runner.get_spec_variants(spec)
                spec_inputs = runner.get_spec_inputs(spec)
                inits = runner.get_spec_inits(spec)

                logger.info(
                    f"Kernel: {spec_path.parent.name} / {spec_path.name} [{runner.backend}]"
                )
                logger.info(f"Variant: {variants[variant_idx]}")

                model = runner.init_model(model_obj, variants[variant_idx], inits)
                if backend == str(ai_hc.Backend.PYTORCH_COMPILE):
                    model.compile(dynamic=False)
                # save pytorch model for correctness check
                if backend == str(ai_hc.Backend.PYTORCH):
                    pytorch_model = model

                # prepare inputs
                inputs = ai_hc.get_inputs(
                    variants[variant_idx], spec_inputs, device=runner.device
                )

                # correctness check, only if we have a reference PyTorch model and we're not currently benchmarking the PyTorch backend
                if backend != str(ai_hc.Backend.PYTORCH) and pytorch_model:
                    weights_copied = copy_model_weights(pytorch_model, model)
                    if not weights_copied:
                        logger.warning(
                            "Could not copy weights - using seed-based initialization"
                        )

                    # Clone for fair comparison (in case kernels modify inputs) # TODO is this needed here?
                    inputs_orig = [inp.clone() for inp in inputs]
                    inputs_cur = [inp.clone() for inp in inputs]

                    # Run both kernels
                    with torch.no_grad():
                        pytorch_fn = pytorch_model.forward
                        pytorch_output = pytorch_fn(*inputs_orig)
                        cur_fn = model.forward
                        cur_output = cur_fn(*inputs_cur)

                    # Resolve tolerances: CLI override > spec value > default
                    effective_rtol = (
                        rtol
                        if rtol is not None
                        else ai_hc.get_rtol(variants[variant_idx])
                    )
                    effective_atol = (
                        atol
                        if atol is not None
                        else ai_hc.get_atol(variants[variant_idx])
                    )

                    #  Compare outputs
                    correct = check_correctness(
                        pytorch_output, cur_output, effective_rtol, effective_atol
                    )
                    if correct:
                        logger.info(
                            f"\033[92m✔\033[0m  Correctness check PASSED for {backend} "
                            f"(rtol={effective_rtol:.1e}, atol={effective_atol:.1e})"
                        )
                    else:
                        logger.warning(
                            f"\033[91m✘\033[0m  Correctness check FAILED for {backend} "
                            f"(rtol={effective_rtol:.1e}, atol={effective_atol:.1e})"
                        )

                # benchmark
                kernel_stats = runner.benchmark_model(
                    variants[variant_idx], model, inputs
                )

                # Continue if desired configuration is not available or
                # if there is nothing extra to report.
                if not kernel_stats:
                    logger.info(
                        f"Warning: received no results for {backend} backend. (variant={variants[variant_idx]})"
                    )
                    continue

                variant_result.backends[str(backend)] = kernel_stats
                variant_result.variant = variants[variant_idx]

            except Exception as e:
                logger.info(f"error: {e}")
                variant_result.backends[str(backend)] = None

        pytorch_res = variant_result.backends.get(str(ai_hc.Backend.PYTORCH))

        if pytorch_res:
            baseline_time = pytorch_res.meas_us
            variant_result.speedups = {
                b: baseline_time / r.meas_us
                for b, r in variant_result.backends.items()
                if r
            }

        logger.info("Perf summary:")
        print_variant_results_brief(variant_result, variant_idx)

        if variant_idx + 1 >= len(variants):
            completed = True
            break
        else:
            logger.info(f"-> Moving to next variant (idx={variant_idx + 1})")
            variant_idx += 1

    results = {"problem": problem, "device": str(device), "variants": variant_results}

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
    indicator = (
        "★★★" if cv_pct < 1 else "★★" if cv_pct < 5 else "★" if cv_pct < 10 else "⚠"
    )
    return f"{cv_pct:.2f}% {indicator}"


def print_variant_results_brief(variant_result: VariantResult, idx: int = 0):
    """Pretty print variant results - brief version."""
    fastest, fastest_time = None, float("inf")
    for backend, res in variant_result.backends.items():
        if res and res.meas_us > 0 and res.meas_us < fastest_time:
            fastest, fastest_time = backend, res.meas_us
    speedups = variant_result.speedups
    speedup_strs = [f"{b}: {s:.2f}x" for b, s in speedups.items()]

    logger.info(f"  Variant {idx}: {variant_result.variant}")
    logger.info(
        f"  Fastest: {fastest} ({fastest_time:.2f}μs) | {' '.join(speedup_strs)}"
    )


def print_variant_results(variant_result: VariantResult, idx: int = 0):
    """Pretty print variant results."""
    logger.info(f"{'=' * 80}")
    logger.info(f"Variant {idx}:")
    logger.info(variant_result.variant)
    logger.info(f"{'=' * 80}")
    spec_flop, spec_mem = variant_result.spec_flop, variant_result.spec_mem_bytes
    if spec_flop or spec_mem:
        parts = []
        if spec_flop:
            parts.append(f"FLOPs={_fmt_sci(spec_flop)}")
        if spec_mem:
            parts.append(f"Bytes={_fmt_sci(spec_mem)}")
        logger.info("Spec: " + "  ".join(parts))

    speedups = variant_result.speedups
    have_bw = any(r and r.mem_bw for r in variant_result.backends.values())

    flops_unit_list = [
        r.flops_unit for r in variant_result.backends.values() if r is not None
    ]
    flops_unit = next((f for f in flops_unit_list if f is not None), None)

    if not flops_unit:
        flops_unit = "FLOPS"
    if have_bw:
        logger.info(
            f"{'Backend':<18} {'Time (μs)':>12} {flops_unit:>8} {'GB/s':>8} {'Speedup':>10}"
        )
        logger.info("-" * 80)
    else:
        logger.info(
            f"{'Backend':<18} {'Time (μs)':>12} {flops_unit:>10} {'Speedup':>10}"
        )
        logger.info("-" * 70)

    for backend, res in variant_result.backends.items():
        if not res:
            logger.info(f"{backend:<18} {'ERROR: no results'}")
            continue
        speedup_str = f"{speedups.get(backend, 1.0):.2f}x"
        flops_str = f"{res.flops:.2f}" if res.flops else "N/A"

        if have_bw:
            gbs_str = f"{res.mem_bw:.1f}" if res.mem_bw else "N/A"
            logger.info(
                f"{backend:<18} {res.meas_us:>12.2f} {flops_str:>8} {gbs_str:>8} {speedup_str:>10}"
            )
        else:
            logger.info(
                f"{backend:<18} {res.meas_us:>12.2f} {flops_str:>10} {speedup_str:>10}"
            )

    logger.info("-" * (80 if have_bw else 70))

    valid = [(b, r) for b, r in variant_result.backends.items() if r]
    if valid:
        fastest = min(valid, key=lambda x: x[1].meas_us)
        slowest = max(valid, key=lambda x: x[1].meas_us)
        logger.info(f"Fastest: {fastest[0]} ({fastest[1].meas_us:.2f} μs)")
        if len(valid) > 1:
            logger.info(
                f"Max speedup: {slowest[1].meas_us / fastest[1].meas_us:.2f}x ({fastest[0]} vs {slowest[0]})"
            )

    logger.info(f"{'=' * (80 if have_bw else 70)}")


def print_comparison_brief(results: dict):
    """Pretty print comparison results - brief version."""
    logger.info("COMPARISON FINAL RESULTS:")
    logger.info(f"{'=' * 80}")
    logger.info(f"Problem: {results['problem']}")
    logger.info(f"Device:  {results['device']}")
    logger.info(f"{'=' * 80}")
    variant_results = results.get("variants", [])
    for idx, variant_result in enumerate(variant_results):
        logger.info(f"{'=' * 80}")
        print_variant_results_brief(variant_result, idx)


def print_comparison(results: dict):
    """Pretty print comparison results."""
    logger.info("COMPARISON FINAL RESULTS:")
    logger.info(f"{'=' * 80}")
    logger.info(f"Problem: {results['problem']}")
    logger.info(f"Device:  {results['device']}")
    logger.info(f"{'=' * 80}")
    for idx, variant_result in enumerate(results.get("variants", [])):
        print_variant_results(variant_result, idx)
