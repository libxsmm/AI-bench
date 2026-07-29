from collections.abc import Callable
import math
import os
import time as pytime
import warnings

import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile
from torch.profiler import record_function


def time_cpu(
    fn: Callable,
    args: tuple,
    warmup: int = 25,
    rep: int = 100,
    min_cache_nuke_mib: int = 0,
) -> float:
    """Measure execution time of the provided function on CPU.
    Args:
        fn: Function to measure
        args: Arguments to pass to the function
        warmup: Warmup iterations
        rep: Measurement iterations
        min_cache_nuke_mib: Minimum memory size (in MiB) for a cache-nuking GEMM between timed iterations, or 0 to disable
    Returns:
        Mean runtime in microseconds
    """
    # Supress profiler's warning, no event accumulation is needed.
    warnings.filterwarnings(
        "ignore",
        message="Warning: Profiler clears events",
        category=UserWarning,
    )
    # Supress Kineto logging.
    os.environ["KINETO_LOG_LEVEL"] = "99"

    nuke_a, nuke_b = None, None
    if min_cache_nuke_mib > 0:
        nuke_size = 2048
        n_layers = math.ceil(
            min_cache_nuke_mib * 1024 * 1024 / (3 * nuke_size * nuke_size * 2)
        )  # 3 matrices, 2 bytes per bfloat16
        nuke_a = torch.rand((n_layers, nuke_size, nuke_size), dtype=torch.bfloat16)
        nuke_b = torch.rand((n_layers, nuke_size, nuke_size), dtype=torch.bfloat16)

    for _ in range(warmup):
        fn(*args)

    try:
        with profile(activities=[ProfilerActivity.CPU], acc_events=False) as prof:
            for _ in range(rep):
                if min_cache_nuke_mib > 0:
                    torch.bmm(nuke_a, nuke_b)

                with record_function("profiled_fn"):
                    fn(*args)

        events = [e for e in prof.events() if e.name.startswith("profiled_fn")]
        times = torch.tensor([e.cpu_time for e in events], dtype=torch.float)
    except RuntimeError as exc:
        # On some builds, enabling Kineto can fail when XPU profiling support is
        # unavailable even for CPU-only activity requests. Fall back to wall-clock.
        # FIXME: Should be solved in PyTorch 2.14.
        #   See: https://github.com/intel/torch-xpu-ops/issues/4318
        if "Kineto" not in str(exc) and "PTI_ERROR_NOT_IMPLEMENTED" not in str(exc):
            raise

        fallback_times = []
        for _ in range(rep):
            if min_cache_nuke_mib > 0:
                torch.bmm(nuke_a, nuke_b)
            t0 = pytime.perf_counter_ns()
            fn(*args)
            t1 = pytime.perf_counter_ns()
            fallback_times.append((t1 - t0) / 1e3)
        times = torch.tensor(fallback_times, dtype=torch.float)

    # Trim extremes if there are enough measurements.
    if len(times) >= 10:
        times = torch.sort(times).values[1:-1]

    return torch.mean(times).item()


def time_gpu(
    device: torch.device, fn: Callable, args: tuple, warmup: int = 25, rep: int = 100
) -> float:
    """Measure execution time of the provided function on GPU.

    Uses hardware events for accurate GPU-side timing, with L2 cache flushing
    and a dummy matmul to improve accuracy for short-lived kernels.

    Args:
        device: Target device
        fn: Function to measure
        args: Arguments to pass to the function
        warmup: Warmup iterations
        rep: Measurement iterations
    Returns:
        Mean runtime in microseconds
    """
    current_device = torch.accelerator.current_accelerator().type
    assert current_device == device.type, (
        f"Invalid accelerator {current_device}, expected {device.type}"
    )

    # Buffer used to flush L2 cache between kernel runs.
    cache_size = 256 * 1024 * 1024
    cache = torch.empty(cache_size, dtype=torch.int8, device=device)

    # Dummy matmul to fill GPU pipeline - helps with short-lived kernel timing.
    # Without this, fast kernels may complete before the CPU can issue the end event.
    dummy_a = torch.randn(1024, 1024, dtype=torch.float32, device=device)
    dummy_b = torch.randn(1024, 1024, dtype=torch.float32, device=device)

    # Warmup: load kernels and stabilize GPU state.
    for _ in range(warmup):
        cache.zero_()
        fn(*args)
    torch.accelerator.synchronize()

    # Pre-allocate events to reduce timing overhead.
    start_events = [torch.Event(device=device, enable_timing=True) for _ in range(rep)]
    end_events = [torch.Event(device=device, enable_timing=True) for _ in range(rep)]

    # Benchmark loop.
    for i in range(rep):
        # Flush L2 cache.
        cache.zero_()

        # Fill GPU pipeline with a dummy untimed kernel.
        #
        # GPU kernels are dispatched asynchronously.
        # Extra invocations fill up the stream and ensures that CPU has enough time
        # to enqueue timer events before the benchmarked kernel finishes execution.
        # It is particularly helpful to increase measurement accuracy of short-lived
        # workloads e.g., GEMM with small dimensions.
        torch.matmul(dummy_a, dummy_b)

        # Time the main kernel.
        start_events[i].record()
        fn(*args)
        end_events[i].record()

    # Ensure all measurements are recorded.
    torch.accelerator.synchronize()

    # Collect times (elapsed_time returns ms, convert to μs).
    times = torch.tensor(
        [s.elapsed_time(e) * 1e3 for s, e in zip(start_events, end_events)],
        dtype=torch.float,
    )

    # Trim extremes if there are enough measurements.
    if len(times) >= 10:
        times = torch.sort(times).values[1:-1]

    return torch.mean(times).item()


def time(
    fn: Callable,
    args: tuple,
    warmup: int = 25,
    rep: int = 100,
    min_cache_nuke_mib: int = 0,
    device: torch.device | None = None,
) -> float:
    """Measure execution time of the provided function.
    Args:
        fn: Function to measure
        args: Arguments to pass to the function
        warmup: Warmup iterations
        rep: Measurement iterations
        min_cache_nuke_mib: Minimum memory size (in MiB) for a cache-nuking GEMM between timed iterations on CPU, or 0 to disable
        device: Device type to use
    Returns:
        Mean runtime in microseconds
    """
    if not device or device.type == "cpu":
        return time_cpu(
            fn, args, warmup=warmup, rep=rep, min_cache_nuke_mib=min_cache_nuke_mib
        )
    if device.type == "xpu" or device.type == "cuda":
        return time_gpu(device, fn, args, warmup=warmup, rep=rep)
    raise ValueError(f"Unsupported device for timing: {device.type}")
