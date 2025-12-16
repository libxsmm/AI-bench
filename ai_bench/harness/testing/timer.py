from collections.abc import Callable

import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile
from torch.profiler import record_function


def time_cpu(fn: Callable, args: tuple, warmup: int = 25, rep: int = 100) -> float:
    """Measure execution time of the provided function on CPU.
    Args:
        fn: Function to measure
        args: Arguments to pass to the function
        warmup: Warmup iterations
        rep: Measurement iterations
    Returns:
        Mean runtime in microseconds
    """
    for _ in range(warmup):
        fn(*args)

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(rep):
            with record_function("profiled_fn"):
                fn(*args)

    events = [e for e in prof.events() if e.name.startswith("profiled_fn")]
    times = torch.tensor([e.cpu_time for e in events], dtype=torch.float)

    # Trim extremes if there are enough measurements.
    if len(times) >= 10:
        times = torch.sort(times).values[1:-1]

    return torch.mean(times).item()


def time_xpu(fn: Callable, args: tuple, warmup: int = 25, rep: int = 100) -> float:
    """Measure execution time of the provided function on XPU.

    Uses hardware events for accurate GPU-side timing, with L2 cache flushing
    and a dummy matmul to improve accuracy for short-lived kernels.

    Args:
        fn: Function to measure
        args: Arguments to pass to the function
        warmup: Warmup iterations
        rep: Measurement iterations
    Returns:
        Mean runtime in microseconds
    """
    device = torch.device("xpu")

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
    torch.xpu.synchronize()

    # Pre-allocate events to reduce timing overhead.
    start_events = [torch.xpu.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.xpu.Event(enable_timing=True) for _ in range(rep)]

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
    torch.xpu.synchronize()

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
    device: torch.device | None = None,
) -> float:
    """Measure execution time of the provided function.
    Args:
        fn: Function to measure
        args: Arguments to pass to the function
        warmup: Warmup iterations
        rep: Measurement iterations
        device: Device type to use
    Returns:
        Mean runtime in microseconds
    """
    if not device or device.type == "cpu":
        return time_cpu(fn, args, warmup=warmup, rep=rep)
    if device.type == "xpu":
        return time_xpu(fn, args, warmup=warmup, rep=rep)
    raise ValueError(f"Unsupported device for timing: {device.type}")
