from collections.abc import Callable
import itertools

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


# Based on Intel XPU Triton backend benchmarks.
def time_xpu(fn: Callable, args: tuple, warmup: int = 25, rep: int = 100) -> float:
    """Measure execution time of the provided function on XPU.
    Args:
        fn: Function to measure
        args: Arguments to pass to the function
        warmup: Warmup iterations
        rep: Measurement iterations
    Returns:
        Mean runtime in microseconds
    """

    # A device buffer used to clear L2 cache between kernel runs.
    cache_size = 256 * 1024 * 1024
    cache = torch.empty(cache_size, dtype=torch.int8, device=torch.device("xpu"))

    for _ in range(warmup):
        fn(*args)
        torch.xpu.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        for _ in range(rep):
            # Clear L2 cache.
            cache.zero_()
            torch.xpu.synchronize()

            with record_function("profiled_fn"):
                fn(*args)
        # Ensure all measurements are recorded.
        torch.xpu.synchronize()

    def extract_kernels(funcs):
        """Traverse event tree recursively to extract device kernels."""
        kernels = []
        kernels += list(
            itertools.chain.from_iterable(
                map(lambda func: extract_kernels(func.cpu_children), funcs)
            )
        )
        kernels += list(itertools.chain.from_iterable([func.kernels for func in funcs]))
        return kernels

    events = [e for e in prof.events() if e.name.startswith("profiled_fn")]
    kernels = [extract_kernels(func.cpu_children) for func in events]
    kernels = [kernel for kernel in kernels if kernel]
    if len(kernels) != rep:
        raise AssertionError("Unexpected number of profiled kernels")

    times = torch.tensor(
        [sum([k.duration for k in kernel]) for kernel in kernels], dtype=torch.float
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
