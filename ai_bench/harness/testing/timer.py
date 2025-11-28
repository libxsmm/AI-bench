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
    raise ValueError("Unsupported device for timing")
