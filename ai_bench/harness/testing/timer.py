from collections.abc import Callable
from dataclasses import dataclass
import os
import time as pytime
import warnings

import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile
from torch.profiler import record_function
from torch.utils._pytree import tree_leaves
from torch.utils._pytree import tree_map

_MAX_ROTATION_SETS = 10_000


@dataclass
class TimeStats:
    """
    Per-iteration runtime statistics in microseconds.

    Args:
        mean: Mean
        min: Minimum
        max: Maximum
        median: Median
        std: Sample standard deviation
        rel_std: Standard deviation relative to the mean
    """

    mean: float
    min: float
    max: float
    median: float
    std: float
    rel_std: float


def _summarize(times: torch.Tensor) -> TimeStats:
    mean = times.mean()
    std = times.std() if len(times) > 1 else torch.zeros(())
    return TimeStats(
        mean=mean.item(),
        min=times.min().item(),
        max=times.max().item(),
        median=times.quantile(0.5).item(),
        std=std.item(),
        rel_std=(std / mean).item(),
    )


def _is_non_overlapping_and_dense(tensor: torch.Tensor) -> bool:
    expected_stride = 1
    for stride, size in sorted(zip(tensor.stride(), tensor.shape)):
        if size > 1:
            if stride != expected_stride:
                return False
            expected_stride *= size
    return True


def _validate_rotation_tensors(values, *, reject_views: bool = False):
    inputs = {}
    for leaf in tree_leaves(values):
        if isinstance(leaf, torch.Tensor):
            if (
                leaf.device.type != "cpu"
                or leaf.layout != torch.strided
                or not _is_non_overlapping_and_dense(leaf)
            ):
                raise ValueError(
                    "Cache rotation requires dense, non-overlapping CPU tensors"
                )
            if reject_views and leaf.numel():
                address = leaf.untyped_storage().data_ptr()
                if inputs.setdefault(address, leaf) is not leaf:
                    raise ValueError(
                        "Cache rotation does not support distinct input views sharing storage"
                    )


def _tensor_bytes(values) -> int:
    """Conservative logical footprint, deduplicated by backing storage."""
    storages = {}
    for leaf in tree_leaves(values):
        if isinstance(leaf, torch.Tensor):
            address = leaf.untyped_storage().data_ptr()
            storages[address] = max(storages.get(address, 0), leaf.nbytes)
    return sum(storages.values())


def _clone_args(args):
    copies = {}

    def clone(value):
        if not isinstance(value, torch.Tensor):
            return value
        if id(value) not in copies:
            copies[id(value)] = value.clone()
            if copies[id(value)].stride() != value.stride():
                raise ValueError("Cannot preserve this input's strides")
        return copies[id(value)]

    return tree_map(clone, args)


def time_cpu(
    fn: Callable,
    args: tuple,
    warmup: int = 25,
    rep: int = 100,
    cache_size_mib: int = 0,
) -> TimeStats:
    """Measure execution time of the provided function on CPU.

    With a cache size, fn must return its output tensors. Inputs and outputs
    must be dense, non-overlapping CPU tensors; distinct input views sharing
    storage are unsupported. Repeated references to the same input are preserved.
    fn must not mutate inputs or retain autograd graphs. Shapes and accessed
    regions must be stable across calls, and all input elements must be read.

    An untimed call sizes the nominal combined input/output footprint. Input
    copies rotate round-robin, and an untimed round fills their output slots.
    Output references are released outside timing, even with rotation disabled.
    Output storage reuse remains allocator-dependent, including interference
    from scratch or model-owned outputs. This is not guaranteed cache eviction;
    weights and scratch are not rotated. Excessive ring sizes raise an error.

    Args:
        fn: Function to measure
        args: Arguments to pass to the function
        warmup: Warmup iterations on the original arguments
        rep: Measurement iterations
        cache_size_mib: Target nominal intervening I/O footprint in MiB,
            or 0 to disable rotation
    Returns:
        Runtime statistics in microseconds
    """
    if warmup < 0 or rep <= 0 or cache_size_mib < 0:
        raise ValueError("Require warmup >= 0, rep > 0 and cache_size_mib >= 0")
    if cache_size_mib > 0:
        _validate_rotation_tensors(args, reject_views=True)

    # Supress profiler's warning, no event accumulation is needed.
    warnings.filterwarnings(
        "ignore",
        message="Warning: Profiler clears events",
        category=UserWarning,
    )
    # Supress Kineto logging.
    os.environ["KINETO_LOG_LEVEL"] = "99"

    nuke_a, nuke_b, nuke_c = None, None, None
    if min_cache_nuke_mib > 0:
        n_layers = torch.get_num_threads()  # ensure all cores are utilized
        nuke_size = math.ceil(
            math.sqrt(min_cache_nuke_mib * 1024 * 1024 / (3 * n_layers * 2))
        )  # 3 matrices per layer, 2 bytes per bfloat16
        nuke_a = torch.rand((n_layers, nuke_size, nuke_size), dtype=torch.bfloat16)
        nuke_b = torch.rand((n_layers, nuke_size, nuke_size), dtype=torch.bfloat16)
        nuke_c = torch.empty((n_layers, nuke_size, nuke_size), dtype=torch.bfloat16)

    for _ in range(warmup):
        fn(*args)

    n_sets = 1
    if cache_size_mib > 0:
        # Untimed call to measure the output size.
        out = fn(*args)
        if not any(isinstance(leaf, torch.Tensor) for leaf in tree_leaves(out)):
            raise ValueError("Cache rotation requires fn to return its output tensors")
        _validate_rotation_tensors(out)
        footprint = _tensor_bytes((args, out))
        del out
        if footprint == 0:
            raise ValueError("Cache rotation requires a nonzero tensor footprint")
        cache_bytes = cache_size_mib * 1024 * 1024
        n_sets = (cache_bytes + footprint - 1) // footprint + 1
        if n_sets > _MAX_ROTATION_SETS:
            raise ValueError(
                f"Cache rotation needs {n_sets} sets; limit is {_MAX_ROTATION_SETS}. "
                "Reduce cache_size_mib or use larger inputs."
            )

    arg_sets = [args] + [_clone_args(args) for _ in range(n_sets - 1)]
    outs = [None] * n_sets
    if n_sets > 1:
        # Untimed round in timed order: each timed call frees an output n_sets old.
        for i in range(n_sets):
            idx = (i + 1) % n_sets
            outs[idx] = fn(*arg_sets[idx])

    idx = 0
    try:
        with profile(activities=[ProfilerActivity.CPU], acc_events=False) as prof:
            for _ in range(rep):
                idx = (idx + 1) % n_sets
                outs[idx] = None

                with record_function("profiled_fn"):
                    outs[idx] = fn(*arg_sets[idx])

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
            idx = (idx + 1) % n_sets
            outs[idx] = None
            start = pytime.perf_counter_ns()
            outs[idx] = fn(*arg_sets[idx])
            end = pytime.perf_counter_ns()
            fallback_times.append((end - start) / 1e3)
        times = torch.tensor(fallback_times, dtype=torch.float)

    return _summarize(times)


def _time_gpu_wallclock(
    device: torch.device, fn: Callable, args: tuple, warmup: int = 25, rep: int = 100
) -> TimeStats:
    """Measure host wall-clock time of each call, with device synchronization
    before and after it.
    """
    # Buffer used to flush L2 cache between kernel runs.
    cache_size = 256 * 1024 * 1024
    cache = torch.empty(cache_size, dtype=torch.int8, device=device)

    for _ in range(warmup):
        cache.zero_()
        fn(*args)
    torch.accelerator.synchronize()

    times = []
    for _ in range(rep):
        cache.zero_()
        # Drain the device so the timed region contains only this call.
        torch.accelerator.synchronize()
        start = pytime.perf_counter_ns()
        fn(*args)
        torch.accelerator.synchronize()
        end = pytime.perf_counter_ns()
        times.append((end - start) / 1e3)
    times = torch.tensor(times, dtype=torch.float)

    return _summarize(times)


def _time_gpu_events(
    device: torch.device, fn: Callable, args: tuple, warmup: int = 25, rep: int = 100
) -> TimeStats:
    """Measure GPU-side execution time using device events.

    Only accurate when all work of the function is enqueued asynchronously
    without internal host synchronization.
    """
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

    return _summarize(times)


def time_gpu(
    device: torch.device, fn: Callable, args: tuple, warmup: int = 25, rep: int = 100
) -> TimeStats:
    """Measure execution time of the provided function on GPU.

    The timing strategy is selected with AIBENCH_GPU_TIMER:
      - wallclock (default): host wall-clock with device synchronization
        before and after each call
      - events: GPU event-based timer relying on batch execution and
        asynchronous kernel dispatch

    Args:
        device: Target device
        fn: Function to measure
        args: Arguments to pass to the function
        warmup: Warmup iterations
        rep: Measurement iterations
    Returns:
        Runtime statistics in microseconds
    """
    current_device = torch.accelerator.current_accelerator().type
    assert current_device == device.type, (
        f"Invalid accelerator {current_device}, expected {device.type}"
    )

    mode = os.environ.get("AIBENCH_GPU_TIMER", "wallclock").lower()
    if mode == "wallclock":
        return _time_gpu_wallclock(device, fn, args, warmup=warmup, rep=rep)
    if mode == "events":
        return _time_gpu_events(device, fn, args, warmup=warmup, rep=rep)
    raise ValueError(f"Invalid timer mode: {mode}")


def time(
    fn: Callable,
    args: tuple,
    warmup: int = 25,
    rep: int = 100,
    cache_size_mib: int = 0,
    device: torch.device | None = None,
) -> TimeStats:
    """Measure execution time of the provided function.
    Args:
        fn: Function to measure
        args: Arguments to pass to the function
        warmup: Warmup iterations
        rep: Measurement iterations
        cache_size_mib: Nominal intervening I/O footprint in MiB for CPU
            input rotation and allocator-dependent output reuse, or 0 to disable
        device: Device type to use
    Returns:
        Runtime statistics in microseconds
    """
    if not device or device.type == "cpu":
        return time_cpu(fn, args, warmup=warmup, rep=rep, cache_size_mib=cache_size_mib)
    if device.type == "xpu" or device.type == "cuda":
        return time_gpu(device, fn, args, warmup=warmup, rep=rep)
    raise ValueError(f"Unsupported device for timing: {device.type}")
