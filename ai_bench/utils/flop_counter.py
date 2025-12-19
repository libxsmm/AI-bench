from collections.abc import Callable

from torch.utils.flop_counter import FlopCounterMode


def count_torch_flop(fn: Callable, args: tuple) -> int:
    """
    Estimate total number of floating-point operations in the given function.
    Only PyTorch operations are supported.

    Args:
        fn: Function to estimate number of flop
        args: Arguments to pass to the function
    Returns:
        Number of floating-point operations
    """
    with FlopCounterMode(display=None, depth=None) as flop_counter:
        fn(*args)
    return flop_counter.get_total_flops()
