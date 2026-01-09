from collections import defaultdict
from typing import Callable
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle


class MemoryCounter:
    """
    Context manager that manages memory counting state and holds a summary of results.

    Internally, it counts ideal number of memory transfers through module hooks by tracking
    tensors accessed: inputs, fixed parameters (reads) and results (writes).

    The module should be invocated directly within the memory counter context to trigger all
    registered counting hooks.

    Args:
        module: PyTorch module for tracking

    Example:
        mod = torch.nn.Linear(2, 2)

        with MemoryCounter(mod) as mem_count:
            mod(torch.randn(2, 2))

        bytes = mem_count.get_total_bytes()
    """

    def __init__(self, module: nn.Module):
        self.module: nn.Module = module
        self.memory_stats = defaultdict(lambda: {"reads": 0, "writes": 0, "shapes": []})
        self.hooks: list[RemovableHandle] = []

    # Clears previous measurements on reentry.
    def __enter__(self):
        self._reset_stats()
        self._register_hooks()

        # Gather static module information.
        # It is assumed that these special module attributes
        # are loaded only once.
        reads = 0
        shapes = []

        # Count reads from all parameters.
        for param in self.module.parameters(recurse=True):
            if isinstance(param, torch.Tensor):
                reads += self._get_tensor_memory_size(param)
                shapes.append(param.size())
        # Count reads from all buffers.
        for buffer in self.module.buffers(recurse=True):
            if isinstance(buffer, torch.Tensor):
                reads += self._get_tensor_memory_size(buffer)
                shapes.append(buffer.size())

        # Store top-level statistics.
        self.memory_stats[self.module._get_name()]["reads"] += reads
        self.memory_stats[self.module._get_name()]["shapes"].append(shapes)

        return self

    def __exit__(self, *args):
        self._remove_hooks()

    def get_total_bytes(self) -> int:
        """Get total number of memory access bytes."""
        stats = self.get_total_stats()
        return stats["total_memory_bytes"]

    def get_total_stats(self) -> Dict[str, int]:
        """Get memory statistics summary across all modules."""
        total_reads = sum(stats["reads"] for stats in self.memory_stats.values())
        total_writes = sum(stats["writes"] for stats in self.memory_stats.values())

        return {
            "total_reads_bytes": total_reads,
            "total_writes_bytes": total_writes,
            "total_memory_bytes": total_reads + total_writes,
        }

    def print_memory_report(self):
        """Print a formatted memory counter report."""
        from pprint import pprint

        total_stats = self.get_total_stats()

        print("=" * 50)
        print("MEMORY TRACKING REPORT")
        print("=" * 50)
        print(
            f"Total Reads:  {total_stats['total_reads_bytes']:,} bytes ({total_stats['total_reads_bytes'] / 1024**2:.2f} MB)"
        )
        print(
            f"Total Writes: {total_stats['total_writes_bytes']:,} bytes ({total_stats['total_writes_bytes'] / 1024**2:.2f} MB)"
        )
        print(
            f"Total Memory: {total_stats['total_memory_bytes']:,} bytes ({total_stats['total_memory_bytes'] / 1024**2:.2f} MB)"
        )
        print("=" * 50)

        print("\nPER-MODULE BREAKDOWN:")
        print("-" * 50)
        print(f"{'Module':<15} | {'Reads (MB)':<15} | {'Writes (MB)':<15}")
        print("-" * 50)
        for name, stats in self.memory_stats.items():
            reads_mb = stats["reads"] / 1024**2
            writes_mb = stats["writes"] / 1024**2
            print(f"{name:<15} | {reads_mb:<15.2f} | {writes_mb:<15.2f}")

        print("")
        for name, stats in self.memory_stats.items():
            print(f"Module {name} - shapes:")
            for shape in stats["shapes"]:
                pprint(shape, compact=True)

    def _get_tensor_memory_size(self, tensor: torch.Tensor) -> int:
        """Calculate memory size in bytes for a tensor."""
        return tensor.numel() * tensor.element_size()

    def _create_hook(self, name: str) -> Callable:
        """Create a hook function for a specific module."""

        def hook_fn(module: nn.Module, inputs, outputs) -> None:
            reads = 0
            writes = 0
            shapes = []

            # Count reads from inputs.
            if isinstance(inputs, (tuple, list)):
                for tensor in inputs:
                    if isinstance(tensor, torch.Tensor):
                        reads += self._get_tensor_memory_size(tensor)
                        shapes.append(tensor.size())
            elif isinstance(inputs, torch.Tensor):
                reads += self._get_tensor_memory_size(inputs)
                shapes.append(inputs.size())

            # Count writes to outputs.
            if isinstance(outputs, (tuple, list)):
                for tensor in outputs:
                    if isinstance(tensor, torch.Tensor):
                        writes += self._get_tensor_memory_size(tensor)
                        shapes.append(tensor.size())
            elif isinstance(outputs, torch.Tensor):
                writes += self._get_tensor_memory_size(outputs)
                shapes.append(outputs.size())

            # Store statistics.
            self.memory_stats[name]["reads"] += reads
            self.memory_stats[name]["writes"] += writes
            self.memory_stats[name]["shapes"].append(shapes)

        return hook_fn

    def _register_hooks(self):
        """Register hooks for all modules in the model."""
        for name, mod in self.module.named_modules():
            # Only track leaf modules.
            if len(list(mod.children())) == 0:
                hook = mod.register_forward_hook(self._create_hook(name))
                self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _reset_stats(self):
        """Reset all statistics."""
        self.memory_stats.clear()


def count_torch_memory_bytes(module: nn.Module, args: tuple) -> int:
    """
    Estimate total number of memory access bytes (reads + writes)
    in the given PyTorch module.

    Args:
        module: PyTorch module for estimation
        args: Arguments to pass to the module
    Returns:
        Number of memory access bytes
    """
    # Disable gradient as only forward inference passes are considered.
    with torch.no_grad(), MemoryCounter(module) as mem_count:
        module(*args)
    return mem_count.get_total_bytes()
