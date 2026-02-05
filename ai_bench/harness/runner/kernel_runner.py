from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import types

import torch
import yaml

from . import config
from ai_bench import utils as ai_utils
from ai_bench.harness import core as ai_hc
from ai_bench.harness import testing
from ai_bench.utils.logger import setup_logger


@dataclass
class KernelStats:
    """
    Kernel execution statistics.

    Args:
        variant: Specs' variant entry
        meas_us: Mean runtime in microseconds
        flop: Number of floating point operations (FLOP)
        flops: FLOP per second (FLOPS)
        flops_unit: FLOPS measurement unit
        flops_note: FLOPS annotation
        mem_bytes: Number of memory access bytes
        mem_bw: Memory bandwidth
        mem_bw_unit: Memory bandwidth measurement unit
        mem_note: Memory bandwidth annotation
    """

    variant: dict
    meas_us: float
    flop: float | None
    flops: float | None
    flops_unit: config.FlopsUnit
    flops_note: config.NotesSymbols | None
    mem_bytes: float | None
    mem_bw: float | None
    mem_bw_unit: config.MemBwUnit
    mem_note: config.NotesSymbols | None


class KernelRunner:
    """
    Run a kernel problem.

    The kernel implementatation is expected to be wrapped in 'torch.nn.Module'
    and invoked in its 'forward' method.

    Args:
        spec_type: Type of problem spec to use
        device: Device to use
        backend: Backend to use
        flops_unit: FLOPS unit to use for reporting
        csv_path: Path to CSV file for logging (optional)
        note: Optional note to include in CSV
    """

    def __init__(
        self,
        spec_type: ai_hc.SpecKey = ai_hc.SpecKey.V_CI,
        device: torch.device | None = None,
        backend: ai_hc.Backend = ai_hc.Backend.PYTORCH,
        flops_unit: config.FlopsUnit = config.FlopsUnit.TFLOPS,
        mem_bw_unit: config.MemBwUnit = config.MemBwUnit.GBS,
    ):
        self.backend = backend
        self.logger = setup_logger()
        self.flops_unit = flops_unit
        self.mem_bw_unit = mem_bw_unit

        self.spec_type = spec_type
        self.device = device if device else torch.device("cpu")
        if self.is_cpu():
            self.warmup = 5
            self.rep = 20
        elif self.is_gpu():
            self.warmup = 200
            self.rep = 100
        else:
            self.warmup = 25
            self.rep = 100

    def is_torch_backend(self) -> bool:
        """Check if the backend is a torch variant.
        Returns:
            True if the current backend is torch-based.
        """
        return self.backend in [ai_hc.Backend.PYTORCH, ai_hc.Backend.PYTORCH_COMPILE]

    def is_cpu(self) -> bool:
        """Check if the device is a CPU."""
        return self.device.type == "cpu"

    def is_xpu(self) -> bool:
        """Check if the device is an XPU."""
        return self.device.type == "xpu"

    def is_cuda(self) -> bool:
        """Check if the device is a CUDA device."""
        return self.device.type == "cuda"

    def is_gpu(self) -> bool:
        """Check if the device is a GPU."""
        return self.is_xpu() or self.is_cuda()

    def load_model(self, kernel_path: Path) -> types.ModuleType | None:
        """Load a kernel model.

        All kernel modules are standarized with a class wrapper containing
        computation definition and a runner method.
        These models can be imported and used directly by the runner.

        Args:
            kernel_path: Path to PyTorch module '.py' file
        Returns:
            Loaded model if available
        """
        if not kernel_path.is_file():
            return None
        mod = ai_utils.import_from_path("kernel_model", kernel_path)
        if not hasattr(mod, "Model"):
            return None
        return mod.Model

    def print_info_legend(self, print_fn: Callable):
        """Print information legend.
        Args:
            print_fn: Callback to a printing function
        """
        print_fn("Legend:")
        print_fn(f"  - {config.NotesSymbols.ESTIMATE} : Estimated value")

    def print_info(self, print_fn: Callable | None = None):
        """Print general runner info.
        Args:
            print_fn: Callback to a printing function.
                Defaults to an INFO logger.
        """
        if not print_fn:
            print_fn = self.logger.info

        print_fn(f"Backend: {self.backend}, Device: {self.device}")
        print_fn(f"Problem spec: {self.spec_type}")
        self.print_info_legend(print_fn)
        print_fn("-" * 60)

    def run_kernel_spec(
        self, kernel_path: Path | str, spec_path: Path | str
    ) -> list[KernelStats] | None:
        """Run a kernel with a spec.
        Args:
            kernel_path: Path to kernel wrapped in PyTorch module '.py' file
            spec_path: Path to problem spec '.yaml' file
        Returns:
            Kernel statistics for all benchmarked variants.
            No statistics are available for CI spec.
            None is returned when execution is unsuccessful.
        """
        if isinstance(kernel_path, str):
            kernel_path = Path(kernel_path)
        if isinstance(spec_path, str):
            spec_path = Path(spec_path)

        with open(spec_path) as f:
            spec = yaml.safe_load(f)
        # Bail if desired configuration is not available.
        if self.spec_type not in spec:
            return None
        variants = spec[self.spec_type]
        inputs = spec[ai_hc.SpecKey.INS]
        inits = []
        if ai_hc.SpecKey.INITS in spec:
            inits = spec[ai_hc.SpecKey.INITS]

        # Import kernel file to access underlying Model and execution method.
        model_obj = self.load_model(kernel_path)
        if not model_obj:
            self.logger.debug(f"Missing kernel for: {kernel_path.name}")
            return None

        # Run the kernel with provided input configurations.
        self.logger.info(
            f"Kernel: {spec_path.parent.name} / {spec_path.name} [{self.backend}]"
        )
        stats = []
        for variant in variants:
            model_inits = ai_hc.get_inits(variant, inits)
            model_dtype = ai_hc.get_variant_torch_dtype(variant)
            model = model_obj(*model_inits).to(self.device, dtype=model_dtype)

            if self.backend == ai_hc.Backend.PYTORCH_COMPILE:
                model.compile(dynamic=False)

            # Call model directly to avoid skipping extra hooks if present.
            # It allows 'torch.compile' decorator to be invoked correctly.
            fn = model
            args = ai_hc.get_inputs(variant, inputs, device=self.device)

            # Simple CI run to verify functionality.
            if self.spec_type == ai_hc.SpecKey.V_CI:
                self.logger.info(f"Validating: {variant}")
                fn(*args)
                continue

            self.logger.info(f"Benchmarking: {variant}")
            meas_us = testing.time(
                fn, args, warmup=self.warmup, rep=self.rep, device=self.device
            )

            # Statistics - FLOPs.
            flop = ai_hc.get_flop(variant)
            flop_is_estimate = False
            if not flop and self.is_torch_backend():
                flop = ai_utils.count_torch_flop(fn, args)
                flop_is_estimate = True

            flops_val = None
            flops_unit = None
            flops_note = None
            if flop:
                tflops = flop / meas_us / 1e6
                match self.flops_unit:
                    case config.FlopsUnit.TFLOPS:
                        flops_val = tflops
                    case config.FlopsUnit.GFLOPS:
                        flops_val = tflops * 1000
                    case _:
                        raise ValueError(f"Invalid FLOPS unit: {self.flops_unit}")
                flops_unit = str(self.flops_unit)
                if flop_is_estimate:
                    flops_note = config.NotesSymbols.ESTIMATE

            self.logger.info(
                f"  time [us]: {meas_us:.6f} {str(flops_unit or '')}: {str(flops_val or '')} {str(flops_note or '')}"
            )

            # Statistics - memory bandwidth.
            mem_bytes = ai_hc.get_mem_bytes(variant)
            mem_is_estimate = False
            if not mem_bytes and self.is_torch_backend():
                mem_bytes = ai_utils.count_torch_memory_bytes(model, args)
                mem_is_estimate = True

            mem_bw_val = None
            mem_bw_unit = None
            mem_note = None
            if mem_bytes:
                gbs = mem_bytes / meas_us / 1e3
                match self.mem_bw_unit:
                    case config.MemBwUnit.GBS:
                        mem_bw_val = gbs
                    case config.MemBwUnit.MBS:
                        mem_bw_val = gbs * 1000
                    case _:
                        raise ValueError(
                            f"Invalid memory bandwidth unit: {self.mem_bw_unit}"
                        )
                mem_bw_unit = str(self.mem_bw_unit)
                if mem_is_estimate:
                    mem_note = config.NotesSymbols.ESTIMATE

                self.logger.info(
                    f"  {str(mem_bw_unit or '')}: {str(mem_bw_val or '')} {str(mem_note or '')}"
                )

            kernel_stats = KernelStats(
                variant=variant,
                meas_us=meas_us,
                flop=flop,
                flops=flops_val,
                flops_unit=flops_unit,
                flops_note=flops_note,
                mem_bytes=mem_bytes,
                mem_bw=mem_bw_val,
                mem_bw_unit=mem_bw_unit,
                mem_note=mem_note,
            )
            stats.append(kernel_stats)

        # Report statistics for all variants
        return stats
