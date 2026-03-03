import json
import os
from pathlib import Path

import torch

from . import config
from .kernel_runner import KernelRunner
from .kernel_runner import KernelStats
from ai_bench import utils as ai_utils
from ai_bench.harness import core as ai_hc
from ai_bench.utils.csv_logger import CSVLogger


class KernelBenchRunner(KernelRunner):
    """
    Run KernelBench problems.

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
        csv_path: str | None = None,
        note: str = "",
    ):
        super().__init__(
            spec_type=spec_type,
            device=device,
            backend=backend,
            flops_unit=flops_unit,
            mem_bw_unit=mem_bw_unit,
        )
        self.specs = ai_utils.specs() / "KernelBench"
        self.csv_path = csv_path
        self.note = note
        self.csv_fieldnames = [
            "kernel_name",
            "kernel_type",
            "problem_level",
            "flops",
            "flops_val",
            "flops_unit",
            "flops_note",
            "mem_bytes",
            "mem_bw_val",
            "mem_bw_unit",
            "mem_note",
            "time_us",
            "input_values",
            "note",
        ]
        aibench_env_keys = sorted(
            [k for k in os.environ.keys() if k.startswith("AIBENCH_")]
        )
        self.csv_fieldnames.extend(aibench_env_keys)

        if csv_path:
            self.csv_logger = CSVLogger(csv_path, self.csv_fieldnames)
        else:
            self.csv_logger = None

        # Set kernel directory based on backend.
        if self.is_torch_backend():
            self.kernels = ai_utils.kernel_bench_dir() / "KernelBench"
        elif self.backend == ai_hc.Backend.TRITON:
            self.kernels = (
                ai_utils.triton_kernels_dir() / self.device.type / "KernelBench"
            )
        elif self.backend == ai_hc.Backend.HELION:
            self.kernels = (
                ai_utils.helion_kernels_dir() / self.device.type / "KernelBench"
            )
        elif self.backend == ai_hc.Backend.MLIR:
            self.kernels = (
                ai_utils.mlir_kernels_dir() / self.device.type / "KernelBench"
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        if not os.path.isdir(self.kernels):
            raise ValueError(
                f"Missing kernels directory for {self.backend}: {self.kernels}"
            )

    def get_spec_dirs(self) -> list[Path]:
        """Get KernelBench level dirs.
        Returns:
            Paths to spec directories
        """
        return sorted(
            [Path(entry) for entry in os.scandir(self.specs) if entry.is_dir()]
        )

    def run_kernels(self):
        """Run all KernelBench kernels."""
        self.logger.info(f"Kernels: {self.kernels}")
        self.print_info(self.logger.info)

        # Iterate over specs of kernel levels.
        for spec_dir in self.get_spec_dirs():
            # Iterate over specs - one per kernel.
            for file in sorted(os.listdir(spec_dir)):
                # Spec and kernel file names are expected to be identical.
                kernel_dir = self.kernels / spec_dir.name
                kernel_file = Path(kernel_dir / file.replace(".yaml", ".py"))

                # Run the kernel in all compatible variants.
                run_stats: list[KernelStats] | None = self.run_kernel_spec(
                    kernel_file, spec_dir / file
                )

                # Continue if desired configuration is not available or
                # if there is nothing extra to report.
                if not run_stats:
                    continue

                if self.csv_logger:
                    # Log all executed variants.
                    aibench_env = {
                        k: v for k, v in os.environ.items() if k.startswith("AIBENCH_")
                    }
                    for run in run_stats:
                        row = {
                            "kernel_name": file,
                            "kernel_type": str(self.backend),
                            "problem_level": spec_dir.name,
                            "flops": str(run.flop or ""),
                            "flops_val": str(run.flops or ""),
                            "flops_unit": str(run.flops_unit or ""),
                            "flops_note": str(run.flops_note or ""),
                            "mem_bytes": str(run.mem_bytes or ""),
                            "mem_bw_val": str(run.mem_bw or ""),
                            "mem_bw_unit": str(run.mem_bw_unit or ""),
                            "mem_note": str(run.mem_note or ""),
                            "time_us": run.meas_us,
                            "input_values": json.dumps(
                                run.variant.get(ai_hc.VKey.DIMS, {})
                            ),
                            "note": self.note,
                        }
                        row.update(aibench_env)
                        self.csv_logger.log(row)
