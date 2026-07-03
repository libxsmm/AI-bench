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

_NATIVE_BACKENDS = frozenset({ai_hc.Backend.SYCL})
_NATIVE_EXTENSIONS = {ai_hc.Backend.SYCL: ".cpp"}


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
        elif self.backend == ai_hc.Backend.GLUON:
            self.kernels = (
                ai_utils.gluon_kernels_dir() / self.device.type / "KernelBench"
            )
        elif self.backend == ai_hc.Backend.SYCL:
            self.kernels = (
                ai_utils.sycl_kernels_dir() / self.device.type / "KernelBench"
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

    def _kernel_extension(self) -> str:
        """Return the kernel file extension for the current backend."""
        return _NATIVE_EXTENSIONS.get(self.backend, ".py")

    def run_kernels(self):
        """Run all KernelBench kernels."""
        self.logger.info(f"Kernels: {self.kernels}")
        self.print_info(self.logger.info)

        ext = self._kernel_extension()

        # Iterate over specs of kernel levels.
        for spec_dir in self.get_spec_dirs():
            # Iterate over specs - one per kernel.
            for file in sorted(os.listdir(spec_dir)):
                # Spec and kernel file names are expected to be identical.
                kernel_dir = self.kernels / spec_dir.name
                kernel_file = Path(kernel_dir / file.replace(".yaml", ext))

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

    def run_kernel_spec(
        self, kernel_path: Path | str, spec_path: Path | str
    ) -> list[KernelStats] | None:
        """Run a kernel with a spec.

        Dispatches to native (subprocess) execution for SYCL backends,
        otherwise falls through to the base Python-based runner.
        """
        if isinstance(kernel_path, str):
            kernel_path = Path(kernel_path)
        if isinstance(spec_path, str):
            spec_path = Path(spec_path)

        if self.backend in _NATIVE_BACKENDS:
            return self._run_native_kernel_spec(kernel_path, spec_path)

        return super().run_kernel_spec(kernel_path, spec_path)

    def _run_native_kernel_spec(
        self, kernel_path: Path, spec_path: Path
    ) -> list[KernelStats] | None:
        """Compile and run a native (C++) kernel via subprocess."""
        import yaml

        from ai_bench.sycl.compiler import SYCLCompiler

        if not kernel_path.is_file():
            self.logger.debug(f"Missing native kernel: {kernel_path}")
            return None

        with open(spec_path) as f:
            spec = yaml.safe_load(f)

        if self.spec_type not in spec:
            return None

        compiler = SYCLCompiler()
        binary = compiler.compile(kernel_path)
        if binary is None:
            self.logger.error(f"Failed to compile: {kernel_path}")
            return None

        self.logger.info(
            f"Kernel: {spec_path.parent.name} / {spec_path.name} [{self.backend}]"
        )

        variants = spec[self.spec_type]
        stats = []
        for variant in variants:
            dims = variant.get("dims", {})
            m = dims.get("M", dims.get("N", 128))
            n = dims.get("N", m)
            k = dims.get("K", m)
            dtype = variant.get(ai_hc.VKey.TYPE)

            is_ci = self.spec_type == ai_hc.SpecKey.V_CI
            iterations = 0 if is_ci else 20

            self.logger.info(f"{'Validating' if is_ci else 'Benchmarking'}: {variant}")

            result = compiler.run(
                binary,
                m=m,
                n=n,
                k=k,
                iterations=iterations,
                verify=1,
                dtype=dtype,
            )

            if not result.success:
                self.logger.error(f"Execution failed: {result.error}")
                if result.raw_output:
                    self.logger.error(f"Output:\n{result.raw_output}")
                continue

            if result.passed is False:
                self.logger.error("Correctness check FAILED")
                continue

            if is_ci:
                self.logger.info("  Disposition: Passed")
                continue

            time_us = (result.time_ms * 1000) if result.time_ms else 0.0
            flop = ai_hc.get_flop(variant)
            flops_val = result.tflops
            flops_unit = str(config.FlopsUnit.TFLOPS) if flops_val else None

            self.logger.info(
                f"  time [us]: {time_us:.6f} {flops_unit or ''}: {flops_val or ''}"
            )

            stats.append(
                KernelStats(
                    variant=variant,
                    meas_us=time_us,
                    flop=flop,
                    flops=flops_val,
                    flops_unit=flops_unit,
                    flops_note=None,
                    mem_bytes=ai_hc.get_mem_bytes(variant),
                    mem_bw=None,
                    mem_bw_unit=None,
                    mem_note=None,
                )
            )

        return stats if stats else None
