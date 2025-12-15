import json
import os
from pathlib import Path
import types

import torch
import yaml

from ai_bench import utils as ai_utils
from ai_bench.harness import core as ai_hc
from ai_bench.harness import testing
from ai_bench.utils.csv_logger import CSVLogger
from ai_bench.utils.logger import setup_logger


class KernelBenchRunner:
    """
    Run KernelBench problems.

    Args:
        spec_type: Type of problem spec to use
        device: Device to use
        backend: Backend to use (pytorch or triton)
        csv_path: Path to CSV file for logging (optional)
        note: Optional note to include in CSV
    """

    def __init__(
        self,
        spec_type: ai_hc.SpecKey = ai_hc.SpecKey.V_CI,
        device: torch.device | None = None,
        backend: ai_hc.Backend = ai_hc.Backend.PYTORCH,
        csv_path: str | None = None,
        note: str = "",
    ):
        self.specs = ai_utils.specs() / "KernelBench"
        self.backend = backend
        self.logger = setup_logger()
        self.csv_path = csv_path
        self.note = note
        self.csv_fieldnames = [
            "kernel_name",
            "kernel_type",
            "problem_level",
            "flops",
            "flops_val",
            "flops_unit",
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
        if backend == ai_hc.Backend.PYTORCH:
            self.kernels = ai_utils.kernel_bench_dir() / "KernelBench"
        elif backend == ai_hc.Backend.TRITON:
            self.kernels = ai_utils.triton_kernels_dir() / "KernelBench"
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        if not os.path.isdir(self.kernels):
            raise ValueError(f"Missing kernels directory for {backend}: {self.kernels}")

        self.spec_type = spec_type
        self.device = device if device else torch.device("cpu")
        if self.device.type == "cpu":
            self.warmup = 5
            self.rep = 20
        elif self.device.type == "xpu":
            self.warmup = 20
            self.rep = 100
        else:
            self.warmup = 25
            self.rep = 100

    def get_spec_dirs(self) -> list[Path]:
        """Get KernelBench level dirs.
        Returns:
            Paths to spec directories
        """
        return sorted([Path(dir) for dir in os.scandir(self.specs) if dir.is_dir()])

    def load_model(self, kernel_path: Path) -> types.ModuleType | None:
        """Load KernelBench model.
        All kernel modules are standarized with a class wrapper containing
        computation definition and a runner method.
        These models can be imported and used directly by the runner.
        Args:
            kernel_path: Path to KernelBench module '.py' file
        Returns:
            Loaded KernelBench model if available
        """
        if not kernel_path.is_file():
            return None
        mod = ai_utils.import_from_path("kernel_bench_model", kernel_path)
        if not hasattr(mod, "Model"):
            return None
        return mod.Model

    def run_kernels(self):
        """Run all KernelBench kernels."""
        self.logger.info(f"Backend: {self.backend}, Device: {self.device}")
        self.logger.info(f"Kernels: {self.kernels}")
        self.logger.info("-" * 60)

        # Iterate over specs of kernel levels.
        for spec_dir in self.get_spec_dirs():
            # Iterate over specs - one per kernel.
            for file in sorted(os.listdir(spec_dir)):
                with open(spec_dir / file) as f:
                    spec = yaml.safe_load(f)
                # Skip if desired configuration is not available.
                if self.spec_type not in spec:
                    continue
                variants = spec[self.spec_type]
                inputs = spec[ai_hc.SpecKey.INS]
                inits = []
                if ai_hc.SpecKey.INITS in spec:
                    inits = spec[ai_hc.SpecKey.INITS]

                # Import kernel file to access underlying Model and execution method.
                # Spec and kernel file names are expected to be identical.
                kernel_dir = self.kernels / spec_dir.name
                kernel_file = Path(kernel_dir / file.replace(".yaml", ".py"))
                model_obj = self.load_model(kernel_file)
                if not model_obj:
                    self.logger.debug(f"Missing kernel for: {file}")
                    continue
                # Run the kernel with provided input configurations.
                self.logger.info(f"Kernel: {spec_dir.name} / {file} [{self.backend}]")
                for variant in variants:
                    model_inits = ai_hc.get_inits(variant, inits)
                    model_dtype = ai_hc.get_variant_torch_dtype(variant)
                    model = model_obj(*model_inits).to(self.device, dtype=model_dtype)
                    fn = model.forward
                    args = ai_hc.get_inputs(variant, inputs, device=self.device)

                    # Simple CI run to verify functionality.
                    if self.spec_type == ai_hc.SpecKey.V_CI:
                        self.logger.info(f"Validating: {variant}")
                        fn(*args)
                        continue

                    self.logger.info(f"Benchmarking: {variant}")
                    meas = testing.time(
                        fn, args, warmup=self.warmup, rep=self.rep, device=self.device
                    )
                    flop = ai_hc.get_flop(variant)
                    flops_val = ""
                    flops_unit = ""
                    if flop:
                        tflops = flop / meas / 1e6
                        if tflops >= 1.0:
                            flops_val = tflops
                            flops_unit = "TFLOPS"
                        else:
                            flops_val = tflops * 1000
                            flops_unit = "GFLOPS"

                    self.logger.info(f"time [us]: {meas:.6f} {flops_unit}: {flops_val}")

                    if self.csv_logger:
                        aibench_env = {
                            k: v
                            for k, v in os.environ.items()
                            if k.startswith("AIBENCH_")
                        }
                        row = {
                            "kernel_name": file,
                            "kernel_type": str(self.backend),
                            "problem_level": spec_dir.name,
                            "flops": flop if flop is not None else "",
                            "flops_val": flops_val,
                            "flops_unit": flops_unit,
                            "time_us": meas,
                            "input_values": json.dumps(
                                variant.get(ai_hc.VKey.DIMS, {})
                            ),
                            "note": self.note,
                        }
                        row.update(aibench_env)
                        self.csv_logger.log(row)
