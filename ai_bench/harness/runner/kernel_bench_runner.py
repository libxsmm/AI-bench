import os
from pathlib import Path
import sys
import types

import torch
import yaml

from ai_bench import utils as ai_utils
from ai_bench.harness import core as ai_hc


class KernelBenchRunner:
    def __init__(self, device: torch.device | None = None):
        self.specs = ai_utils.specs() / "KernelBench"
        self.kernels = ai_utils.kernel_bench_dir() / "KernelBench"
        if not os.path.isdir(self.kernels):
            raise ValueError("Missing KernelBench kernels directory")
        self.device = device
        self.spec_type = ai_hc.VKey.VAR_CI

    def get_spec_dirs(self) -> list[Path]:
        return [Path(dir) for dir in os.scandir(self.specs) if dir.is_dir()]

    def load_model(self, kernel_path: Path) -> types.ModuleType | None:
        if not kernel_path.is_file():
            return None
        mod = ai_utils.import_from_path("kernel_bench_model", kernel_path)
        if not hasattr(mod, "Model"):
            return None
        return mod.Model()

    def run_kernels(self):
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
                inputs = spec[ai_hc.InKey.INS]

                # Import kernel file to access underlying Model and execution method.
                # Spec and kernel file names are expected to be identical.
                kernel_dir = self.kernels / spec_dir.name
                kernel_file = Path(kernel_dir / file.replace(".yaml", ".py"))
                model = self.load_model(kernel_file)
                if not model:
                    print(f"Missing kernel for: {file}", file=sys.stderr)
                    continue

                # Run the kernel with provided input configurations.
                print(f"Kernel: {file}")
                for variant in variants:
                    print(f"Running: {variant}")
                    args = ai_hc.get_inputs(variant, inputs, device=self.device)
                    model.forward(*args)
