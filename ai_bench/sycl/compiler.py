"""SYCL kernel compiler and runner.

Compiles standalone SYCL C++ kernels (e.g. CUTLASS SYCL GEMM examples) into
executables and runs them as subprocesses. The compiled binary is expected to
print timing and correctness results to stdout.

Environment variables:
    AIBENCH_SYCL_COMPILER   Path to the SYCL compiler (default: icpx)
    AIBENCH_SYCL_FLAGS      Extra compiler flags (space-separated)
    AIBENCH_SYCL_INCLUDE    Colon-separated include paths for CUTLASS/SYCL headers
    AIBENCH_SYCL_TARGET     Target device for AOT compilation (default: bmg-g31)
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
import subprocess
import tempfile

logger = logging.getLogger(__name__)


@dataclass
class SYCLRunResult:
    success: bool
    passed: bool | None = None
    time_ms: float | None = None
    tflops: float | None = None
    raw_output: str = ""
    error: str = ""


class SYCLCompiler:
    """Compile and execute SYCL C++ kernel sources."""

    _DEFAULT_DEFINES = ["-DCUTLASS_ENABLE_SYCL", "-DSYCL_INTEL_TARGET"]
    _DEFAULT_FLAGS = ["-O2", "-std=c++17", "-fno-sycl-instrument-device-code"]

    def __init__(
        self,
        compiler: str | None = None,
        flags: list[str] | None = None,
        include_dirs: list[str] | None = None,
        target_device: str | None = None,
    ):
        self.compiler = compiler or os.environ.get("AIBENCH_SYCL_COMPILER", "icpx")
        self.target_device = target_device or os.environ.get("AIBENCH_SYCL_TARGET", "")

        if include_dirs is not None:
            self.include_dirs = include_dirs
        else:
            env_include = os.environ.get("AIBENCH_SYCL_INCLUDE", "")
            self.include_dirs = [d for d in env_include.split(":") if d]

        if flags is not None:
            self.flags = flags
        else:
            env_flags = os.environ.get("AIBENCH_SYCL_FLAGS", "")
            self.flags = env_flags.split() if env_flags else list(self._DEFAULT_FLAGS)

        self._build_dir = Path(tempfile.mkdtemp(prefix="aibench_sycl_"))

    def compile(self, source_path: Path) -> Path | None:
        """Compile a SYCL C++ source file into an executable.

        Args:
            source_path: Path to .cpp file
        Returns:
            Path to compiled binary, or None on failure
        """
        binary_name = source_path.stem
        binary_path = self._build_dir / binary_name

        cmd = [self.compiler, "-fsycl"]
        cmd.extend(self._DEFAULT_DEFINES)
        cmd.extend(self.flags)

        if self.target_device:
            cmd.extend(
                [
                    "-fsycl-targets=spir64_gen",
                    "-Xsycl-target-backend=spir64_gen",
                    f"-device {self.target_device}",
                    "-Xspirv-translator",
                    "-spirv-ext=+SPV_INTEL_split_barrier"
                    ",+SPV_INTEL_2d_block_io"
                    ",+SPV_INTEL_subgroup_matrix_multiply_accumulate",
                ]
            )

        for inc_dir in self.include_dirs:
            cmd.extend(["-I", inc_dir])

        cmd.extend(["-o", str(binary_path), str(source_path)])

        logger.info("Compiling SYCL kernel: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except FileNotFoundError:
            logger.error(
                "SYCL compiler not found: %s. Set AIBENCH_SYCL_COMPILER.",
                self.compiler,
            )
            return None
        except subprocess.TimeoutExpired:
            logger.error("SYCL compilation timed out for %s", source_path)
            return None

        if result.returncode != 0:
            logger.error("SYCL compilation failed:\n%s", result.stderr)
            return None

        logger.info("Compiled: %s", binary_path)
        return binary_path

    def run(
        self,
        binary_path: Path,
        m: int,
        n: int,
        k: int,
        iterations: int = 20,
        verify: int = 1,
        dtype: str | None = None,
    ) -> SYCLRunResult:
        """Execute a compiled SYCL kernel binary.

        Args:
            binary_path: Path to compiled executable
            m, n, k: Matrix dimensions
            iterations: Benchmark iterations
            verify: Whether to verify correctness (1=yes, 0=no)
            dtype: Data type hint (e.g. "bfloat16", "float16"). Forwarded to binary if set.
        Returns:
            Parsed execution results
        """
        cmd = [
            str(binary_path),
            f"--m={m}",
            f"--n={n}",
            f"--k={k}",
            f"--iterations={iterations}",
            f"--verify={verify}",
        ]
        if dtype:
            cmd.append(f"--dtype={dtype}")

        logger.info("Running SYCL kernel: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            return SYCLRunResult(success=False, error="Execution timed out")
        except Exception as e:
            return SYCLRunResult(success=False, error=str(e))

        output = result.stdout + result.stderr

        if result.returncode != 0:
            return SYCLRunResult(
                success=False, raw_output=output, error=f"Exit code {result.returncode}"
            )

        return self._parse_output(output)

    def compile_and_run(
        self,
        source_path: Path,
        m: int,
        n: int,
        k: int,
        iterations: int = 20,
        verify: int = 1,
        dtype: str | None = None,
    ) -> SYCLRunResult:
        """Compile a source file and run it in one step."""
        binary = self.compile(source_path)
        if binary is None:
            return SYCLRunResult(success=False, error="Compilation failed")
        return self.run(
            binary, m=m, n=n, k=k, iterations=iterations, verify=verify, dtype=dtype
        )

    @staticmethod
    def _parse_output(output: str) -> SYCLRunResult:
        """Parse stdout from a CUTLASS SYCL kernel binary.

        Expected output format:
            Disposition: Passed
            Problem Size: 5120x4096x4096x1
            Cutlass GEMM Performance:     [123.456]TFlop/s  (1.2345)ms
        """
        passed = None
        tflops = None
        time_ms = None

        disp_match = re.search(r"Disposition:\s*(Passed|Failed)", output)
        if disp_match:
            passed = disp_match.group(1) == "Passed"

        perf_match = re.search(r"\[([0-9.]+)\]\s*TFlop/s\s+\(([0-9.]+)\)\s*ms", output)
        if perf_match:
            tflops = float(perf_match.group(1))
            time_ms = float(perf_match.group(2))

        return SYCLRunResult(
            success=True,
            passed=passed,
            tflops=tflops,
            time_ms=time_ms,
            raw_output=output,
        )
