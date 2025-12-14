import ctypes
import hashlib
import os
from pathlib import Path
import subprocess

from ai_bench.utils.logger import setup_logger


class SYCLCompiler:
    """
    Compile SYCL kernels to shared libraries.

    Handles compilation with icpx -fsycl and caches compiled
    libraries by content hash to avoid recompilation.

    Args:
        cache_dir: Directory for cached compiled libraries
        target_arch: Target Intel GPU architecture
        debug: Enable debug symbols
        verbose: Print compilation commands
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        target_arch: str | None = None,
        debug: bool = False,
        verbose: bool = False,
    ):
        self.logger = setup_logger()
        self.cache_dir = cache_dir or Path.home() / ".cache" / "ai_bench" / "sycl"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Get target from environment or use default.
        self.target_arch = target_arch or os.environ.get(
            "DPCPP_SYCL_TARGET", "intel_gpu_pvc"
        )
        self.debug = debug
        self.verbose = verbose

        # Get SYCL TLA directory from environment if available.
        self.sycl_tla_dir = os.environ.get("SYCL_TLA_DIR", None)

    def _compute_hash(self, source_path: Path) -> str:
        """Compute hash of source file for caching."""
        content = source_path.read_bytes()
        return hashlib.sha256(content).hexdigest()[:12]

    def _get_cached_lib(self, source_path: Path) -> Path | None:
        """Check if compiled library exists in cache."""
        source_hash = self._compute_hash(source_path)
        lib_name = f"{source_path.stem}_{source_hash}.so"
        lib_path = self.cache_dir / lib_name
        if lib_path.exists():
            return lib_path
        return None

    def compile(self, source_path: Path) -> Path:
        """
        Compile SYCL source to shared library.

        Args:
            source_path: Path to .cpp SYCL source file

        Returns:
            Path to compiled shared library
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"SYCL source not found: {source_path}")

        # Check cache first.
        cached_lib = self._get_cached_lib(source_path)
        if cached_lib:
            self.logger.debug(f"Using cached library: {cached_lib}")
            return cached_lib

        # Compute output path.
        source_hash = self._compute_hash(source_path)
        lib_name = f"{source_path.stem}_{source_hash}.so"
        output_lib = self.cache_dir / lib_name

        # Build include directories.
        include_dirs = []
        if self.sycl_tla_dir:
            include_dirs.extend(
                [
                    f"{self.sycl_tla_dir}/include",
                    f"{self.sycl_tla_dir}/tools/util/include",
                ]
            )

        # Compiler flags.
        cxx_flags = [
            "-std=c++17",
            "-fsycl",
            "-fPIC",
            "-shared",
            f"-fsycl-targets={self.target_arch}",
            "-O3",
        ]

        if self.debug:
            cxx_flags.extend(["-g", "-DDEBUG"])
        else:
            cxx_flags.append("-DNDEBUG")

        # Warning flags.
        cxx_flags.extend(
            [
                "-Wall",
                "-Wextra",
                "-Wno-unused-parameter",
                "-Wno-unknown-pragmas",
            ]
        )

        # Build include flags.
        include_flags = [f"-I{d}" for d in include_dirs]

        # Link flags.
        link_flags = [
            "-lsycl",
        ]

        # Build compilation command.
        cmd = [
            "icpx",
            *cxx_flags,
            *include_flags,
            str(source_path),
            "-o",
            str(output_lib),
            *link_flags,
        ]

        if self.verbose:
            self.logger.info(f"Compilation command: {' '.join(cmd)}")

        # Run compilation.
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            if self.verbose and result.stdout:
                self.logger.info(result.stdout)
            self.logger.info(f"Compiled SYCL kernel: {output_lib}")
            return output_lib

        except subprocess.CalledProcessError as e:
            self.logger.error(f"SYCL compilation failed for {source_path}")
            self.logger.error(f"STDERR: {e.stderr}")
            if e.stdout:
                self.logger.error(f"STDOUT: {e.stdout}")
            raise RuntimeError(f"SYCL compilation failed: {e.stderr}") from e

        except FileNotFoundError:
            raise RuntimeError(
                "icpx compiler not found. Please install Intel oneAPI and source setvars.sh"
            )


class SYCLKernelLoader:
    """
    Load compiled SYCL shared libraries.

    Provides a wrapper around ctypes for loading and
    accessing SYCL kernel functions.

    Args:
        lib_path: Path to compiled shared library
    """

    def __init__(self, lib_path: Path):
        self.lib_path = Path(lib_path)
        self._lib = None

    def load(self) -> ctypes.CDLL:
        """Load the shared library."""
        if self._lib is None:
            self._lib = ctypes.CDLL(str(self.lib_path))
        return self._lib

    def get_function(self, name: str) -> ctypes.CFUNCTYPE:
        """Get a function from the loaded library."""
        lib = self.load()
        return getattr(lib, name)


__all__ = [
    "SYCLCompiler",
    "SYCLKernelLoader",
]
