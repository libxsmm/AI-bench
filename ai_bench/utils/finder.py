"""Path configuration for ai_bench.

When used as a library, paths must be configured via configure() or environment variables.
When used as CLI from project root, paths are auto-detected.
"""

import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

# Global path configuration
_specs_dir: Path | None = None
_kernels_dir: Path | None = None
_triton_kernels_dir: Path | None = None
_env_loaded: bool = False


class ConfigurationError(Exception):
    """Raised when required paths are not configured."""

    pass


def load_env(env_path: Path | str | None = None, override: bool = False) -> bool:
    """Load configuration from .env file.

    Searches for .env file in the following order:
    1. Explicit path if provided
    2. Current working directory
    3. Project root directory

    Args:
        env_path: Explicit path to .env file (optional)
        override: If True, override existing environment variables

    Returns:
        True if .env file was loaded, False otherwise

    Example:
        >>> import ai_bench
        >>> ai_bench.load_env()  # Auto-find .env
        >>> ai_bench.load_env("/path/to/.env")  # Explicit path
        >>> ai_bench.load_env(override=True)  # Override existing vars
    """
    global _env_loaded

    if env_path is not None:
        path = Path(env_path)
        if path.is_file():
            load_dotenv(path, override=override)
            _env_loaded = True
            return True
        return False

    # Search order: CWD, then project root
    search_paths = [
        Path.cwd() / ".env",
        project_root() / ".env",
    ]

    for path in search_paths:
        if path.is_file():
            load_dotenv(path, override=override)
            _env_loaded = True
            return True

    return False


def is_env_loaded() -> bool:
    """Check if .env file has been loaded.

    Returns:
        True if load_env() successfully loaded a .env file
    """
    return _env_loaded


def configure(
    specs_dir: Path | str | None = None,
    kernels_dir: Path | str | None = None,
    triton_kernels_dir: Path | str | None = None,
) -> None:
    """Configure library paths.

    Call this before using KernelBenchRunner when using ai_bench as a library.

    Args:
        specs_dir: Path to YAML spec files directory
        kernels_dir: Path to PyTorch kernel implementations
        triton_kernels_dir: Path to Triton kernel implementations

    Example:
        >>> import ai_bench
        >>> ai_bench.configure(
        ...     specs_dir="/path/to/specs",
        ...     kernels_dir="/path/to/kernels",
        ... )
    """
    global _specs_dir, _kernels_dir, _triton_kernels_dir

    if specs_dir is not None:
        _specs_dir = Path(specs_dir)
    if kernels_dir is not None:
        _kernels_dir = Path(kernels_dir)
    if triton_kernels_dir is not None:
        _triton_kernels_dir = Path(triton_kernels_dir)


def reset_configuration() -> None:
    """Reset all path configurations to None.

    Useful for testing or reconfiguring.
    """
    global _specs_dir, _kernels_dir, _triton_kernels_dir, _env_loaded
    _specs_dir = None
    _kernels_dir = None
    _triton_kernels_dir = None
    _env_loaded = False


def _get_path(
    configured: Path | None,
    env_var: str,
    default_fn: Callable[[], Path],
    name: str,
) -> Path:
    """Get path from configuration, environment, or default.

    Priority:
    1. Explicitly configured via configure()
    2. Environment variable
    3. Default (relative to project_root)

    Args:
        configured: Explicitly configured path
        env_var: Environment variable name to check
        default_fn: Function returning default path
        name: Human-readable name for error messages

    Returns:
        Resolved path

    Raises:
        ConfigurationError: If path cannot be determined or doesn't exist
    """
    # Priority 1: Explicit configuration
    if configured is not None:
        if not configured.exists():
            raise ConfigurationError(f"{name} does not exist: {configured}")
        return configured

    # Priority 2: Environment variable
    env_path = os.environ.get(env_var)
    if env_path:
        path = Path(env_path)
        if not path.exists():
            raise ConfigurationError(f"{name} from {env_var} does not exist: {path}")
        return path

    # Priority 3: Default (project structure)
    try:
        return default_fn()
    except Exception:
        raise ConfigurationError(
            f"{name} not configured. Either:\n"
            f"  1. Call ai_bench.configure({name.lower().replace(' ', '_')}=...)\n"
            f"  2. Set {env_var} environment variable\n"
            f"  3. Run from project root with standard directory structure"
        )


def project_root() -> Path:
    """Path to the project's root directory.

    Returns:
        Project root path (parent of ai_bench package)
    """
    return Path(__file__).parent.parent.parent


def specs() -> Path:
    """Path to the problem specs directory.

    Can be configured via:
    - ai_bench.configure(specs_dir=...)
    - AIBENCH_SPECS_DIR environment variable
    - Auto-detected from project structure

    Returns:
        Path to specs directory

    Raises:
        ConfigurationError: If path cannot be determined
    """

    def default() -> Path:
        path = project_root() / "problems" / "specs"
        if not path.exists():
            raise FileNotFoundError(f"Default specs path not found: {path}")
        return path

    return _get_path(_specs_dir, "AIBENCH_SPECS_DIR", default, "Specs directory")


def kernel_bench_dir() -> Path:
    """Path to the KernelBench directory (PyTorch kernels).

    Can be configured via:
    - ai_bench.configure(kernels_dir=...)
    - AIBENCH_KERNELS_DIR environment variable
    - Auto-detected from project structure

    Returns:
        Path to PyTorch kernels directory

    Raises:
        ConfigurationError: If path cannot be determined
    """

    def default() -> Path:
        path = project_root() / "third_party" / "KernelBench"
        if not path.exists():
            raise FileNotFoundError(f"Default kernels path not found: {path}")
        return path

    return _get_path(_kernels_dir, "AIBENCH_KERNELS_DIR", default, "Kernels directory")


def triton_kernels_dir() -> Path:
    """Path to the Triton kernels directory.

    Can be configured via:
    - ai_bench.configure(triton_kernels_dir=...)
    - AIBENCH_TRITON_KERNELS_DIR environment variable
    - Auto-detected from project structure

    Returns:
        Path to Triton kernels directory

    Raises:
        ConfigurationError: If path cannot be determined
    """

    def default() -> Path:
        path = project_root() / "backends" / "triton"
        if not path.exists():
            raise FileNotFoundError(f"Default Triton kernels path not found: {path}")
        return path

    return _get_path(
        _triton_kernels_dir,
        "AIBENCH_TRITON_KERNELS_DIR",
        default,
        "Triton kernels directory",
    )
