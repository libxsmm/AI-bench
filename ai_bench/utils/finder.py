from pathlib import Path


def project_root() -> Path:
    """Path to the project's root directory."""
    return Path(__file__).parent.parent.parent


def specs() -> Path:
    """Path to the problem specs directory."""
    return project_root() / "problems" / "specs"


def kernel_bench_dir() -> Path:
    """Path to the KernelBench directory (PyTorch kernels)."""
    return project_root() / "third_party" / "KernelBench"


def triton_kernels_dir() -> Path:
    """Path to the Triton kernels directory."""
    return project_root() / "backends" / "triton"
