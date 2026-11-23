"""Scatter-plot the raw timing samples of repeated DSL benchmark runs.

Writes one <kernel>.png per benchmark into the current directory, with the
backends on the x-axis and one color/marker per run directory.
"""

import argparse
import base64
import io
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SAMPLES_COLUMN = "times_us_npy_b64"
# Default x-axis order; other CSVs follow alphabetically.
BACKEND_ORDER = (
    "torch-eager",
    "torch-compile",
    "torch-compile-max-autotune",
    "triton",
    "cutile",
    "helion-mlir",
    "mlir",
)
# The lengths are coprime, so color/marker pairs only repeat after 70 runs.
RUN_COLORS = (
    "green",
    "blue",
    "red",
    "orange",
    "purple",
    "brown",
    "magenta",
    "cyan",
    "black",
    "gray",
)
RUN_MARKERS = ("o", "s", "^", "D", "v", "P", "X")


def load_samples(path: Path) -> dict[str, np.ndarray]:
    """Map each kernel_name in the CSV to its raw time samples in us."""
    frame = pd.read_csv(path)
    if SAMPLES_COLUMN not in frame:
        raise ValueError(f"{path}: missing column {SAMPLES_COLUMN!r}")
    names = frame["kernel_name"]
    duplicates = names[names.duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"{path}: duplicate kernel_name entries: {duplicates}")
    return {
        kernel: np.load(io.BytesIO(base64.b64decode(frame[SAMPLES_COLUMN][row])))
        for row, kernel in enumerate(names)
    }


def common_backends(run_dirs: list[Path]) -> list[str]:
    """CSV stems present in every run directory."""
    stems = [{path.stem for path in directory.glob("*.csv")} for directory in run_dirs]
    common = set.intersection(*stems)
    skipped = sorted(set.union(*stems) - common)
    if skipped:
        print(
            f"Skipping CSVs not present in every directory: {skipped}", file=sys.stderr
        )
    known = [backend for backend in BACKEND_ORDER if backend in common]
    return known + sorted(common.difference(BACKEND_ORDER))


def plot_kernel(
    kernel: str,
    backends: list[str],
    run_labels: list[str],
    samples: dict[str, list[dict[str, np.ndarray]]],
    output: Path,
) -> None:
    """Plot samples[backend][run][kernel] as one strip per backend and run."""
    group_width = 0.8
    strip_width = group_width / len(run_labels)

    fig, ax = plt.subplots(figsize=(max(8.0, 2.5 * len(backends)), 6))
    for run, label in enumerate(run_labels):
        for position, backend in enumerate(backends):
            y = samples[backend][run][kernel]
            # Spread the samples across the strip in iteration order.
            left = position - group_width / 2 + run * strip_width
            x = left + strip_width * np.linspace(0.1, 0.9, len(y))
            ax.plot(
                x,
                y,
                linestyle="none",
                marker=RUN_MARKERS[run % len(RUN_MARKERS)],
                color=RUN_COLORS[run % len(RUN_COLORS)],
                markersize=2,
                alpha=0.5,
                label=label if position == 0 else None,
            )

    ax.set_xticks(np.arange(len(backends)))
    ax.set_xticklabels(backends)
    ax.set_xlabel("Backend (samples in iteration order within each strip)")
    ax.set_ylabel("Time per iteration (us)")
    ax.set_title(Path(kernel).stem)
    ax.legend(markerscale=4)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scatter-plot the raw time samples of DSL benchmark runs, "
        "one <kernel>.png per benchmark in the current directory."
    )
    parser.add_argument(
        "directories",
        nargs="+",
        type=Path,
        help="Run directories containing the backend CSVs.",
    )
    parser.add_argument(
        "-b",
        "--backend",
        action="append",
        type=lambda name: name.removesuffix(".csv"),
        metavar="NAME",
        help="Backend CSV to plot, without the .csv suffix (e.g. 'triton'). "
        "Repeatable; plotted in the given order "
        "(default: all CSVs present in every directory).",
    )
    args = parser.parse_args()
    try:
        backends = args.backend or common_backends(args.directories)
        if not backends:
            raise ValueError("No CSV is present in every directory.")
        samples = {
            backend: [
                load_samples(directory / f"{backend}.csv")
                for directory in args.directories
            ]
            for backend in backends
        }
        kernels = list(samples[backends[0]][0])
        for backend in backends:
            for directory, run_samples in zip(args.directories, samples[backend]):
                if set(run_samples) != set(kernels):
                    raise ValueError(
                        f"{directory / f'{backend}.csv'}: kernel set differs from "
                        f"{args.directories[0] / f'{backends[0]}.csv'}"
                    )
        run_labels = [directory.resolve().name for directory in args.directories]
        for kernel in kernels:
            plot_kernel(
                kernel, backends, run_labels, samples, Path(f"{Path(kernel).stem}.png")
            )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
