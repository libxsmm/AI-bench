import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np
import pandas as pd


# Keys in the input_values dict that map to each matmul dimension.
M_KEYS = ("BATCH_SIZE", "BATCH")
N_KEYS = ("OUT_FEATURES", "HIDDEN_SIZE", "OUTPUT_SIZE", "OUT_FEAT")
K_KEYS = ("IN_FEATURES", "INPUT_SIZE", "IN_FEAT", "IN_SIZE")


def _first(values: dict, keys: tuple) -> str:
    for key in keys:
        if key in values:
            return str(values[key])
    return "?"


def arithmetic_intensity(input_values: str) -> float:
    """Matmul arithmetic intensity: 2*M*N*K flops / (M*K + K*N + M*N) elems."""
    values = json.loads(input_values)
    m = float(_first(values, M_KEYS))
    n = float(_first(values, N_KEYS))
    k = float(_first(values, K_KEYS))
    return 2.0 * m * n * k / (m * k + k * n + m * n)


TORCH_CSV = "torch-eager.csv"
COMPILE_CSV = "torch-compile.csv"
COMPILE_MAX_AT_CSV = "torch-compile-max-autotune.csv"
# Okabe-Ito colorblind-safe palette; orange is reserved for Torch Inductor.
INDUCTOR_COLOR = "#E69F00"
BACKEND_COLORS = (
    "#0072B2",  # blue
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)


def parse_backend(spec: str) -> tuple[str, str]:
    label, sep, filename = spec.partition(":")
    if not sep or not label or not filename:
        raise argparse.ArgumentTypeError(f"expected LABEL:CSV, got {spec!r}")
    return label, filename


def average_runs(run_dirs: list[Path], filename: str) -> pd.DataFrame:
    """Average numeric columns by kernel and print runtime mean/stddev/min/max."""
    if not run_dirs:
        raise ValueError("At least one run directory is required.")

    frames = [
        pd.read_csv(directory / filename)
        .drop_duplicates(subset="kernel_name", keep="first")
        .set_index("kernel_name")
        for directory in run_dirs
    ]
    first = frames[0]
    for directory, frame in zip(run_dirs[1:], frames[1:]):
        if set(frame.index) != set(first.index):
            raise ValueError(f"{directory / filename}: kernel set differs from {run_dirs[0]}")
        for column in ("input_values", "dtype"):
            if column not in first:
                continue
            expected = first[column]
            actual = frame.reindex(first.index)[column]
            if column == "input_values":
                expected = expected.map(json.loads)
                actual = actual.map(json.loads)
            if not actual.equals(expected):
                raise ValueError(
                    f"{directory / filename}: {column} differs from {run_dirs[0]}"
                )

    grouped = pd.concat(frames).groupby(level="kernel_name", sort=False)
    means = grouped.mean(numeric_only=True)
    averaged = first.copy()
    averaged[means.columns] = means
    summary = pd.DataFrame({
        "mean": averaged["time_us"],
        "stddev": grouped["time_us"].std(ddof=1),
        "min": grouped["time_us"].min(),
        "max": grouped["time_us"].max(),
    })
    print(
        f"\n{filename}: runtime across {len(frames)} run(s) "
        "(us; sample stddev, ddof=1; NaN for one run)"
    )
    print(summary.to_string(float_format=lambda value: f"{value:.3f}"))
    return averaged.reset_index()


def plot_speedups(
    torch: pd.DataFrame,
    compile_: pd.DataFrame,
    compile_max_at: pd.DataFrame,
    backends: list[tuple[str, pd.DataFrame]],
    output: str | Path = "speedup.png",
    sort_by_intensity: bool = False,
    annotate_gflops: bool = False,
) -> None:
    """Plot speedups over torch eager for Torch Inductor and the given backends.

    Each dataframe must have unique kernel_name values.
    """
    if len(backends) > len(BACKEND_COLORS):
        raise ValueError(f"At most {len(BACKEND_COLORS)} backends are supported.")

    # Align all frames on kernel_name so the bars line up per kernel.
    merged = torch.add_suffix("_torch").rename(columns={"kernel_name_torch": "kernel_name"})
    for suffix, frame in (
        ("compile", compile_),
        ("compile_max_at", compile_max_at),
        *((str(i), frame) for i, (_, frame) in enumerate(backends)),
    ):
        merged = merged.merge(
            frame.add_suffix(f"_{suffix}"),
            left_on="kernel_name",
            right_on=f"kernel_name_{suffix}",
        )

    if sort_by_intensity:
        # Arithmetic intensity is derived from the M, N, K input dims.
        merged = merged.sort_values(
            by="input_values_torch", key=lambda col: col.map(arithmetic_intensity), kind="stable",
        ).reset_index(drop=True)

    def speedup_and_gflops(time_us: pd.Series) -> tuple[pd.Series, pd.Series]:
        # Speedup over torch-eager = torch runtime / candidate runtime.
        speedup = merged["time_us_torch"] / time_us
        # Absolute throughput from the torch flop count divided by the runtime.
        gflops = merged["flops_torch"] / (time_us * 1e-6) / 1e9
        return speedup, gflops

    # Best of compile and compile max-autotune per kernel.
    best_compile_time_us = np.minimum(merged["time_us_compile"], merged["time_us_compile_max_at"])
    series = [(*speedup_and_gflops(best_compile_time_us), "Torch Inductor", INDUCTOR_COLOR)]
    for i, ((label, _), color) in enumerate(zip(backends, BACKEND_COLORS)):
        series.append((*speedup_and_gflops(merged[f"time_us_{i}"]), label, color))

    def geomean(series: pd.Series) -> float:
        return float(np.exp(np.log(series).mean()))

    # Geomean speedups per backend vs. eager and vs. best-of Torch Inductor.
    print("\\begin{tabular}{lrr}")
    print("\\toprule")
    print("Backend & vs. Torch eager & vs. Torch Inductor \\\\")
    print("\\midrule")
    for i, (label, _) in enumerate(backends):
        time_us = merged[f"time_us_{i}"]
        vs_eager = geomean(merged["time_us_torch"] / time_us)
        vs_inductor = geomean(best_compile_time_us / time_us)
        print(f"{label} & {vs_eager:.2f}$\\times$ & {vs_inductor:.2f}$\\times$ \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

    kernels = merged["kernel_name"].tolist()
    labels = [f"{name[:name.find('_')]}" for name in kernels]
    x = np.arange(len(kernels))

    group_width = 0.8
    bar_width = group_width / len(series)

    fig, ax = plt.subplots(figsize=(24, 8))
    for i, (values, gflops, label, color) in enumerate(series):
        y = np.asarray(values, dtype=float)
        offsets = x - group_width / 2 + (i + 0.5) * bar_width
        ax.bar(offsets, y, width=bar_width, label=label, color=color)
        if not annotate_gflops:
            continue
        for xi, yi, gi in zip(offsets, y, np.asarray(gflops, dtype=float)):
            ax.annotate(
                f"{gi:.0f}",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=9,
                color=color,
            )

    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xlim(x[0] - group_width / 2 - 0.2, x[-1] + group_width / 2 + 0.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylabel("Speedup over Torch eager execution", fontsize=20)
    ax.set_xlabel("KernelBench - level2/...", fontsize=20)
    ax.legend(fontsize=16)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 5, 10]))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", linestyle="-", alpha=0.4)
    ax.grid(axis="y", which="minor", linestyle=":", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average repeated DSL benchmark runs and plot speedups."
    )
    parser.add_argument(
        "directories", nargs="*", type=Path, default=[Path(".")],
        help="Run directories containing the backend CSVs (default: current directory).",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("dsl-paper-eval-speedup.png"),
        help="Output image path (default: %(default)s).",
    )
    parser.add_argument(
        "-b", "--backend", action="append", type=parse_backend, metavar="LABEL:CSV", default=[],
        help="Backend to plot next to Torch Inductor, as plot label and CSV filename "
        "(e.g. 'Triton CPU:triton.csv'). Repeatable.",
    )
    parser.add_argument(
        "-s", "--sort", action="store_true",
        help="Sort kernels by arithmetic intensity instead of CSV order.",
    )
    parser.add_argument(
        "-g", "--gflops", action="store_true",
        help="Annotate each bar with its absolute throughput in GFLOPS.",
    )
    args = parser.parse_args()
    try:
        torch, compile_, compile_max_at = (
            average_runs(args.directories, filename)
            for filename in (TORCH_CSV, COMPILE_CSV, COMPILE_MAX_AT_CSV)
        )
        backend_frames = [
            (label, average_runs(args.directories, filename)) for label, filename in args.backend
        ]
        plot_speedups(
            torch, compile_, compile_max_at, backend_frames,
            output=args.output, sort_by_intensity=args.sort,
            annotate_gflops=args.gflops,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
