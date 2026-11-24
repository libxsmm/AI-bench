import argparse
from fractions import Fraction
import json
from pathlib import Path
import re
import sys

from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MaxNLocator
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
    raise ValueError(f"None of the keys {keys} found in {values}")


def matmul_dims(input_values: str) -> tuple[int, int, int]:
    """M, N and K of the matmul underlying the kernel."""
    values = json.loads(input_values)
    m, n, k = (int(_first(values, keys)) for keys in (M_KEYS, N_KEYS, K_KEYS))
    return m, n, k


def arithmetic_intensity(input_values: str) -> float:
    """Matmul arithmetic intensity: 2*M*N*K flops / (M*K + K*N + M*N) elems."""
    m, n, k = matmul_dims(input_values)
    return 2.0 * m * n * k / (m * k + k * n + m * n)


TORCH_CSV = "torch-eager.csv"
COMPILE_CSV = "torch-compile.csv"
COMPILE_MAX_AT_CSV = "torch-compile-max-autotune.csv"
REL_STD_WARN_PCT = 5.0
MEAN_MEDIAN_WARN_PCT = 1.0
# Per-iteration mean, min, max, median and std, in LaTeX table order.
TABLE_TIME_COLUMNS = (
    "time_us",
    "time_min_us",
    "time_max_us",
    "time_median_us",
    "time_std_us",
)
EXCEL_DROPPED_COLUMNS = [
    "kernel_type",
    "problem_level",
    "flops_note",
    "mem_bytes",
    "mem_bw_val",
    "mem_bw_unit",
    "mem_note",
    "note",
    "AIBENCH_SPECS_DIR",
    "times_us_npy_b64",
]
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
# Plot order of the categories and of the kernels they contain.
KERNEL_CATEGORIES = {
    "Element-wise only": (
        "9_Matmul_Subtract_Multiply_ReLU",
        "12_Gemm_Multiply_LeakyReLU",
        "33_Gemm_Scale_BatchNorm",
        "39_Gemm_Scale_BatchNorm",
        "40_Matmul_Scaling_ResidualAdd",
        "63_Gemm_ReLU_Divide",
        "68_Matmul_Min_Subtract",
        "76_Gemm_Add_ReLU",
    ),
    "Transcendental": (
        "29_Matmul_Mish_Mish",
        "41_Gemm_BatchNorm_GELU_ReLU",
        "53_Gemm_Scaling_Hardtanh_GELU",
        "59_Matmul_Swish_Scaling",
        "70_Gemm_Sigmoid_Scaling_ResidualAdd",
        "81_Gemm_Swish_Divide_Clamp_Tanh_Clamp",
        "86_Matmul_Divide_GELU",
        "95_Matmul_Add_Swish_Tanh_GELU_Hardtanh",
        "97_Matmul_BatchNorm_BiasAdd_Divide_Swish",
    ),
    "Reduction": (
        "14_Gemm_Divide_Sum_Scaling",
        "30_Gemm_GroupNorm_Hardtanh",
        "62_Matmul_GroupNorm_LeakyReLU_Sum",
    ),
    "Transcendental and reduction": (
        "37_Matmul_Swish_Sum_GroupNorm",
        "56_Matmul_Sigmoid_Sum",
        "88_Gemm_GroupNorm_Swish_Multiply_Swish",
        "94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm",
    ),
}
# The full names do not fit under the single-column hero plot.
CATEGORY_ABBREVIATIONS = {
    "Element-wise only": "(E)",
    "Transcendental": "(T)",
    "Reduction": "(R)",
    "Transcendental and reduction": "(T+R)",
}


def kernel_category(kernel_name: str) -> int:
    """Position in KERNEL_CATEGORIES of the category containing the kernel."""
    stem = kernel_name.removesuffix(".yaml")
    for index, kernels in enumerate(KERNEL_CATEGORIES.values()):
        if stem in kernels:
            return index
    raise ValueError(f"{kernel_name}: kernel is not assigned to a category")


def order_kernels(
    frame: pd.DataFrame,
    input_values: str,
    sort_by_intensity: bool,
    group_by_category: bool,
) -> pd.DataFrame:
    """Sort rows by arithmetic intensity, then group them by category."""
    if sort_by_intensity:
        # Arithmetic intensity is derived from the M, N, K input dims.
        frame = frame.sort_values(
            by=input_values,
            key=lambda col: col.map(arithmetic_intensity),
            kind="stable",
        )
    if group_by_category:
        # Stable, so kernels keep their CSV or intensity order within a category.
        frame = frame.sort_values(
            by="kernel_name",
            key=lambda col: col.map(kernel_category),
            kind="stable",
        )
    return frame.reset_index(drop=True)


def parse_backend(spec: str) -> tuple[str, str]:
    label, sep, filename = spec.partition(":")
    if not sep or not label or not filename:
        raise argparse.ArgumentTypeError(f"expected LABEL:CSV, got {spec!r}")
    return label, filename


def _warn_above(
    warnings: list[str], source: object, what: str, values: pd.Series, limit: float
) -> None:
    for kernel, value in values[values > limit].items():
        warnings.append(f"{source}: {kernel}: {what} = {value:.2f}% (> {limit:g}%)")


def average_runs(
    run_dirs: list[Path], filename: str, warnings: list[str], verbose: bool = False
) -> pd.DataFrame:
    """Average the runtime by kernel; if verbose, print its statistics across runs.

    Kernels whose relative standard deviation exceeds REL_STD_WARN_PCT or whose
    mean and median differ by more than MEAN_MEDIAN_WARN_PCT, within a run or
    across runs, are appended to warnings.
    """
    if not run_dirs:
        raise ValueError("At least one run directory is required.")

    frames = [
        pd.read_csv(directory / filename).set_index("kernel_name")
        for directory in run_dirs
    ]
    for directory, frame in zip(run_dirs, frames):
        duplicates = frame.index[frame.index.duplicated()].unique().tolist()
        if duplicates:
            raise ValueError(
                f"{directory / filename}: duplicate kernel_name entries: {duplicates}"
            )
        # NaN compares False, so it is reported alongside zero and negative times.
        invalid = frame.index[~(frame["time_us"] > 0)].tolist()
        if invalid:
            raise ValueError(
                f"{directory / filename}: time_us must be > 0 for kernels: {invalid}"
            )
        # CSVs written before the per-iteration statistics lack these columns.
        if "time_rel_std_pct" in frame:
            _warn_above(
                warnings,
                directory / filename,
                "rel std across iterations",
                frame["time_rel_std_pct"],
                REL_STD_WARN_PCT,
            )
        if "time_median_us" in frame:
            _warn_above(
                warnings,
                directory / filename,
                "mean/median difference across iterations",
                100
                * (frame["time_us"] - frame["time_median_us"]).abs()
                / frame["time_us"],
                MEAN_MEDIAN_WARN_PCT,
            )
    first = frames[0]
    for directory, frame in zip(run_dirs[1:], frames[1:]):
        if set(frame.index) != set(first.index):
            raise ValueError(
                f"{directory / filename}: kernel set differs from {run_dirs[0]}"
            )
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

    # Statistics over the per-run mean runtimes, not over individual iterations.
    run_times = grouped["time_us"]
    mean = run_times.mean()
    median = run_times.median()
    std = run_times.std(ddof=1)
    rel_std = 100 * std / mean
    runs = f"across {len(frames)} runs"
    _warn_above(warnings, filename, f"rel std {runs}", rel_std, REL_STD_WARN_PCT)
    _warn_above(
        warnings,
        filename,
        f"mean/median difference {runs}",
        100 * (mean - median).abs() / mean,
        MEAN_MEDIAN_WARN_PCT,
    )
    if verbose:
        summary = pd.DataFrame(
            {
                "mean": mean,
                "min": run_times.min(),
                "max": run_times.max(),
                "median": median,
                "std": std,
                "rel std [%]": rel_std,
            }
        )
        print(
            f"\n{filename}: per-run mean runtime across {len(frames)} run(s) "
            "(us; sample std, ddof=1; NaN for one run)"
        )
        print(summary.to_string(float_format=lambda value: f"{value:.3f}"))

    # Produce summary frame for plotting.
    averaged = first.copy()
    averaged["time_us"] = mean
    return averaged.reset_index()


def best_of(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    """Per kernel, the row of whichever frame has the lower mean runtime."""
    if set(first["kernel_name"]) != set(second["kernel_name"]):
        raise ValueError("cannot pick the best of differing kernel sets")
    first = first.set_index("kernel_name")
    second = second.set_index("kernel_name").loc[first.index]
    for column in ("input_values", "dtype"):
        expected = first[column]
        actual = second[column]
        if column == "input_values":
            expected = expected.map(json.loads)
            actual = actual.map(json.loads)
        if not actual.equals(expected):
            raise ValueError(f"cannot pick the best of frames with differing {column}")
    second_wins = second["time_us"] < first["time_us"]
    best = pd.concat([first[~second_wins], second[second_wins]])
    return best.loc[first.index].reset_index()


def _tex_escape(text: str) -> str:
    return re.sub(r"([&%$#_{}])", r"\\\1", text)


def _write_latex_document(output: Path, body: list[str]) -> None:
    lines = [
        "\\documentclass[letterpaper]{article}",
        "\\usepackage[margin=0.75in]{geometry}",
        "\\usepackage{booktabs}",
        "\\begin{document}",
        "",
        *body,
        "\\end{document}",
    ]
    output.write_text("\n".join(lines) + "\n")
    print(f"Saved {output}")


def write_geomean_table(
    torch: pd.DataFrame,
    inductor: pd.DataFrame,
    backends: list[tuple[str, pd.DataFrame]],
    output: Path,
) -> None:
    """Save the geomean speedups of the backends as a LaTeX document.

    The speedups are relative to Torch eager and to Torch Inductor. All frames
    must cover the same kernels.
    """

    def geomean(speedups: pd.Series) -> float:
        return float(np.exp(np.log(speedups).mean()))

    torch_time_us = torch.set_index("kernel_name")["time_us"]
    inductor_time_us = inductor.set_index("kernel_name")["time_us"]
    lines = [
        "\\begin{center}",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Backend & vs. Torch eager & vs. Torch Inductor \\\\",
        "\\midrule",
    ]
    for label, frame in backends:
        # The divisions align on the kernel name.
        time_us = frame.set_index("kernel_name")["time_us"]
        vs_eager = geomean(torch_time_us / time_us)
        vs_inductor = geomean(inductor_time_us / time_us)
        lines.append(
            f"{_tex_escape(label)} & {vs_eager:.2f}$\\times$ & "
            f"{vs_inductor:.2f}$\\times$ \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{center}", ""]
    _write_latex_document(output, lines)


def write_latex_tables(
    entries: list[tuple[str, pd.DataFrame]],
    output: Path,
    sort_by_intensity: bool = False,
    group_by_category: bool = False,
) -> None:
    """Save one booktabs table of per-iteration statistics per entry as a document.

    Each frame must stem from a single run. The throughput is that of the
    underlying matmul alone, 2*M*N*K flops over the mean runtime, so it ignores
    the epilogue.
    """
    for label, frame in entries:
        missing = [
            column
            for column in (*TABLE_TIME_COLUMNS, "time_rel_std_pct")
            if column not in frame
        ]
        if missing:
            raise ValueError(f"{label}: CSV lacks the statistics columns {missing}")
    ordered = order_kernels(
        entries[0][1], "input_values", sort_by_intensity, group_by_category
    )
    kernels = ordered["kernel_name"]
    categories = list(KERNEL_CATEGORIES)
    # N and K get a common width, so that the multiplication signs line up.
    nk_digits = max(
        len(str(size // 1024))
        for _, *nk in ordered["input_values"].map(matmul_dims)
        for size in nk
    )

    def number(value: float, digits: int) -> str:
        # Backends that do not time individual iterations leave the cells empty.
        return "--" if pd.isna(value) else f"{value:.{digits}f}"

    def dim(size: int, digits: int = 0) -> str:
        # In units of 1024; non-multiples as a slanted fraction, e.g. 1/8 for 128.
        units = Fraction(size, 1024)
        if units.denominator == 1:
            text = str(units.numerator)
            return "\\phantom{0}" * (digits - len(text)) + text
        return f"{{}}^{{{units.numerator}}}\\!/\\!_{{{units.denominator}}}"

    lines = []
    for label, frame in entries:
        lines += [
            "\\begin{table}[p]",
            "\\centering",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{4pt}",
            f"\\caption{{{_tex_escape(label)}. Runtime statistics over the timed "
            "iterations of a single run; throughput of the $2MNK$ matmul flops at "
            "the mean runtime.}",
            "\\begin{tabular}{@{}lrrrrrrrr@{}}",
            "\\toprule",
            " & $M \\times N \\times K$ & \\multicolumn{4}{c}{Runtime [$\\mu$s]} "
            "& \\multicolumn{2}{c}{Std} & Matmul \\\\",
            "\\cmidrule(lr){3-6} \\cmidrule(lr){7-8}",
            "Model & [$\\times 1024$] & Mean & Min & Max & Median & "
            "[$\\mu$s] & [\\%] & GFLOPS \\\\",
            "\\midrule",
        ]
        previous = None
        for kernel, row in frame.set_index("kernel_name").loc[kernels].iterrows():
            if group_by_category and (category := kernel_category(kernel)) != previous:
                if previous is not None:
                    lines.append("\\addlinespace")
                name = _tex_escape(categories[category])
                lines.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{name}}}}} \\\\")
                previous = category
            m, n, k = matmul_dims(row["input_values"])
            gflops = 2.0 * m * n * k / (row["time_us"] * 1e-6) / 1e9
            cells = [
                _tex_escape(kernel.removesuffix(".yaml")),
                f"${dim(m)}{{\\times}}{dim(n, nk_digits)}{{\\times}}"
                f"{dim(k, nk_digits)}$",
                *(number(row[column], 0) for column in TABLE_TIME_COLUMNS),
                number(row["time_rel_std_pct"], 2),
                number(gflops, 0),
            ]
            lines.append(" & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    _write_latex_document(output, lines)


def write_excel(run_dirs: list[Path], filenames: list[str], output: Path) -> None:
    """Save the CSVs as an Excel workbook with one sheet per run directory.

    Each sheet stacks the CSVs of its directory in the given order.
    """
    sheets = {}
    for directory in run_dirs:
        frames = []
        for filename in filenames:
            frame = pd.read_csv(directory / filename)
            frame = frame.drop(columns=EXCEL_DROPPED_COLUMNS, errors="ignore")
            frame.insert(0, "backend", Path(filename).stem)
            frames.append(frame)
        # Resolved, so that "." yields the name of the current directory.
        name = re.sub(r"[\\/*?:\[\]]", "_", directory.resolve().name)
        name = name[:31].strip("'") or "Run"
        if any(name.casefold() == existing.casefold() for existing in sheets):
            raise ValueError(
                f"{directory}: normalized Excel sheet name {name!r} is not unique"
            )
        sheets[name] = pd.concat(frames, ignore_index=True)
    with pd.ExcelWriter(output) as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    print(f"Saved {output}")


def plot_speedups(
    torch: pd.DataFrame,
    inductor: pd.DataFrame,
    backends: list[tuple[str, pd.DataFrame]],
    output: str | Path = "speedup.png",
    sort_by_intensity: bool = False,
    gflops_series: int | None = None,
    group_by_category: bool = False,
    hero_output: str | Path | None = None,
) -> None:
    """Plot speedups over torch eager for Torch Inductor and the given backends.

    Each dataframe must have unique kernel_name values. If gflops_series is given,
    the bars of that series (0 for Torch Inductor, N for the N-th backend) are
    labelled with their absolute GFLOPS. If hero_output is given, the same speedups
    are also saved there as a single-column line chart.
    """
    if len(backends) > len(BACKEND_COLORS):
        raise ValueError(f"At most {len(BACKEND_COLORS)} backends are supported.")

    # Align all frames on kernel_name so the bars line up per kernel.
    merged = torch.add_suffix("_torch").rename(
        columns={"kernel_name_torch": "kernel_name"}
    )
    for suffix, frame in (
        ("inductor", inductor),
        *((str(i), frame) for i, (_, frame) in enumerate(backends)),
    ):
        if torch.empty or frame.empty:
            raise ValueError(f"{suffix}: cannot compare empty kernel sets")
        if set(frame["kernel_name"]) != set(torch["kernel_name"]):
            raise ValueError(f"{suffix}: kernel set differs from Torch eager")
        merged = merged.merge(
            frame.add_suffix(f"_{suffix}"),
            left_on="kernel_name",
            right_on=f"kernel_name_{suffix}",
        )
        for column in ("input_values", "dtype"):
            expected = merged[f"{column}_torch"]
            actual = merged[f"{column}_{suffix}"]
            if column == "input_values":
                expected = expected.map(json.loads)
                actual = actual.map(json.loads)
            if not actual.equals(expected):
                raise ValueError(f"{suffix}: {column} differs from Torch eager")

    merged = order_kernels(
        merged, "input_values_torch", sort_by_intensity, group_by_category
    )

    def speedup_and_gflops(time_us: pd.Series) -> tuple[pd.Series, pd.Series]:
        # Speedup over torch-eager = torch runtime / candidate runtime.
        speedup = merged["time_us_torch"] / time_us
        # Absolute throughput from the torch flop count divided by the runtime.
        gflops = merged["flops_torch"] / (time_us * 1e-6) / 1e9
        return speedup, gflops

    inductor_time_us = merged["time_us_inductor"]
    series = [(*speedup_and_gflops(inductor_time_us), "Torch Inductor", INDUCTOR_COLOR)]
    for i, ((label, _), color) in enumerate(zip(backends, BACKEND_COLORS)):
        series.append((*speedup_and_gflops(merged[f"time_us_{i}"]), label, color))

    kernels = merged["kernel_name"].tolist()
    labels = [f"{name[: name.find('_')]}" for name in kernels]
    x = np.arange(len(kernels))

    group_width = 0.8
    bar_width = group_width / len(series)

    fig, ax = plt.subplots(figsize=(24, 8))
    gflops_labels = []
    for i, (values, gflops, label, color) in enumerate(series):
        y = np.asarray(values, dtype=float)
        offsets = x - group_width / 2 + (i + 0.5) * bar_width
        bars = ax.bar(offsets, y, width=bar_width, label=label, color=color)
        if i != gflops_series:
            continue
        luminance = np.dot(to_rgb(color), (0.299, 0.587, 0.114))
        for bar, xi, yi, gi in zip(bars, offsets, y, np.asarray(gflops, dtype=float)):
            text = ax.annotate(
                f"{gi:.0f}",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, -3),
                # Aligned before rotating, to centre the digits without descender.
                ha="right",
                va="center_baseline",
                rotation=90,
                rotation_mode="anchor",
                fontsize=9,
                color="black" if luminance > 0.5 else "white",
            )
            gflops_labels.append((bar, text))

    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    for xi in x[::2]:
        ax.axvspan(xi - 0.5, xi + 0.5, color="black", alpha=0.05, linewidth=0)
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylabel("Speedup over Torch eager execution", fontsize=20)
    ax.set_xlabel("KernelBench - level2/...", fontsize=20)
    if group_by_category:
        category_of_kernel = merged["kernel_name"].map(kernel_category).to_numpy()
        for index, name in enumerate(KERNEL_CATEGORIES):
            members = x[category_of_kernel == index]
            if not members.size:
                continue
            if members[0] > x[0]:
                ax.axvline(members[0] - 0.5, color="black", linewidth=1.5)
            ax.annotate(
                name,
                (members.mean(), 0),
                xycoords=ax.get_xaxis_transform(),
                textcoords="offset points",
                xytext=(0, -32),
                ha="center",
                va="top",
                fontsize=16,
                fontweight="bold",
            )
        # Make room for the category names between tick labels and axis label.
        ax.xaxis.labelpad = 34
    ax.legend(fontsize=16)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 5, 10]))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", linestyle="-", alpha=0.4)
    ax.grid(axis="y", which="minor", linestyle=":", alpha=0.3)

    fig.tight_layout()
    # Labels that do not fit inside their bar go on top of it instead.
    renderer = fig.canvas.get_renderer()
    for bar, text in gflops_labels:
        if text.get_window_extent(renderer).y0 < bar.get_window_extent(renderer).y0:
            text.set(ha="left", color=bar.get_facecolor())
            text.xyann = (0, 3)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output}")

    if hero_output is None:
        return

    # One column of a two-column paper, so the fonts are not scaled down in print.
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    for values, _, label, color in series:
        ax.plot(x, np.asarray(values, dtype=float), label=label, color=color)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.margins(x=0)
    ax.set_ylim(bottom=0)
    ax.set_xticks(x)
    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("Selected KernelBench level2 kernels", fontsize=8)
    ax.set_ylabel("Speedup over Torch eager", fontsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", alpha=0.4)
    if group_by_category:
        for index, name in enumerate(KERNEL_CATEGORIES):
            members = x[category_of_kernel == index]
            if not members.size:
                continue
            if members[0] > x[0]:
                ax.axvline(members[0] - 0.5, color="black", linewidth=0.8)
            # The outer categories end at the axis limits, not half a kernel beyond.
            left = max(members[0] - 0.5, x[0])
            right = min(members[-1] + 0.5, x[-1])
            ax.annotate(
                CATEGORY_ABBREVIATIONS[name],
                ((left + right) / 2, 0),
                xycoords=ax.get_xaxis_transform(),
                textcoords="offset points",
                xytext=(0, -4),
                ha="center",
                va="top",
                fontsize=8,
            )
        # Make room for the category labels above the axis label.
        ax.xaxis.labelpad = 11
    # Above the axes, since the lines leave no free corner at this size.
    ax.legend(
        fontsize=7,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        handlelength=1.5,
        columnspacing=1.0,
    )

    fig.tight_layout(pad=0.3)
    fig.savefig(hero_output, dpi=300)
    plt.close(fig)
    print(f"Saved {hero_output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average repeated DSL benchmark runs and plot speedups."
    )
    parser.add_argument(
        "directories",
        nargs="*",
        type=Path,
        default=[Path(".")],
        help="Run directories containing the backend CSVs (default: current directory).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("dsl-paper-eval-speedup.png"),
        help="Output image path (default: %(default)s).",
    )
    parser.add_argument(
        "-b",
        "--backend",
        action="append",
        type=parse_backend,
        metavar="LABEL:CSV",
        default=[],
        help="Backend to plot next to Torch Inductor, as plot label and CSV filename "
        "(e.g. 'Triton CPU:triton.csv'). Repeatable.",
    )
    parser.add_argument(
        "-s",
        "--sort",
        action="store_true",
        help="Sort kernels by arithmetic intensity instead of CSV order.",
    )
    parser.add_argument(
        "-a",
        "--absolute",
        type=int,
        metavar="N",
        help="Label the bars of the N-th backend (in -b order, 0 for Torch Inductor) "
        "with their absolute throughput in GFLOPS.",
    )
    parser.add_argument(
        "-c",
        "--categories",
        action="store_true",
        help="Group kernels by operation category ("
        + ", ".join(name.lower() for name in KERNEL_CATEGORIES)
        + ").",
    )
    parser.add_argument(
        "--hero",
        type=Path,
        metavar="PATH",
        help="Also save a single-column line chart of the speedups to PATH.",
    )
    parser.add_argument(
        "-t",
        "--table",
        type=Path,
        metavar="PATH",
        help="Also save the full results of a single run directory, one table per "
        "backend, as a standalone LaTeX document to PATH.",
    )
    parser.add_argument(
        "-g",
        "--geomean",
        type=Path,
        metavar="PATH",
        help="Also save the geomean speedups of the backends over Torch eager and "
        "Torch Inductor as a standalone LaTeX document to PATH.",
    )
    parser.add_argument(
        "-x",
        "--excel",
        type=Path,
        metavar="PATH",
        help="Also save the Torch and backend CSVs as an Excel workbook with one "
        "sheet per run directory to PATH.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print the statistics of the per-run mean runtimes across the runs.",
    )
    args = parser.parse_args()
    if args.table and len(args.directories) > 1:
        # The CSVs lack the data to merge per-iteration statistics across runs.
        parser.error("--table requires a single run directory")
    if args.absolute is not None and not 0 <= args.absolute <= len(args.backend):
        parser.error(f"--absolute must be between 0 and {len(args.backend)}")
    warnings: list[str] = []
    try:
        torch, compile_, compile_max_at = (
            average_runs(args.directories, filename, warnings, args.verbose)
            for filename in (TORCH_CSV, COMPILE_CSV, COMPILE_MAX_AT_CSV)
        )
        # Best of compile and compile max-autotune per kernel.
        inductor = best_of(compile_, compile_max_at)
        backend_frames = [
            (label, average_runs(args.directories, filename, warnings, args.verbose))
            for label, filename in args.backend
        ]
        plot_speedups(
            torch,
            inductor,
            backend_frames,
            output=args.output,
            sort_by_intensity=args.sort,
            gflops_series=args.absolute,
            group_by_category=args.categories,
            hero_output=args.hero,
        )
        if args.geomean:
            write_geomean_table(torch, inductor, backend_frames, args.geomean)
        if args.table:
            write_latex_tables(
                [
                    ("Torch eager", torch),
                    ("Torch Inductor", inductor),
                    *backend_frames,
                ],
                args.table,
                sort_by_intensity=args.sort,
                group_by_category=args.categories,
            )
        if args.excel:
            write_excel(
                args.directories,
                [
                    TORCH_CSV,
                    COMPILE_CSV,
                    COMPILE_MAX_AT_CSV,
                    *(filename for _, filename in args.backend),
                ],
                args.excel,
            )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if warnings:
        # Keeps the block last when stdout and stderr share a pipe or file.
        sys.stdout.flush()
        print("\nWarnings:", file=sys.stderr)
        for warning in warnings:
            print(f"  {warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
