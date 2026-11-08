"""Plots of the CSP decomposition sweep `csp_test1` (8 instances: s200-s500, full dataset and no-USA).

Reads results/runs/csp/pipeline_results/csp_test1/ (sweep_summary.json and each instance's
decomposition_summary.csv) and writes four PNGs next to this script:
    1_runtime.png               CSP notebook runtime per instance, full dataset vs no-USA (line)
    2_exact_components.png      connected components decomposed exactly per instance (line)
    3_incomplete_leaves.png     sub-problems left undecomposed (incomplete leaves), bars grouped by dataset
    4_largest_component.png     largest connected component (qubits after dominance), bars grouped by dataset
    5_component_sizes.png       distribution of connected-component sizes, one panel per instance size

Run from anywhere:  python tmp/decomposition_sweep_plots/plot_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
SWEEP_DIR = PROJECT_ROOT / "results" / "runs" / "csp" / "pipeline_results" / "csp_test1"

# reference palette (dataviz skill): categorical slots 1-2, light surface, text and grid tokens
DATASETS = {  # key prefix -> (label, colour)
    "": ("Full dataset", "#2a78d6"),
    "no-usa_": ("No-USA", "#eb6834"),
}
SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SATELLITES = [200, 300, 400, 500]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY, "axes.titlecolor": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY, "text.color": TEXT_PRIMARY,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def load_sweep() -> pd.DataFrame:
    """One row per instance: dataset key, satellites and the plotted metrics."""
    rows = []
    for run in json.loads((SWEEP_DIR / "sweep_summary.json").read_text(encoding="utf-8")):
        name = run["dataset"]
        prefix = "no-usa_" if name.startswith("no-usa_") else ""
        decomposition = pd.read_csv(SWEEP_DIR / name / "decomposition_summary.csv")
        largest = decomposition.loc[decomposition["input_qubits"].idxmax()]
        # slots after dominance held by the components decomposed exactly (no incomplete leaf)
        exact_slots = decomposition.loc[decomposition["status"] == "complete", "input_qubits"].sum()
        rows.append({
            "instance": name,
            "dataset": prefix,
            "satellites": int(name.removeprefix(prefix).split("_")[0][1:]),
            "runtime_min": run["runtime_seconds"] / 60,
            "components": run["graph"]["num subgraphs"],
            "exact_components": run["other"]["Components decomposed exactly"],
            "exact_slot_pct": 100 * exact_slots / decomposition["input_qubits"].sum(),
            "incomplete_leaves": run["qubo split"]["incomplete leaves"],
            "largest_component_qubits": int(largest["input_qubits"]),
            "largest_component_satellites": int(largest["satellites"]),
        })
    return pd.DataFrame(rows).sort_values(["dataset", "satellites"]).reset_index(drop=True)


def line_plot(data: pd.DataFrame, column: str, title: str, ylabel: str, path: Path,
              log: bool = False, value_format=str, label_points: bool = False,
              point_label=None, all_above: bool = False, subtitle: str | None = None) -> None:
    """One line per dataset over the number of satellites, labelled at its last point and in a legend.
    With `label_points`, every point also shows a label (`point_label(row)`, by default its value):
    above it for the higher of the two series at that x and below it for the lower one, so the
    labels of the two series never overlap, or above every point with `all_above`."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    highest = data.groupby("satellites")[column].max()
    point_label = point_label or (lambda row: value_format(row[column]))
    for prefix, (label, colour) in DATASETS.items():
        series = data[data["dataset"] == prefix].sort_values("satellites")
        ax.plot(series["satellites"], series[column], color=colour, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
        last = series.iloc[-1]
        end_label = label if label_points else f"{label}  {value_format(last[column])}"
        ax.annotate(end_label, (last["satellites"], last[column]),
                    xytext=(10, 0), textcoords="offset points", va="center", color=TEXT_SECONDARY, fontsize=9)
        if label_points:
            for _, row in series.iterrows():
                above = all_above or row[column] >= highest[row["satellites"]]
                ax.annotate(point_label(row), (row["satellites"], row[column]),
                            xytext=(0, 9 if above else -9), textcoords="offset points", ha="center",
                            va="bottom" if above else "top", color=TEXT_PRIMARY, fontsize=8.5, zorder=4,
                            # surface-coloured backing: a line passing under the label stays behind the text
                            bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none"))
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        # room above the highest and below the lowest point for their value labels (about a third of a decade)
        ax.set_ylim(data[column].min() / 2.2, data[column].max() * 2.2)
    elif label_points:
        ax.margins(y=0.15)   # room above the highest point for its label
    ax.set_xticks(SATELLITES)
    ax.set_xlim(170, 610)   # room for the end labels
    ax.set(xlabel="Satellites in the instance (50 ground stations)", ylabel=ylabel)
    if subtitle:
        ax.set_title(title, pad=22)
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=TEXT_SECONDARY, va="bottom")
    else:
        ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def grouped_bars(data: pd.DataFrame, column: str, title: str, ylabel: str, path: Path,
                 label_format=str) -> None:
    """Bars grouped by dataset (one group per dataset, one bar per instance size), value on each bar."""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    width, gap = 0.78, 1.2   # bar width (the rest of each slot is the gap between bars) and space between groups
    ticks, tick_labels = [], []
    for group, (prefix, (label, colour)) in enumerate(DATASETS.items()):
        series = data[data["dataset"] == prefix].sort_values("satellites")
        positions = [group * (len(SATELLITES) + gap) + i for i in range(len(series))]
        bars = ax.bar(positions, series[column], width=width, color=colour, linewidth=0, zorder=3)
        for bar, (_, row) in zip(bars, series.iterrows()):
            ax.annotate(label_format(row), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        color=TEXT_SECONDARY, fontsize=8.5)
        ticks += positions
        tick_labels += [f"s{n}" for n in series["satellites"]]
        ax.text(sum(positions) / len(positions), -0.13, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=10.5, fontweight="bold", color=TEXT_PRIMARY)
    ax.set_xticks(ticks, tick_labels)
    ax.tick_params(axis="x", length=0)
    ax.set(title=title, ylabel=ylabel)
    ax.margins(y=0.12)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def load_component_sizes() -> pd.DataFrame:
    """One row per connected component of every instance: dataset key, satellites, size in slots
    after dominance (= domain-wall qubits), and the sweep's QPU terminal limit."""
    rows = []
    for run in json.loads((SWEEP_DIR / "sweep_summary.json").read_text(encoding="utf-8")):
        name = run["dataset"]
        prefix = "no-usa_" if name.startswith("no-usa_") else ""
        satellites = int(name.removeprefix(prefix).split("_")[0][1:])
        limit = json.loads((SWEEP_DIR / name / "sweep_config.json").read_text(encoding="utf-8"))["max_qpu_terminal_nodes"]
        for size in pd.read_csv(SWEEP_DIR / name / "decomposition_summary.csv")["input_qubits"]:
            rows.append({"dataset": prefix, "satellites": satellites, "size": int(size), "qpu_limit": limit})
    return pd.DataFrame(rows)


def component_size_distribution(sizes: pd.DataFrame, path: Path) -> None:
    """Number of components per size bin (bins doubling in size: 1, 2-3, 4-7, ...), full dataset and
    no-USA side by side in each bin, one panel per instance size; both axes logarithmic."""
    edges = 2.0 ** np.arange(0, 13)   # 1 ... 4096 slots: the largest component has 3,216
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharex=True, sharey=True)
    for ax, satellites in zip(axes.flat, SATELLITES):
        largest = []
        for side, (prefix, (label, colour)) in enumerate(DATASETS.items()):
            values = sizes[(sizes["dataset"] == prefix) & (sizes["satellites"] == satellites)]["size"]
            counts, _ = np.histogram(values, bins=edges)
            # each bin [a, 2a) is split in two halves on the log axis, with a small gap around each bar
            left = edges[:-1] * 2 ** (0.5 * side + 0.04)
            right = edges[:-1] * 2 ** (0.5 * side + 0.46)
            # bars rise from 0.7 (the log axis has no zero), so their height is count - 0.7: tops sit at the count
            ax.bar(left, np.where(counts > 0, counts - 0.7, np.nan), width=right - left, align="edge",
                   bottom=0.7, color=colour, linewidth=0, zorder=3)
            largest.append(f"{label}: {values.max():,}")
        limit = sizes.loc[sizes["satellites"] == satellites, "qpu_limit"].iloc[0]
        ax.axvline(limit, color=TEXT_SECONDARY, linestyle="--", linewidth=1, zorder=2)
        ax.set_title(f"{satellites} satellites", fontsize=11)
        ax.text(0.98, 0.95, "Largest component\n" + "\n".join(largest), transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5, color=TEXT_SECONDARY, linespacing=1.4)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlim(1, 4096)
        ax.set_ylim(0.7, 400)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xticks([1, 4, 16, 64, 256, 1024, 4096])
    for ax in axes[1]:
        ax.set_xlabel("Component size [slots after dominance = qubits], bins doubling")
    for ax in axes[:, 0]:
        ax.set_ylabel("Number of components")
    handles = [Patch(color=colour, label=label) for label, colour in DATASETS.values()]
    handles.append(Line2D([], [], color=TEXT_SECONDARY, linestyle="--", linewidth=1, label="QPU terminal limit (200 qubits)"))
    fig.legend(handles=handles, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 0.985))
    fig.suptitle("Distribution of connected-component sizes", x=0.01, y=0.985, ha="left",
                 fontsize=12, fontweight="bold", color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=3)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    data = load_sweep()
    data.to_csv(HERE / "sweep_plot_data.csv", index=False)
    print(data.to_string(index=False))

    line_plot(data, "runtime_min", "CSP pipeline runtime per instance", "Runtime [min, log scale]",
              HERE / "1_runtime.png", log=True, label_points=True,
              value_format=lambda v: f"{v * 60:.0f} s" if v < 1 else f"{v:.1f} min")
    line_plot(data, "exact_components", "Connected components decomposed exactly",
              "Components with no incomplete leaf", HERE / "2_exact_components.png", value_format=lambda v: f"{v:.0f}",
              label_points=True, all_above=True, point_label=lambda row: f"{row['exact_slot_pct']:.1f}%",
              subtitle="Labels: share of the slots left after dominance that these components contain")
    grouped_bars(data, "incomplete_leaves", "Sub-problems left undecomposed (incomplete leaves)",
                 "Incomplete leaves", HERE / "3_incomplete_leaves.png",
                 label_format=lambda row: f"{row['incomplete_leaves']:,}")
    grouped_bars(data, "largest_component_qubits", "Largest connected component after dominance",
                 "Size [qubits = slots, domain-wall]", HERE / "4_largest_component.png",
                 label_format=lambda row: f"{row['largest_component_qubits']:,}\n({row['largest_component_satellites']} sat.)")
    sizes = load_component_sizes()
    sizes.to_csv(HERE / "component_sizes.csv", index=False)
    component_size_distribution(sizes, HERE / "5_component_sizes.png")
    print(f"Saved 5 plots to {HERE}")


if __name__ == "__main__":
    main()
