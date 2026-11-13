"""COP constraint graph of a slot dataset, before and after the exact dominance reduction.

The COP model is the one of results/02.B_CSP_pipeline.ipynb (src/CSP.py): one variable per
satellite, whose domain is its candidate slots; two satellites are constrained when some pair of
their slots conflicts on the same antenna (overlap, or less than the reconfiguration time between
them). The "at most one slot per satellite" rule lives inside each variable, so it adds no edges.

The figure has four panels:
    top left      constraint graph of the input COP (edges faint: it is dense)
    top right     constraint graph after dominance_reduction (src/CSP_decomposition.py), the problem
                  the decomposition actually works on; nodes coloured by connected component (the
                  largest components get their own colour, the others grey), edge width by the
                  number of conflicting slot pairs
    bottom left   degree distribution before / after dominance
    bottom right  size of the connected components after dominance

Both graphs are drawn with the same node positions (spring layout of the reduced graph), so the
right panel shows what is left of the left one. With --store-csv the script also writes, next to
the PNG, <stem>_cop_nodes.csv (per satellite: domain size, degree, component, before/after) and
<stem>_cop_edges.csv (reduced constraint graph, with the number of conflicting slot pairs).

The input is the path to one slot CSV.

Example:
    .venv/Scripts/python.exe scripts/visualization/cop_conflict_graph.py data/test/slots_s500_g50.csv
    .venv/Scripts/python.exe scripts/visualization/cop_conflict_graph.py data/test/slots_s500_g50.csv --store-csv
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from CSP import CSP, CSP_Variable, Timeslot  # noqa: E402
from CSP_decomposition import dominance_reduction  # noqa: E402

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / f"plot_{Path(__file__).stem}"  # scripts/visualization/plot_<script name>
COLUMNS = ["slot_id", "satellite_name", "antenna_id", "start_time_hours", "end_time_hours",
           "reconfiguration_time_hours", "weight"]
TOP_COMPONENTS = 10  # components with their own colour in the reduced-graph panel


def build_csp(slots: pd.DataFrame) -> CSP:
    """COP of the slots, built as in the CSP pipeline notebook (section 3)."""
    csp = CSP()
    for name, rows in slots.groupby("satellite_name", sort=True):
        var = CSP_Variable(name=name)
        for row in rows.itertuples(index=False):
            var.add_value(Timeslot.from_hours(row.slot_id, row.antenna_id, row.reconfiguration_time_hours,
                                              row.start_time_hours, row.end_time_hours))
        csp.add_var(var)
    return csp


def constraint_graph(csp: CSP) -> nx.Graph:
    """Satellites as nodes (with domain size), constrained pairs as edges with the number of
    conflicting slot pairs still in the domains."""
    pairs: Counter = Counter()
    for var in csp.vars:
        for slot in var.domain:
            for other in csp.conflicting_values(var, slot):
                owner = csp.index.owner[other]
                if var.name < owner:
                    pairs[var.name, owner] += 1
    graph = nx.Graph()
    graph.add_nodes_from((v.name, {"domain": len(v.domain)}) for v in csp.vars)
    graph.add_edges_from((a, b, {"pairs": n}) for (a, b), n in pairs.items())
    return graph


def plot(full: nx.Graph, reduced: nx.Graph, title: str, output_path: Path, dpi: int, seed: int) -> None:
    components = sorted(nx.connected_components(reduced), key=len, reverse=True)
    component_of = {n: i for i, c in enumerate(components) for n in c}
    pos = nx.spring_layout(reduced, k=1.5 / np.sqrt(max(len(reduced), 1)), iterations=200, seed=seed)

    palette = plt.get_cmap("tab10")
    def node_color(n):
        i = component_of[n]
        return palette(i % 10) if i < TOP_COMPONENTS and len(components[i]) > 1 else (0.6, 0.6, 0.6, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14), gridspec_kw={"height_ratios": [3, 1]})
    (ax_full, ax_red), (ax_deg, ax_comp) = axes

    # top left: input COP, positions shared with the reduced graph
    nx.draw_networkx_edges(full, pos, ax=ax_full, width=0.3, alpha=min(0.5, 300 / max(full.number_of_edges(), 1)),
                           edge_color="tab:blue")
    nx.draw_networkx_nodes(full, pos, ax=ax_full, node_size=10, node_color="black")
    ax_full.set_title(f"input COP: {full.number_of_nodes()} satellites, {full.number_of_edges()} constraints, "
                      f"density {nx.density(full):.3f}\n"
                      f"{nx.number_connected_components(full)} components, largest "
                      f"{len(max(nx.connected_components(full), key=len))}")

    # top right: after dominance, coloured by component
    if reduced.number_of_edges():
        widths = np.array([d["pairs"] for _, _, d in reduced.edges(data=True)], dtype=float)
        nx.draw_networkx_edges(reduced, pos, ax=ax_red, width=0.4 + 2.0 * widths / widths.max(), alpha=0.5,
                               edge_color=[node_color(u) for u, _ in reduced.edges()])
    sizes = [8 + 3 * np.sqrt(reduced.nodes[n]["domain"]) for n in reduced]
    nx.draw_networkx_nodes(reduced, pos, ax=ax_red, node_size=sizes, node_color=[node_color(n) for n in reduced],
                           edgecolors="black", linewidths=0.2)
    singletons = sum(1 for c in components if len(c) == 1)
    ax_red.set_title(f"after dominance: {sum(d['domain'] for _, d in reduced.nodes(data=True))} slots left, "
                     f"{reduced.number_of_edges()} constraints\n{len(components)} components "
                     f"({singletons} single satellites), largest {len(components[0])}  "
                     f"(node size ~ domain size, edge width ~ conflicting slot pairs)")
    for ax in (ax_full, ax_red):
        ax.set_axis_off()

    # bottom left: degree distributions
    deg_full = [d for _, d in full.degree()]
    deg_red = [d for _, d in reduced.degree()]
    top = max(deg_full + [1])
    bins = np.arange(0, top + 2) - 0.5 if top < 60 else np.linspace(0, top + 1, 41)
    ax_deg.hist(deg_full, bins=bins, alpha=0.6, label=f"input (mean {np.mean(deg_full):.1f})")
    ax_deg.hist(deg_red, bins=bins, alpha=0.6, label=f"after dominance (mean {np.mean(deg_red):.1f})")
    ax_deg.set(xlabel="constraints per satellite (degree)", ylabel="satellites", yscale="log")
    ax_deg.legend()

    # bottom right: component sizes after dominance
    size_counts = Counter(len(c) for c in components)
    xs = sorted(size_counts)
    ax_comp.bar([str(x) for x in xs], [size_counts[x] for x in xs],
                color=[palette(0) if x == len(components[0]) and x > 1 else "tab:gray" for x in xs])
    ax_comp.set(xlabel="component size (satellites)", ylabel="components", yscale="log",
                title="connected components after dominance")
    if len(xs) > 25:
        ax_comp.tick_params(axis="x", labelrotation=90, labelsize=7)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def draw_dataset(slots_path: Path, out_dir: Path, dpi: int, seed: int, store_csv: bool = False) -> None:
    print(f"Reading {slots_path}")
    slots = pd.read_csv(slots_path, usecols=COLUMNS)
    started = time.perf_counter()
    csp = build_csp(slots)
    full = constraint_graph(csp)
    removed = dominance_reduction(csp, dict(zip(slots["slot_id"], slots["weight"])))
    reduced = constraint_graph(csp)
    print(f"  {len(slots)} slots, {full.number_of_nodes()} satellites, {full.number_of_edges()} constraints; "
          f"dominance removed {removed} slots -> {reduced.number_of_edges()} constraints, largest component "
          f"{len(max(nx.connected_components(reduced), key=len))} ({time.perf_counter() - started:.1f} s)")

    stem = slots_path.stem
    title = f"{stem}: COP constraint graph (satellites = variables)"
    plot(full, reduced, title, out_dir / f"{stem}_cop_graph.png", dpi, seed)
    if not store_csv:
        print(f"  saved {stem}_cop_graph.png to {out_dir}")
        return

    component_of = {n: i for i, c in enumerate(sorted(nx.connected_components(reduced), key=len, reverse=True))
                    for n in c}
    pd.DataFrame([{"satellite_name": n, "domain_input": full.nodes[n]["domain"], "degree_input": full.degree(n),
                   "domain_reduced": reduced.nodes[n]["domain"], "degree_reduced": reduced.degree(n),
                   "component": component_of[n]} for n in full]).to_csv(out_dir / f"{stem}_cop_nodes.csv", index=False)
    pd.DataFrame([{"satellite_a": a, "satellite_b": b, "conflicting_slot_pairs": d["pairs"]}
                  for a, b, d in reduced.edges(data=True)]).to_csv(out_dir / f"{stem}_cop_edges.csv", index=False)
    print(f"  saved {stem}_cop_graph.png, {stem}_cop_nodes.csv and {stem}_cop_edges.csv to {out_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("slots", type=Path, help="path to the slot CSV")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="default: scripts/visualization/plot_cop_conflict_graph/")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0, help="seed of the spring layout")
    parser.add_argument("--store-csv", action="store_true",
                        help="also write <stem>_cop_nodes.csv and <stem>_cop_edges.csv next to the PNG")
    args = parser.parse_args(argv)
    if not args.slots.is_file():
        parser.error(f"slot CSV not found: {args.slots}")
    draw_dataset(args.slots, args.out_dir, args.dpi, args.seed, args.store_csv)


if __name__ == "__main__":
    main()
