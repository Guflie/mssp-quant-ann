"""COP -> QUBO translation of a stored component, side by side, coloured by COP variable.

Reads a <stem>_components.pkl written by scripts/cop_components_export.py (components after
dominance, with their QUBO) and draws, for each requested component, one figure with two panels:

    left   COP constraint graph: one node per satellite (COP variable, area ~ domain size), one
           edge per constraint (width ~ number of conflicting slot pairs)
    right  QUBO graph of the same component: one node per binary variable, one edge per coupler
           (non-zero off-diagonal coefficient). The bits of a satellite sit on a ring around the
           satellite's position in the left panel, in chain order, and take its colour, so every
           COP variable can be followed into the logical variables it becomes. Couplers inside a
           satellite (domain-wall chain, or one-hot pairs) are drawn in the satellite colour, the
           conflict couplers between satellites in grey

Colours: the 8 slots of a colour-blind-checked categorical palette, assigned by graph colouring,
so that two satellites sharing a constraint, or drawn close to each other, never share a colour
(identity is carried by colour plus position plus label, not by colour alone: 42 unique hues
could not be told apart). The colouring is deterministic, so qubo_embedding.py, which copies it,
gives every satellite the same colour.

Self-contained: besides src/ (needed to unpickle the components) it depends on no other script.

Output: <stem>_<component_id>_<encoding>_cop_qubo.png in --out-dir
(default scripts/visualization/plot_cop_qubo_translation/).

Example:
    .venv/Scripts/python.exe scripts/visualization/cop_qubo_translation.py "scripts/cop_components_export/starlink_scheduling_slots_s500_g50_ deg50_seed65416843_components.pkl"
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))     # the pickled components are src/CSP objects

DEFAULT_OUT_DIR = HERE / f"plot_{Path(__file__).stem}"  # scripts/visualization/plot_<script name>

# categorical palette (dataviz reference palette, light mode), in its validated order
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
NEUTRAL_EDGE = "#8f8e88"

# figure size: inches per layout unit (0.022 between consecutive bits -> ~0.1 in on paper), clamped
INCH_PER_UNIT = 4.5
MIN_PANEL = (8.0, 7.5)      # inches, width x height of one panel at least
MAX_PANEL = 60.0            # inches, longer side of one panel at most
MAX_PIXELS = 30000          # longer side of the PNG at most (dpi lowered beyond it)
TOP_MARGIN, BOTTOM_MARGIN = 1.3, 0.8     # inches for titles and legends
AUTO_LABELS_ALL = 80        # name every satellite up to this many
AUTO_LABELS_TOP = 40        # otherwise name only the largest domains
NODE_FILL = 0.8             # COP node diameter, as a fraction of the satellite's ring of bits


# ---------------------------------------------------------------- data
def load_export(pkl_path: Path) -> dict[str, Any]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def short_name(name: str) -> str:
    return name.replace("STARLINK-", "").replace(" [DTC]", "-DTC")


def bit_owner(component, index_map: dict[str, tuple]) -> dict[int, str]:
    """{bit: satellite}: the first bit of each slot's indicator is the slot's own bit."""
    return {index_map[t.slot_id][0]: v.name for v in component.vars for t in v.domain}


def cop_graph(component) -> nx.Graph:
    """Satellites as nodes (with domain size), constraints as edges with the conflicting slot pairs."""
    pairs: Counter = Counter()
    for v in component.vars:
        for t in v.domain:
            for c in component.conflicting_values(v, t):
                u = component.index.owner[c]
                if v.name < u:
                    pairs[(v.name, u)] += 1
    G = nx.Graph()
    # sorted: domains are sets of Timeslot, whose order changes between processes (hash seed)
    G.add_nodes_from((v.name, {"domain": len(v.domain)}) for v in sorted(component.vars, key=lambda v: v.name))
    G.add_edges_from((a, b, {"pairs": n}) for (a, b), n in sorted(pairs.items()))
    return G


def qubo_graph(Q: dict[tuple[int, int], float], owner: dict[int, str]) -> nx.Graph:
    """Bits as nodes, couplers as edges; an edge is "encoding" inside a satellite, else "conflict"."""
    G = nx.Graph()
    G.add_nodes_from((b, {"satellite": s}) for b, s in sorted(owner.items()))    # sorted: reproducible
    for (i, j), coeff in sorted(Q.items()):
        if i != j:
            G.add_edge(i, j, coeff=coeff, kind="encoding" if owner[i] == owner[j] else "conflict")
    return G


# ---------------------------------------------------------------- layout and colours (same in qubo_embedding.py)
BIT_SPACING = 0.022     # distance between consecutive bits on a ring (layout units)
MIN_RING = 0.025        # radius of the smallest rings (2-3 bits)
RING_GAP = 0.07         # free space kept between two rings (room for the labels)


def ring_radii(G: nx.Graph) -> dict[str, float]:
    """Radius of each satellite's ring of bits: large enough to keep BIT_SPACING between its bits."""
    return {n: max(MIN_RING, G.nodes[n]["domain"] * BIT_SPACING / (2 * math.pi)) for n in G}


def cop_layout(G: nx.Graph, radius: dict[str, float], seed: int) -> dict[str, np.ndarray]:
    """Spring layout of the constraint graph (in [-1, 1]), then satellites pushed apart until no two
    rings overlap: the large domains get the room their bits need, the graph shape is kept."""
    names = list(G)
    if len(names) == 1:
        return {names[0]: np.zeros(2)}
    start = nx.spring_layout(G, k=2.2 / math.sqrt(len(names)), iterations=300, seed=seed, weight=None)
    P = np.array([start[n] for n in names], dtype=float)
    r = np.array([radius[n] for n in names])
    need = r[:, None] + r[None] + RING_GAP
    np.fill_diagonal(need, 0.0)
    rng = np.random.default_rng(seed)
    for _ in range(2000):
        diff = P[:, None] - P[None]
        dist = np.linalg.norm(diff, axis=-1)
        overlap = np.clip(need - dist, 0.0, None)
        np.fill_diagonal(overlap, 0.0)
        if overlap.max() < 1e-4:
            break
        coincident = (dist < 1e-9) & (overlap > 0)
        if coincident.any():                                # identical positions: no direction to push
            P += rng.normal(scale=1e-3, size=P.shape)
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            unit = np.where(dist[..., None] > 0, diff / dist[..., None], 0.0)
        P += 0.5 * (overlap[..., None] * unit).sum(axis=1)
    return {n: P[i] for i, n in enumerate(names)}


def qubo_layout(owner: dict[int, str], pos: dict[str, np.ndarray], radius: dict[str, float]) -> dict[int, np.ndarray]:
    """Each satellite's bits on a ring around its COP position, in bit (chain) order, starting at the top."""
    bits_of: dict[str, list[int]] = {}
    for b, s in sorted(owner.items()):
        bits_of.setdefault(s, []).append(b)
    layout = {}
    for s, bits in bits_of.items():
        if len(bits) == 1:
            layout[bits[0]] = pos[s].copy()
            continue
        for k, b in enumerate(bits):
            angle = math.pi / 2 - 2 * math.pi * k / len(bits)
            layout[b] = pos[s] + radius[s] * np.array([math.cos(angle), math.sin(angle)])
    return layout


def colour_graph(G: nx.Graph, extra_edges=()) -> tuple[dict[Any, str], int]:
    """Assign palette colours so that adjacent nodes differ (DSATUR greedy colouring). Nodes the
    greedy colouring puts past the palette take the colour least used among their neighbours.
    Returns the colours and the number of adjacent pairs left with the same colour."""
    H = nx.Graph()
    H.add_nodes_from(sorted(G, key=str))
    H.add_edges_from(sorted((tuple(sorted(e, key=str)) for e in [*G.edges, *extra_edges]), key=str))
    order = nx.coloring.greedy_color(H, strategy="DSATUR")
    colour: dict[Any, int] = {}
    for n in sorted(H, key=lambda n: (order[n], str(n))):
        if order[n] < len(PALETTE):
            colour[n] = order[n]
        else:
            used = Counter(colour[m] for m in H[n] if m in colour)
            colour[n] = min(range(len(PALETTE)), key=lambda c: (used[c], c))
    clashes = sum(1 for a, b in H.edges if colour[a] == colour[b])
    return {n: PALETTE[c] for n, c in colour.items()}, clashes


def satellite_colours(G: nx.Graph, pos: dict[str, np.ndarray], radius: dict[str, float]) -> tuple[dict[str, str], int]:
    """Constraint graph plus the pairs of satellites drawn close together (rings almost touching)."""
    names = list(G)
    close = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
             if np.linalg.norm(pos[a] - pos[b]) < 1.5 * (radius[a] + radius[b]) + 2 * RING_GAP]
    return colour_graph(G, close)


# ---------------------------------------------------------------- drawing
def style_axis(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left", pad=8)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_edges(ax, segments, colours, widths, alpha=1.0, zorder=1) -> None:
    if segments:
        ax.add_collection(LineCollection(segments, colors=colours, linewidths=widths, alpha=alpha,
                                         zorder=zorder, capstyle="round"))


def figure_geometry(lo: np.ndarray, hi: np.ndarray, scale: float, dpi: int) -> dict[str, Any]:
    """Panel size grown with the layout, so consecutive bits stay ~INCH_PER_UNIT * BIT_SPACING apart on
    paper whatever the component size (clamped to MIN_PANEL / MAX_PANEL); dpi lowered if the image
    would exceed MAX_PIXELS per side."""
    extent = hi - lo
    ipu = INCH_PER_UNIT * scale
    ipu = min(ipu, MAX_PANEL / extent.max())                       # largest panels capped
    ipu = max(ipu, min(MIN_PANEL[0] / extent[0], MIN_PANEL[1] / extent[1]))   # small components fill the panel
    panel = extent * ipu
    width, height = 2 * panel[0] + 1.2, panel[1] + TOP_MARGIN + BOTTOM_MARGIN
    dpi = min(dpi, int(MAX_PIXELS / max(width, height)))
    return {"inch_per_unit": ipu, "figsize": (width, height), "dpi": dpi}


def plot_component(stem: str, component_id: str, component, qubo: dict, encoding: str,
                   out_path: Path, seed: int, dpi: int, labels: int, scale: float) -> dict[str, Any]:
    owner = bit_owner(component, qubo["index_map"])
    cop = cop_graph(component)
    qg = qubo_graph(qubo["Q"], owner)
    radius = ring_radii(cop)
    pos = cop_layout(cop, radius, seed)
    bit_pos = qubo_layout(owner, pos, radius)
    sat_colour, clashes = satellite_colours(cop, pos, radius)

    encoding_edges = [(i, j) for i, j, d in qg.edges(data=True) if d["kind"] == "encoding"]
    conflict_edges = [(i, j) for i, j, d in qg.edges(data=True) if d["kind"] == "conflict"]

    # identical limits on both panels, so they overlay; the figure grows with the layout
    every = np.array([*pos.values(), *bit_pos.values()])
    pad = 0.06 * max(np.ptp(every[:, 0]), np.ptp(every[:, 1]), 1e-6)
    lo, hi = every.min(axis=0) - pad, every.max(axis=0) + pad
    geometry = figure_geometry(lo, hi, scale, dpi)
    fig, (ax_cop, ax_qubo) = plt.subplots(1, 2, figsize=geometry["figsize"], facecolor=SURFACE)

    # --- left: COP; each satellite as wide as NODE_FILL of its ring of bits on the right (the ring grows
    # with the domain, and rings never overlap, so neither do the nodes)
    max_pairs = max((d["pairs"] for *_, d in cop.edges(data=True)), default=1)
    draw_edges(ax_cop, [(pos[a], pos[b]) for a, b in cop.edges],
               NEUTRAL_EDGE, [0.6 + 3.4 * d["pairs"] / max_pairs for *_, d in cop.edges(data=True)], alpha=0.55)
    names = list(cop)
    node_diameter = {n: 2 * NODE_FILL * radius[n] * geometry["inch_per_unit"] * 72 for n in names}   # points
    ax_cop.scatter([pos[n][0] for n in names], [pos[n][1] for n in names],
                   s=[node_diameter[n] ** 2 for n in names],
                   c=[sat_colour[n] for n in names], edgecolors=SURFACE, linewidths=0.8, zorder=3)

    # --- right: QUBO
    draw_edges(ax_qubo, [(bit_pos[i], bit_pos[j]) for i, j in conflict_edges], NEUTRAL_EDGE, 0.5, alpha=0.35)
    draw_edges(ax_qubo, [(bit_pos[i], bit_pos[j]) for i, j in encoding_edges],
               [sat_colour[owner[i]] for i, _ in encoding_edges], 1.8, zorder=2)
    bits = sorted(owner)
    ax_qubo.scatter([bit_pos[b][0] for b in bits], [bit_pos[b][1] for b in bits], s=22,
                    c=[sat_colour[owner[b]] for b in bits], edgecolors=SURFACE, linewidths=0.6, zorder=3)

    for ax in (ax_cop, ax_qubo):
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])

    # names: every satellite on small components, the largest domains only on large ones (-1 = auto)
    n_labels = len(names) if labels < 0 and len(names) <= AUTO_LABELS_ALL else AUTO_LABELS_TOP if labels < 0 else labels
    labelled = sorted(names, key=lambda n: (-cop.nodes[n]["domain"], n))[:n_labels]
    point = 1 / (72 * geometry["inch_per_unit"])                    # one typographic point, in layout units
    for n in labelled:
        for ax, offset in ((ax_cop, node_diameter[n] / 2 * point + 2 * point), (ax_qubo, radius[n] + 4 * point)):
            ax.annotate(short_name(n), pos[n] + np.array([0, offset]), ha="center", va="bottom",
                        fontsize=6.5, color=TEXT_SECONDARY, zorder=4)

    style_axis(ax_cop, f"COP  ·  {cop.number_of_nodes()} satellites (variables), "
                       f"{cop.number_of_edges()} constraints, {len(bits)} slots in the domains")
    style_axis(ax_qubo, f"QUBO ({encoding})  ·  {len(bits)} binary variables, {qg.number_of_edges()} couplers: "
                        f"{len(encoding_edges)} encoding, {len(conflict_edges)} conflict")

    # one legend per panel, under it: only the symbols drawn in that panel
    legend_cop = [
        Line2D([], [], marker="o", ls="", color=NEUTRAL_EDGE, markersize=9,
               label="satellite (COP variable), as wide as its ring of bits on the right (~ domain size)"),
        Line2D([], [], color=NEUTRAL_EDGE, lw=2.5, alpha=0.6, label="constraint, width ~ conflicting slot pairs"),
    ]
    legend_qubo = [
        Line2D([], [], marker="o", ls="", color=NEUTRAL_EDGE, markersize=5, label="binary variable (one per slot)"),
        Line2D([], [], color=TEXT_SECONDARY, lw=1.8,
               label="encoding coupler, same satellite (" + ("domain-wall chain" if encoding == "domain_wall" else "one-hot pair") + ")"),
        Line2D([], [], color=NEUTRAL_EDGE, lw=0.8, alpha=0.6, label="conflict coupler, two satellites"),
    ]
    # margins in inches, so titles and legends keep their size whatever the figure height
    height = geometry["figsize"][1]
    note = ("colour = satellite: its bits on the right keep its colour and sit around its position "
            "on the left (satellites that share a constraint or sit close never share a colour)")
    if n_labels < len(names):
        note += f"  ·  names of the {n_labels} largest domains only"
    fig.suptitle(f"{stem}  ·  {component_id}: COP → QUBO", fontsize=13, color=TEXT_PRIMARY, x=0.012, ha="left",
                 y=1 - 0.25 / height)
    fig.text(0.012, 1 - 0.62 / height, note, fontsize=8.5, color=TEXT_SECONDARY, ha="left")
    fig.tight_layout(rect=(0, BOTTOM_MARGIN / height, 1, 1 - TOP_MARGIN / height))
    # legends on the same baseline, each centred under its own panel
    for ax, handles in ((ax_cop, legend_cop), (ax_qubo, legend_qubo)):
        box = ax.get_position()
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=((box.x0 + box.x1) / 2, 0.0), ncol=2,
                   frameon=False, fontsize=8.5, labelcolor=TEXT_SECONDARY)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=geometry["dpi"], facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return {"component_id": component_id, "satellites": cop.number_of_nodes(), "constraints": cop.number_of_edges(),
            "bits": len(bits), "encoding_couplers": len(encoding_edges), "conflict_couplers": len(conflict_edges),
            "colour_clashes": clashes, "figsize": geometry["figsize"], "dpi": geometry["dpi"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pkl", type=Path, help="<stem>_components.pkl from scripts/cop_components_export.py")
    parser.add_argument("--components", nargs="+", help="component ids (default: every stored component)")
    parser.add_argument("--encoding", help="QUBO encoding to draw (default: the first one stored)")
    parser.add_argument("--labels", type=int, default=-1, metavar="N",
                        help=f"name the N largest satellites (default: all up to {AUTO_LABELS_ALL} satellites, "
                             f"else the {AUTO_LABELS_TOP} largest; 0 = none)")
    parser.add_argument("--no-labels", action="store_true", help="same as --labels 0")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="multiply the figure size (default 1: grows with the component, see INCH_PER_UNIT)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the spring layout")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="default: scripts/visualization/plot_cop_qubo_translation/")
    args = parser.parse_args(argv)
    if not args.pkl.is_file():
        parser.error(f"pickle not found: {args.pkl}")
    if args.pkl.suffix != ".pkl":
        parser.error(f"expected the <stem>_components.pkl of scripts/cop_components_export.py, got {args.pkl.name}"
                     + (" (the summary CSV next to it: pass the .pkl)" if args.pkl.suffix == ".csv" else ""))

    data = load_export(args.pkl)
    stem = args.pkl.stem.removesuffix("_components")
    encoding = args.encoding or data["metadata"]["qubo_encodings"][0]
    for component_id in args.components or list(data["components"]):
        if component_id not in data["components"]:
            parser.error(f"{component_id} not in {args.pkl.name}: {', '.join(data['components'])}")
        if encoding not in data["qubos"][component_id]:
            parser.error(f"{encoding} QUBO not stored (rerun the export with --encodings {encoding})")
        out_path = args.out_dir / f"{stem}_{component_id}_{encoding}_cop_qubo.png"
        stats = plot_component(stem, component_id, data["components"][component_id],
                               data["qubos"][component_id][encoding], encoding, out_path,
                               args.seed, args.dpi, 0 if args.no_labels else args.labels, args.scale)
        w, h = stats["figsize"]
        print(f"{component_id}: {stats['satellites']} satellites, {stats['constraints']} constraints -> "
              f"{stats['bits']} bits, {stats['encoding_couplers']} encoding + {stats['conflict_couplers']} "
              f"conflict couplers; colour clashes {stats['colour_clashes']}; figure {w:.0f} x {h:.0f} in "
              f"at {stats['dpi']} dpi; saved {out_path}")


if __name__ == "__main__":
    main()
