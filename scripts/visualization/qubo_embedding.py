"""QUBO -> minor embedding of a stored component, side by side, coloured by logical variable.

Reads a <stem>_components.pkl written by scripts/cop_components_export.py, embeds the QUBO of each
requested component with minorminer into an ideal D-Wave hardware graph (no QPU access, no faulty
qubits) and draws one figure with two panels:

    left   QUBO graph: one node per logical (binary) variable, one edge per coupler; the bits of a
           satellite sit on a ring in chain order, with the same layout as cop_qubo_translation.py
    right  hardware graph, the whole chip: every logical variable becomes a chain of physical qubits,
           drawn in the colour of its logical variable, with its internal couplers thick; the couplers
           between chains that carry a logical coupler are grey, the unused qubits and couplers faint.
           With --zoom the panel is cropped to the bounding box (in the topology's drawing coordinates)
           of every qubit the embedding uses, plus a ZOOM_PAD margin

Colours (--color-by):
    logical    (default) one colour per logical variable, assigned by graph colouring so that two
               logical variables coupled in the QUBO, or whose chains touch on the hardware, never
               share a colour; the 8 slots of a colour-blind-checked palette (identity is carried by
               colour plus adjacency: 100+ unique hues could not be told apart). If 8 colours are not
               enough the remaining nodes take the least-used colour and the clashes are reported
    satellite  the colour of the satellite (COP variable) the bit comes from, the same colours as
               cop_qubo_translation.py: COP variable -> logical variables -> chains
With --highlight SAT [SAT ...] (names as in the figures, e.g. 3848 or 11516-DTC, at most 8) only
the bits of those satellites and their chains are coloured, one colour per satellite, everything
else grey: the way to follow one satellite through the embedding by eye.

Topologies (--topology): pegasus = Advantage (P16, 5640 qubits), zephyr = Advantage2 (Z12 t4).
The embedding is heuristic and random: --seed fixes it, --tries / --timeout are passed to
minorminer. It is saved as JSON next to the PNG ({bit: [qubits]} plus statistics), so the same
chains can be reused or compared with the QPU run.

Self-contained: besides src/ (needed to unpickle the components) it depends on no other script.

Output, in --out-dir (default scripts/visualization/plot_qubo_embedding/):
    <stem>_<component_id>_<encoding>_<topology>_embedding[_<colouring>][_zoom].png
    <stem>_<component_id>_<encoding>_<topology>_embedding[_<colouring>][_zoom].json

Example:
    .venv/Scripts/python.exe scripts/visualization/qubo_embedding.py "scripts/cop_components_export/starlink_scheduling_slots_s500_g50_ deg50_seed65416843_components.pkl"
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import warnings

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="dwave-networkx is deprecated", category=DeprecationWarning)
import dwave_networkx as dnx  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import minorminer  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from dwave.embedding import is_valid_embedding  # noqa: E402
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
UNUSED = "#d6d5cf"
MUTED = "#bdbcb6"       # logical variables / chains outside the --highlight satellites
CHAIN_WIDTH = 1.6       # line width of the chain couplers (same logical variable) on the hardware
ZOOM_PAD = 0.04         # --zoom: margin around the used qubits, as a fraction of the box's longer side
# marker sizes and line widths of the hardware panel: the whole chip has ~25x more qubits in the same space
ZOOM_SIZES = {"qubit": 16, "qubit_ring": 0.5, "chain_edge": CHAIN_WIDTH, "logical_edge": 0.7,
              "unused_qubit": 2.5, "unused_edge": 0.3}
FULL_SIZES = {"qubit": 4, "qubit_ring": 0.0, "chain_edge": 1.0, "logical_edge": 0.35,
              "unused_qubit": 0.6, "unused_edge": 0.15}

TOPOLOGIES = {
    "pegasus": {"label": "Pegasus P16 (Advantage)", "graph": lambda: dnx.pegasus_graph(16),
                "layout": dnx.pegasus_layout},
    "zephyr": {"label": "Zephyr Z12 t4 (Advantage2)", "graph": lambda: dnx.zephyr_graph(12, 4),
               "layout": dnx.zephyr_layout},
}


# ---------------------------------------------------------------- data (same as cop_qubo_translation.py)
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


# ---------------------------------------------------------------- layout and colours (same as cop_qubo_translation.py)
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


# ---------------------------------------------------------------- embedding
def embed(S: nx.Graph, T: nx.Graph, seed: int, tries: int, timeout: float) -> dict[int, list[int]]:
    """minorminer embedding of S into T; isolated logical variables (no coupler) get a free qubit each."""
    chains = minorminer.find_embedding(list(S.edges), T, random_seed=seed, tries=tries, timeout=timeout)
    if S.number_of_edges() and not chains:
        raise RuntimeError("minorminer found no embedding (raise --timeout / --tries)")
    chains = {b: list(q) for b, q in chains.items()}
    used = {q for qs in chains.values() for q in qs}
    free = (q for q in sorted(T) if q not in used)
    for b in S:
        if b not in chains:
            chains[b] = [next(free)]
    if not is_valid_embedding(chains, S, T):
        raise RuntimeError("invalid embedding")
    return chains


def embedding_stats(chains: dict[int, list[int]], S: nx.Graph, T: nx.Graph) -> dict[str, Any]:
    qubit_chain = {q: b for b, qs in chains.items() for q in qs}
    lengths = [len(qs) for qs in chains.values()]
    chain_couplers = sum(1 for p, q in T.subgraph(qubit_chain).edges if qubit_chain[p] == qubit_chain[q])
    logical_couplers = sum(1 for p, q in T.subgraph(qubit_chain).edges
                           if qubit_chain[p] != qubit_chain[q] and S.has_edge(qubit_chain[p], qubit_chain[q]))
    return {
        "logical_variables": S.number_of_nodes(),
        "logical_couplers": S.number_of_edges(),
        "physical_qubits": len(qubit_chain),
        "hardware_qubits": T.number_of_nodes(),
        "chain_length_max": max(lengths),
        "chain_length_mean": round(sum(lengths) / len(lengths), 3),
        "chain_length_histogram": dict(sorted(Counter(lengths).items())),
        "chain_couplers": chain_couplers,
        "physical_couplers_for_logical": logical_couplers,
    }


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


def zoom_box(hw_pos: dict[int, np.ndarray], used_qubits) -> tuple[np.ndarray, np.ndarray]:
    """--zoom: bounding box, in the topology's drawing coordinates (dnx.*_layout), of every qubit the
    embedding uses (all chains, also with --highlight), widened by ZOOM_PAD of its longer side."""
    used = np.array([hw_pos[q] for q in used_qubits])
    pad = ZOOM_PAD * max(np.ptp(used[:, 0]), np.ptp(used[:, 1]), 1e-6)
    return used.min(axis=0) - pad, used.max(axis=0) + pad


def full_box(hw_pos: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Default: the whole chip, with a 1% margin."""
    every = np.array(list(hw_pos.values()))
    pad = 0.01 * max(np.ptp(every[:, 0]), np.ptp(every[:, 1]))
    return every.min(axis=0) - pad, every.max(axis=0) + pad


def fit_limits(ax, points: np.ndarray, pad_fraction: float = 0.04) -> None:
    pad = pad_fraction * max(np.ptp(points[:, 0]), np.ptp(points[:, 1]), 1e-6)
    ax.set_xlim(points[:, 0].min() - pad, points[:, 0].max() + pad)
    ax.set_ylim(points[:, 1].min() - pad, points[:, 1].max() + pad)


def resolve_highlight(names: list[str], component) -> list[str]:
    """Satellite names of the component matching the requested names (full or short form)."""
    by_name = {**{v.name: v.name for v in component.vars}, **{short_name(v.name): v.name for v in component.vars}}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise SystemExit(f"--highlight: {', '.join(missing)} not in this component "
                         f"(satellites: {', '.join(sorted(short_name(v.name) for v in component.vars))})")
    if len(names) > len(PALETTE):
        raise SystemExit(f"--highlight: at most {len(PALETTE)} satellites")
    return [by_name[n] for n in names]


def plot_component(stem: str, component_id: str, component, qubo: dict, encoding: str, topology: str,
                   colour_by: str, highlight: list[str], zoom: bool, out_png: Path, seed: int, tries: int,
                   timeout: float, dpi: int) -> dict[str, Any]:
    owner = bit_owner(component, qubo["index_map"])
    S = qubo_graph(qubo["Q"], owner)
    spec = TOPOLOGIES[topology]
    T = spec["graph"]()

    started = time.perf_counter()
    chains = embed(S, T, seed, tries, timeout)
    stats = {**embedding_stats(chains, S, T), "embedding_seconds": round(time.perf_counter() - started, 2)}
    qubit_chain = {q: b for b, qs in chains.items() for q in qs}

    # colours: per logical variable (QUBO couplers + chains touching on the hardware) or per satellite
    cop = cop_graph(component)
    radius = ring_radii(cop)
    pos = cop_layout(cop, radius, seed)
    bit_pos = qubo_layout(owner, pos, radius)
    if highlight:
        sat_colour = {s: PALETTE[k] for k, s in enumerate(highlight)}
        bit_colour = {b: sat_colour.get(s, MUTED) for b, s in owner.items()}
        clashes = 0
    elif colour_by == "logical":
        touching = {(qubit_chain[p], qubit_chain[q]) for p, q in T.subgraph(qubit_chain).edges
                    if qubit_chain[p] != qubit_chain[q]}
        bit_colour, clashes = colour_graph(S, touching)
    else:
        sat_colour, clashes = satellite_colours(cop, pos, radius)
        bit_colour = {b: sat_colour[s] for b, s in owner.items()}
    stats["colour_clashes"] = clashes

    fig, (ax_q, ax_hw) = plt.subplots(1, 2, figsize=(17, 8.8), facecolor=SURFACE)

    # --- left: QUBO
    encoding_edges = [(i, j) for i, j, d in S.edges(data=True) if d["kind"] == "encoding"]
    conflict_edges = [(i, j) for i, j, d in S.edges(data=True) if d["kind"] == "conflict"]
    draw_edges(ax_q, [(bit_pos[i], bit_pos[j]) for i, j in conflict_edges], NEUTRAL_EDGE, 0.5, alpha=0.35)
    draw_edges(ax_q, [(bit_pos[i], bit_pos[j]) for i, j in encoding_edges], NEUTRAL_EDGE, 1.4, alpha=0.8)
    # highlighted elements drawn last, on top of the muted ones
    on_top = (lambda b: bit_colour[b] != MUTED) if highlight else (lambda b: True)
    bits = sorted(owner, key=lambda b: (on_top(b), b))
    ax_q.scatter([bit_pos[b][0] for b in bits], [bit_pos[b][1] for b in bits], s=30,
                 c=[bit_colour[b] for b in bits], edgecolors=SURFACE, linewidths=0.6, zorder=3)
    for s in highlight:
        ring = np.array([bit_pos[b] for b in owner if owner[b] == s])
        ax_q.annotate(short_name(s), (ring[:, 0].mean(), ring[:, 1].max() + 0.03), ha="center", va="bottom",
                      fontsize=8, color=TEXT_PRIMARY, zorder=4)
    fit_limits(ax_q, np.array(list(bit_pos.values())), 0.06)

    # --- right: hardware, the whole chip or (--zoom) cropped to the used qubits
    hw_pos = spec["layout"](T)
    lo, hi = zoom_box(hw_pos, qubit_chain) if zoom else full_box(hw_pos)
    size = ZOOM_SIZES if zoom else FULL_SIZES
    inside = {q for q, p in hw_pos.items() if lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1]}
    background, chain_edges, logical_edges = [], [], []
    for p, q in T.subgraph(inside).edges:
        bp, bq = qubit_chain.get(p), qubit_chain.get(q)
        if bp is not None and bp == bq:
            chain_edges.append((p, q))
        elif bp is not None and bq is not None and S.has_edge(bp, bq):
            logical_edges.append((p, q))
        else:
            background.append((p, q))
    draw_edges(ax_hw, [(hw_pos[p], hw_pos[q]) for p, q in background], UNUSED, size["unused_edge"], alpha=0.6)
    free = sorted(inside - qubit_chain.keys())
    ax_hw.scatter([hw_pos[q][0] for q in free], [hw_pos[q][1] for q in free], s=size["unused_qubit"], c=UNUSED,
                  zorder=1, linewidths=0)
    draw_edges(ax_hw, [(hw_pos[p], hw_pos[q]) for p, q in logical_edges], NEUTRAL_EDGE, size["logical_edge"],
               alpha=0.7, zorder=2)
    chain_edges.sort(key=lambda e: on_top(qubit_chain[e[0]]))
    draw_edges(ax_hw, [(hw_pos[p], hw_pos[q]) for p, q in chain_edges],
               [bit_colour[qubit_chain[p]] for p, _ in chain_edges], size["chain_edge"], zorder=3)
    qubits = sorted(qubit_chain, key=lambda q: (on_top(qubit_chain[q]), q))
    ax_hw.scatter([hw_pos[q][0] for q in qubits], [hw_pos[q][1] for q in qubits], s=size["qubit"],
                  c=[bit_colour[qubit_chain[q]] for q in qubits], edgecolors=SURFACE,
                  linewidths=size["qubit_ring"], zorder=4)
    ax_hw.set_xlim(lo[0], hi[0])
    ax_hw.set_ylim(lo[1], hi[1])

    style_axis(ax_q, f"QUBO ({encoding})  ·  {stats['logical_variables']} logical variables, "
                     f"{stats['logical_couplers']} couplers")
    style_axis(ax_hw, f"{spec['label']}{' (zoom on the used qubits)' if zoom else ''}  ·  "
                      f"{stats['physical_qubits']} of {stats['hardware_qubits']} physical qubits, chain length "
                      f"max {stats['chain_length_max']}, mean {stats['chain_length_mean']:.2f}")

    # one legend per panel, under it: only the symbols drawn in that panel
    legend_qubo = [
        Line2D([], [], marker="o", ls="", color=NEUTRAL_EDGE, markersize=6, label="logical variable (binary)"),
        Line2D([], [], color=NEUTRAL_EDGE, lw=1.4, alpha=0.8,
               label="encoding coupler, same satellite (" + ("domain-wall chain" if encoding == "domain_wall" else "one-hot pair") + ")"),
        Line2D([], [], color=NEUTRAL_EDGE, lw=0.5, alpha=0.5, label="conflict coupler, two satellites"),
    ]
    legend_hw = [
        Line2D([], [], marker="o", ls="", color=NEUTRAL_EDGE, markersize=6, label="physical qubit, colour of its logical variable"),
        Line2D([], [], color=TEXT_SECONDARY, lw=CHAIN_WIDTH, label="chain coupler (same logical variable)"),
        Line2D([], [], color=NEUTRAL_EDGE, lw=0.8, alpha=0.7, label="coupler between chains, carries a QUBO coupler"),
        Line2D([], [], marker="o", ls="-", lw=0.4, color=UNUSED, markersize=3, label="unused qubit / coupler"),
    ]
    fig.suptitle(f"{stem}  ·  {component_id}: QUBO → minor embedding", fontsize=13, color=TEXT_PRIMARY,
                 x=0.012, ha="left")
    if highlight:
        note = (f"highlighted satellites ({', '.join(short_name(s) for s in highlight)}): their logical variables "
                "on the left and the chains of physical qubits they become on the right, one colour per satellite; "
                "everything else grey")
    elif colour_by == "logical":
        note = ("colour = logical variable: its chain of physical qubits on the right keeps its colour "
                "(coupled variables and touching chains never share a colour)")
    else:
        note = ("colour = satellite (COP variable): its logical variables on the left and their chains on the right "
                "keep its colour, as in cop_qubo_translation.py")
    if clashes:
        note += f"  ·  {clashes} adjacent pairs share a colour (more than {len(PALETTE)} needed)"
    fig.text(0.012, 0.935, note, fontsize=8.5, color=TEXT_SECONDARY, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    # legends on the same baseline, each centred under its own panel
    for ax, handles in ((ax_q, legend_qubo), (ax_hw, legend_hw)):
        box = ax.get_position()
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=((box.x0 + box.x1) / 2, 0.0), ncol=2,
                   frameon=False, fontsize=8.5, labelcolor=TEXT_SECONDARY)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    record = {
        "source_pickle": stem + "_components.pkl",
        "component_id": component_id,
        "encoding": encoding,
        "topology": topology,
        "topology_label": spec["label"],
        "seed": seed,
        "tries": tries,
        "timeout": timeout,
        "stats": stats,
        "satellite_of_bit": {str(b): s for b, s in sorted(owner.items())},
        "chains": {str(b): chains[b] for b in sorted(chains)},
    }
    out_png.with_suffix(".json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pkl", type=Path, help="<stem>_components.pkl from scripts/cop_components_export.py")
    parser.add_argument("--components", nargs="+", help="component ids (default: every stored component)")
    parser.add_argument("--encoding", help="QUBO encoding to embed (default: the first one stored)")
    parser.add_argument("--topology", choices=list(TOPOLOGIES), default="pegasus")
    parser.add_argument("--color-by", choices=("logical", "satellite"), default="logical")
    parser.add_argument("--highlight", nargs="+", default=[], metavar="SAT",
                        help="colour only these satellites (e.g. 3848 32641), everything else grey")
    parser.add_argument("--zoom", action="store_true",
                        help="crop the hardware panel to the qubits the embedding uses (default: whole chip)")
    parser.add_argument("--seed", type=int, default=0, help="seed of minorminer and of the QUBO layout")
    parser.add_argument("--tries", type=int, default=10, help="minorminer restarts")
    parser.add_argument("--timeout", type=float, default=120.0, help="minorminer timeout in seconds")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="default: scripts/visualization/plot_qubo_embedding/")
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
        component = data["components"][component_id]
        highlight = resolve_highlight(args.highlight, component) if args.highlight else []
        suffix = ("_highlight_" + "_".join(short_name(s) for s in highlight) if highlight
                  else "" if args.color_by == "logical" else "_by_satellite") + ("_zoom" if args.zoom else "")
        out_png = args.out_dir / f"{stem}_{component_id}_{encoding}_{args.topology}_embedding{suffix}.png"
        stats = plot_component(stem, component_id, component, data["qubos"][component_id][encoding], encoding,
                               args.topology, args.color_by, highlight, args.zoom, out_png, args.seed,
                               args.tries, args.timeout, args.dpi)
        print(f"{component_id}: {stats['logical_variables']} logical variables -> {stats['physical_qubits']} qubits, "
              f"chains max {stats['chain_length_max']} mean {stats['chain_length_mean']:.2f} "
              f"({stats['embedding_seconds']} s); colour clashes {stats['colour_clashes']}; saved {out_png}")


if __name__ == "__main__":
    main()
