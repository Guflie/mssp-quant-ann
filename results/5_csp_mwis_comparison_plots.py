"""Generate reproducible CSP/MWIS figures from the completed frozen campaign.

Run with Python 3.13.14 from any working directory. No experiment is executed.
Only figures and analysis tables are written; source runs are read-only.
Dependencies: numpy, pandas, matplotlib, networkx.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MWIS_CAMPAIGN = "new_pipeline_results_reordered"
DEFAULT_OUTPUT = RESULTS / "figures/csp_mwis_reordered"
COLORS = {"MWIS": "#2463A0", "CSP": "#C75C30"}
ORIGINS = ["Annealing", "Exact preprocessing", "Separator assignments", "Greedy fallback", "MILP"]
ORIGIN_COLORS = ["#2463A0", "#76B7B2", "#B89AC8", "#E9AE45", "#707780"]
HORIZON_HOURS = 4.0  # Frozen experiment design; not inferred from the last contact.


@dataclass(frozen=True)
class Scenario:
    key: str
    satellites: int
    gateways: int
    variables: int
    couplers: int
    mwis_run: str
    csp_bundle: str
    csp_run: str

    @property
    def label(self):
        return f"{self.satellites}/{self.gateways}\n{self.variables}/{self.couplers}"

    @property
    def input_path(self):
        return ROOT / f"data/experiment_inputs/exp_snapshot_2026-10-12/slots_s{self.satellites}_g{self.gateways}.csv"


# The baseline is reused, never treated as an independent replicate.
# The CSP gateway sweep baseline has full notebook clocks; its data/QUBOs match
# the standalone/satellite/terminal-profile baseline copies.
SCENARIOS = [
    Scenario("s100_g50", 100, 50, 30, 200, "exp2_s100_g50", "csp_sat_sweep", "ds_s100_g50"),
    Scenario("baseline", 200, 50, 30, 200, "exp1_s200_g50", "csp_gs_sweep", "gs_s200_g50"),
    Scenario("s500_g50", 500, 50, 30, 200, "exp2_s500_g50", "csp_sat_sweep", "ds_s500_g50"),
    Scenario("s200_g25", 200, 25, 30, 200, "exp3_s200_g25", "csp_gs_sweep", "gs_s200_g25"),
    Scenario("s200_g100", 200, 100, 30, 200, "exp3_s200_g100", "csp_gs_sweep", "gs_s200_g100"),
    Scenario("n20_c100", 200, 50, 20, 100, "exp4_s200_g50_n20_c100", "csp_qubo_sweep", "qpu_n20_c100"),
    Scenario("n50_c350", 200, 50, 50, 350, "exp4_s200_g50_n50_c350", "csp_qubo_sweep", "qpu_n50_c350"),
]
GROUPS = [
    ("Satellite scaling · 50 gateways", ["s100_g50", "baseline", "s500_g50"], ["100", "200", "500"], "Satellites"),
    ("Gateway scaling · 200 satellites", ["s200_g25", "baseline", "s200_g100"], ["25", "50", "100"], "Gateways"),
    ("Terminal sensitivity · 200/50", ["n20_c100", "baseline", "n50_c350"], ["20/100", "30/200", "50/350"], "Variable/coupler limits"),
]


class Inputs:
    """Hash every source actually used, so figures have a precise provenance."""
    def __init__(self):
        self.manifest = {}

    def read(self, path: Path):
        data = path.read_bytes()
        self.manifest[path.relative_to(ROOT).as_posix()] = hashlib.sha256(data).hexdigest()
        return data

    def json(self, path):
        return json.loads(self.read(path))

    def csv(self, path):
        from io import BytesIO
        data = self.read(path)
        try:
            return pd.read_csv(BytesIO(data))
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def metrics(self, path):
        frame = self.csv(path)
        return dict(zip(frame.metric, frame.value))

    def metadata(self, path):
        if path.suffix == ".gz":
            return json.loads(gzip.decompress(self.read(path)))
        return self.json(path)

    def verify_unchanged(self):
        for relative, expected in self.manifest.items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise AssertionError(f"Source changed while plotting: {relative}")


def number(metrics, key):
    if key not in metrics:
        raise KeyError(f"Missing required metric {key!r}")
    return float(metrics[key])


def phase_times(inputs, producer):
    files = list(producer.glob("*phase_profile_*.jsonl"))
    if len(files) != 1:
        raise ValueError(f"Expected exactly one phase log in {producer}, found {len(files)}")
    records = [json.loads(line) for line in inputs.read(files[0]).decode("utf-8").splitlines() if line]
    # Component timings are nested inside global decomposition and must not be added.
    return {row["phase"]: float(row["elapsed_seconds"]) for row in records if row["phase"] != "component_decomposition"}


def csp_notebook_clocks(inputs, scenario):
    summaries = {
        "csp_gs_sweep": "dataset_gs_sweep/gs_sweep_summary.json",
        "csp_sat_sweep": "dataset_sat_sweep/ds_sweep_summary.json",
        "csp_qubo_sweep": "qpu_sweep/qpu_sweep_summary.json",
    }
    path = RESULTS / "runs/csp" / scenario.csp_bundle / summaries[scenario.csp_bundle]
    record = next((r for r in inputs.json(path) if r["run_name"] == scenario.csp_run), None)
    if record is None:
        return np.nan, np.nan, "No saved full-notebook timing; producer stopwatch only"
    clocks = record["runtime_seconds"]
    return float(clocks["producer"]), float(clocks["qubo"]), path.relative_to(ROOT).as_posix()


def trace_reconstruction(metadata, samples):
    """Reconstruct the saved pre-MILP tree, retaining each contact's origin.

    Uses sampled choices and saved greedy leaves; never calls a solver.
    The branch score/tie rule matches the shared reconstruction notebook.
    """
    nodes = {n["node_id"]: n for n in metadata["nodes"]}
    memo = {}

    def merge(base, children, extra=None):
        origins = dict(base)
        for child in children:
            if set(origins) & set(child["origins"]):
                raise AssertionError("Duplicate contact across independent children")
            origins.update(child["origins"])
        if extra:
            if set(origins) & set(extra):
                raise AssertionError("Separator or sample duplicates a fixed contact")
            origins.update(extra)
        return origins

    def evaluate(node_id):
        if node_id in memo:
            return memo[node_id]
        node = nodes[node_id]
        fixed = {slot: "Exact preprocessing" for slot in node.get("forced_selected", [])}
        score = float(node.get("forced_objective", 0.0))
        kind = node["node_type"]
        if kind == "terminal":
            terminal_id = node.get("terminal_id")
            row = samples[terminal_id] if terminal_id else None
            chosen = json.loads(row["selected_nodes"]) if row else []
            origins = merge(fixed, [], {s: "Annealing" for s in chosen})
            score += float(row["objective"]) if row else 0.0
            terminal_ids = {terminal_id} if terminal_id else set()
        elif kind == "incomplete":
            h = node["heuristic_solution"]
            origins = merge(fixed, [], {s: "Greedy fallback" for s in h["heuristic_selected_nodes"]})
            score = float(h["objective"])
            terminal_ids = set()
        elif kind == "independent_sum":
            children = [evaluate(c) for c in node["children"]]
            origins = merge(fixed, children)
            score += sum(c["objective"] for c in children)
            terminal_ids = set().union(*(c["terminal_ids"] for c in children))
        elif kind == "separator":
            alternatives = []
            for branch in node["branches"]:
                children = [evaluate(c) for c in branch["children"]]
                chosen = {s: "Separator assignments" for s in branch["selected_separator"]}
                alternatives.append({
                    "origins": merge(fixed, children, chosen),
                    "objective": score + float(branch["state_objective"]) + sum(c["objective"] for c in children),
                    "terminal_ids": set().union(*(c["terminal_ids"] for c in children)),
                })
            result = max(alternatives, key=lambda r: (r["objective"], tuple(sorted(r["origins"]))))
            memo[node_id] = result
            return result
        else:
            raise ValueError(f"Unknown tree node type: {kind}")
        result = {"origins": origins, "objective": score, "terminal_ids": terminal_ids}
        memo[node_id] = result
        return result

    return [evaluate(c["root_node_id"]) for c in metadata["components"]]


def validate_schedule(slot_ids, raw):
    """Check original scheduling constraints with integer microseconds."""
    if len(slot_ids) != len(set(slot_ids)) or not set(slot_ids) <= set(raw.index):
        raise AssertionError("Duplicate or unknown scheduled contact")
    chosen = raw.loc[slot_ids]
    if chosen.satellite_id.duplicated().any():
        raise AssertionError("Satellite served more than once")
    for _, group in chosen.groupby("antenna_id"):
        ordered = group.sort_values("start_time_hours")
        previous = None
        for row in ordered.itertuples():
            values = tuple(round(float(getattr(row, field)) * 3_600_000_000) for field in
                           ["start_time_hours", "end_time_hours", "reconfiguration_time_hours"])
            if previous and values[0] - previous[1] < max(values[2], previous[2]):
                raise AssertionError("Antenna overlap/setup conflict")
            previous = values
    return sum(int(Decimal(str(w)) * 1000) for w in chosen.weight) / 1000


def load_run(inputs, scenario, method):
    raw = inputs.csv(scenario.input_path).set_index("slot_id", drop=False)
    assert raw.index.is_unique
    assert raw.satellite_id.nunique() == scenario.satellites
    assert raw.station_id.nunique() == scenario.gateways
    assert raw.start_time_hours.min() >= 0 and raw.end_time_hours.max() <= HORIZON_HOURS
    assert (raw.groupby("station_id").antenna_id.nunique() == 1).all()
    if method == "MWIS":
        run = f"{MWIS_CAMPAIGN}/{scenario.mwis_run}"
        base = RESULTS / "runs/mwis"
        producer, annealer = base / "pipeline_results" / run, base / "qubo_annealing" / run
        execution = inputs.json(base / "execution_logs" / run / "execution_summary.json")
        assert execution["status"] == "completed_and_validated"
        producer_clock = next(s["wall_seconds"] for s in execution["stages"] if s["stage"] == "mwis")
        annealer_clock = next(s["wall_seconds"] for s in execution["stages"] if s["stage"] == "annealing")
        timing_source = (base / "execution_logs" / run / "execution_summary.json").relative_to(ROOT).as_posix()
    else:
        run = f"{scenario.csp_bundle}/{scenario.csp_run}"
        base = RESULTS / "runs/csp" / scenario.csp_bundle
        producer, annealer = base / "pipeline_results" / scenario.csp_run, base / "qubo_annealing" / scenario.csp_run
        producer_clock, annealer_clock, timing_source = csp_notebook_clocks(inputs, scenario)
    assert "dominance-enhanced" not in str(producer)
    config = inputs.json(producer / "run_config_log.json")
    config_input = config.get("candidate_slot_csv") if method == "MWIS" else config.get("csp_slot_csv", config.get("slot_csv"))
    assert (ROOT / config_input).resolve() == scenario.input_path.resolve()
    sample_config = inputs.json(annealer / "run_config_log.json")
    assert int(sample_config["max_qpu_terminal_nodes"]) == scenario.variables
    assert int(sample_config["max_qpu_terminal_couplers"]) == scenario.couplers
    metadata = inputs.metadata(producer / "reconstruction_metadata.json.gz")
    producer_metrics = inputs.metrics(producer / "final_summary.csv")
    final_metrics = inputs.metrics(annealer / "reconstructed_summary.csv")
    terminals = inputs.csv(annealer / "terminal_qubo_summary.csv")
    sampled = inputs.csv(annealer / "annealing_results.csv")
    schedule = inputs.csv(annealer / "reconstructed_schedule.csv")
    provenance = inputs.csv(annealer / "component_provenance_summary.csv").set_index("component_id")
    graph_nodes = inputs.csv(producer / "conflict_graph_nodes.csv")
    graph_edges = inputs.csv(producer / "conflict_graph_edges.csv")
    reads = int(sample_config["num_reads"])
    assert len(sampled) == len(terminals) == int(number(final_metrics, "Annealed terminal subgraphs available"))
    assert set(sampled.terminal_id) == set(terminals.terminal_id)
    assert (sampled.num_reads == reads).all() and (sampled.is_independent.astype(str).str.lower() == "true").all()
    assert terminals.variables.max() <= scenario.variables and terminals.couplers.max() <= scenario.couplers
    samples = {r["terminal_id"]: r for r in sampled.to_dict("records")}
    before = trace_reconstruction(metadata, samples)
    pre_ids = [s for result in before for s in result["origins"]]
    pre_objective = validate_schedule(pre_ids, raw)
    assert abs(pre_objective - sum(r["objective"] for r in before)) < 1e-6
    classical_schedule = inputs.csv(producer / ("mwis_solution.csv" if method == "MWIS" else "csp_solution.csv"))
    classical_objective = validate_schedule(classical_schedule.slot_id.tolist(), raw)
    assert abs(classical_objective - number(producer_metrics, "Total feasible objective / lower bound")) < 1e-6
    final_ids = schedule.slot_id.tolist()
    final_objective = validate_schedule(final_ids, raw)
    assert abs(final_objective - number(final_metrics, "Total feasible objective")) < 1e-6
    assert len(final_ids) == int(number(final_metrics, "Selected contacts"))
    upper = number(final_metrics, "Hybrid conservative objective upper bound")
    certified = abs(final_objective - upper) < 1e-6 and final_metrics["Status"] == "hybrid_exact"
    if not certified:
        raise AssertionError("These plots expect all saved final schedules to be certified; inspect the changed run")
    producer_upper = number(producer_metrics, "Total conservative upper bound")
    assert producer_upper + 1e-6 >= pre_objective
    if "component_id" in graph_nodes:
        component_parts = {cid: set(group.slot_id) for cid, group in graph_nodes.groupby("component_id")}
    else:
        graph = nx.Graph()
        graph.add_nodes_from(graph_nodes.slot_id)
        graph.add_edges_from(graph_edges[["source", "target"]].itertuples(index=False, name=None))
        parts = sorted(nx.connected_components(graph), key=lambda p: (-len(p), sorted(p)))
        component_parts = {c["component_id"]: part for c, part in zip(metadata["components"], parts)}
        assert len(parts) == len(metadata["components"])
    assert set(component_parts) == set(provenance.index) == {c["component_id"] for c in metadata["components"]}
    final_set = set(final_ids)
    final_origins = {}
    retained_terminals = set()
    for component, result in zip(metadata["components"], before):
        cid = component["component_id"]
        chosen = final_set & component_parts[cid]
        source = str(provenance.loc[cid, "solution_source"])
        if source.startswith("MILP"):
            final_origins.update({s: "MILP" for s in chosen})
        else:
            assert chosen == set(result["origins"])
            final_origins.update(result["origins"])
            retained_terminals.update(result["terminal_ids"])
    assert set(final_origins) == final_set
    assert len(retained_terminals) == int(number(final_metrics, "Annealed terminal subgraphs used"))
    phases = phase_times(inputs, producer)
    if method == "MWIS":
        build = sum(phases[k] for k in ["candidate_load_and_export", "conflict_graph_build_and_export", "natural_components_and_summary"])
        decomposition = phases["decomposition_and_terminal_export"]
        classical = phases["recursive_solution_evaluation"]
        translation_export = sum(phases[k] for k in ["qubo_translation", "solution_reconstruction_and_exports", "final_reporting_export"])
    else:
        build = phases["csp_build"] + phases["exact_reduction"]
        decomposition = phases["exact_decomposition"]
        classical = phases["classical_evaluation"]
        translation_export = sum(phases[k] for k in ["qubo_translation", "export", "reconstruction_export", "final_reporting_export"])
    producer_stopwatch = phases["pipeline_total_so_far"]
    plotted_producer_total = producer_clock if np.isfinite(producer_clock) else producer_stopwatch
    profiled_sum = build + decomposition + classical + translation_export
    unprofiled = plotted_producer_total - profiled_sum
    assert unprofiled >= -.02, "Overlapping phase clocks must not be summed"
    solver_time = number(final_metrics, "MILP solver time (seconds)") + number(final_metrics, "Triangle-LP fallback solver time (seconds)")
    annealer_remainder = annealer_clock - solver_time if np.isfinite(annealer_clock) else np.nan
    assert not np.isfinite(annealer_remainder) or annealer_remainder >= 0
    counts = Counter(final_origins.values())
    pre_counts = Counter(s for result in before for s in result["origins"].values())
    pre_origins = {slot: origin for result in before for slot, origin in result["origins"].items()}
    pre_origin_objectives = {origin: sum(float(raw.loc[s, "weight"]) for s, kind in pre_origins.items() if kind == origin) for origin in ORIGINS}
    origin_objectives = {origin: sum(float(raw.loc[s, "weight"]) for s, kind in final_origins.items() if kind == origin) for origin in ORIGINS}
    record = {
        "scenario": scenario.key, "method": method, "run": run,
        "satellites": scenario.satellites, "gateways": scenario.gateways,
        "variable_limit": scenario.variables, "coupler_limit": scenario.couplers,
        "encoding": metadata["qubo_encoding"], "reads_per_terminal": reads,
        "input_sha256": inputs.manifest[scenario.input_path.relative_to(ROOT).as_posix()],
        "candidate_contacts": len(raw), "retained_graph_contacts": len(graph_nodes),
        "graph_contacts_retained_percent": 100 * len(graph_nodes) / len(raw),
        "retained_graph_edges": len(graph_edges), "components": len(component_parts),
        "mean_component_contacts": np.mean([len(p) for p in component_parts.values()]),
        "largest_component_contacts": max(map(len, component_parts.values())),
        "terminal_qubos": len(terminals), "retained_terminal_qubos": len(retained_terminals),
        "terminal_retention_percent": 100 * len(retained_terminals) / len(terminals),
        "mean_terminal_variables": terminals.variables.mean(), "median_terminal_variables": terminals.variables.median(),
        "max_terminal_variables": terminals.variables.max(), "mean_terminal_couplers": terminals.couplers.mean(),
        "max_terminal_couplers": terminals.couplers.max(), "total_reads": int(sampled.num_reads.sum()),
        "incomplete_components": int(number(final_metrics, "Components with incomplete separator decomposition (all states)")),
        "incomplete_leaves": int(number(final_metrics, "Incomplete separator nodes (all states)")),
        "pre_milp_objective": pre_objective, "pre_milp_selected_contacts": len(pre_ids),
        "classical_reference_objective": classical_objective,
        "sampling_objective_change_from_reference": pre_objective - classical_objective,
        "classical_gap_to_certified_optimum_percent": 100 * (upper - classical_objective) / upper,
        "final_objective": final_objective, "final_upper_bound": upper, "certified_optimal": certified,
        "pre_milp_gap_to_certified_optimum_percent": 100 * (upper - pre_objective) / upper,
        "pre_milp_conservative_gap_bound_percent": 100 * (producer_upper - pre_objective) / producer_upper,
        "milp_objective_improvement": final_objective - pre_objective,
        "final_selected_contacts": len(final_ids), "selected_conflicts": 0,
        "producer_notebook_seconds": producer_clock, "producer_stopwatch_seconds": producer_stopwatch,
        "annealing_notebook_seconds": annealer_clock,
        "total_notebook_seconds": producer_clock + annealer_clock,
        "time_build_reduction": build, "time_decomposition": decomposition,
        "time_classical_verification": classical, "time_qubo_exports": translation_export,
        "time_unprofiled_producer": max(0, unprofiled),
        "time_annealing_reconstruction_remainder": annealer_remainder, "time_milp_lp_solver": solver_time,
        "producer_timing_complete": bool(np.isfinite(producer_clock)),
        "total_timing_complete": bool(np.isfinite(producer_clock) and np.isfinite(annealer_clock)),
        "timing_source": timing_source,
    }
    for origin in ORIGINS:
        suffix = origin.lower().replace(" ", "_")
        record[f"final_contacts_{suffix}"] = counts[origin]
        record[f"final_objective_{suffix}"] = origin_objectives[origin]
        record[f"pre_milp_contacts_{suffix}"] = pre_counts[origin]
        record[f"pre_milp_objective_{suffix}"] = pre_origin_objectives[origin]
    terminal_frame = terminals.copy()
    terminal_frame.insert(0, "method", method)
    terminal_frame.insert(0, "scenario", scenario.key)
    terminal_frame["retained_in_final_schedule"] = terminal_frame.terminal_id.isin(retained_terminals)
    terminal_frame["num_reads"] = reads
    terminal_sizes = terminals.set_index("terminal_id").variables.to_dict()
    leaf_records = []
    for node in metadata["nodes"]:
        kind = node["node_type"]
        if kind == "terminal" and node.get("terminal_id"):
            assert int(node["qpu_node_count"]) == int(terminal_sizes[node["terminal_id"]])
        elif kind != "incomplete":
            continue  # Empty terminal leaves are solved by exact fixed choices.
        leaf_records.append({"scenario": scenario.key, "method": method, "node_id": node["node_id"],
                             "leaf_kind": kind, "contacts_or_binary_variables": int(node["qpu_node_count"]),
                             "couplers": node.get("qpu_edge_count"), "depth": node["depth"]})
    leaf_frame = pd.DataFrame(leaf_records)
    assert (leaf_frame.leaf_kind == "terminal").sum() == len(terminals)
    assert (leaf_frame.leaf_kind == "incomplete").sum() == record["incomplete_leaves"]
    schedule_frame = raw.loc[final_ids, ["slot_id", "satellite_id", "station_id", "antenna_id", "start_time_hours", "end_time_hours", "reconfiguration_time_hours", "weight"]].copy().reset_index(drop=True)
    schedule_frame["scenario"] = scenario.key
    schedule_frame["method"] = method
    gateway_records = []
    for station_id in sorted(raw.station_id.unique()):
        selected_here = schedule_frame[schedule_frame.station_id == station_id]
        minutes = 60 * (selected_here.end_time_hours - selected_here.start_time_hours).sum()
        gateway_records.append({"scenario": scenario.key, "method": method, "station_id": station_id,
                                "selected_contacts": len(selected_here), "selected_weight": selected_here.weight.sum(),
                                "contact_minutes": minutes, "service_utilization_percent": 100 * minutes / (60 * HORIZON_HOURS),
                                "horizon_hours": HORIZON_HOURS})
    gateway_frame = pd.DataFrame(gateway_records)
    assert gateway_frame.selected_contacts.sum() == len(final_ids)
    assert abs(gateway_frame.selected_weight.sum() - final_objective) < 1e-6
    assert (gateway_frame.service_utilization_percent <= 100 + 1e-6).all()
    return record, terminal_frame, leaf_frame, schedule_frame, gateway_frame


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 10, "figure.titlesize": 16, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18,
                         "axes.axisbelow": True, "savefig.facecolor": "white", "pdf.fonttype": 42})


def save(fig, output, name, caption, captions):
    fig.savefig(output / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    captions[name] = caption


def grouped_bars(ax, data, metric, ylabel, title, log=False):
    x = np.arange(len(SCENARIOS))
    for offset, method in [(-.19, "MWIS"), (.19, "CSP")]:
        values = [float(data.loc[(s.key, method), metric]) for s in SCENARIOS]
        ax.bar(x + offset, values, .36, color=COLORS[method], label=method)
    ax.set_xticks(x, [s.label for s in SCENARIOS], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)


def objectives(data, output, captions):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (title, keys, labels, xlabel) in zip(axes, GROUPS):
        for method in COLORS:
            series = data.loc[[(key, method) for key in keys]]
            for metric, stage, linestyle, marker in [("pre_milp_objective", "Before MILP", "--", "^"), ("final_objective", "Final", "-", "o" if method == "MWIS" else "s")]:
                ax.plot(labels, series[metric], color=COLORS[method], linestyle=linestyle, marker=marker,
                        markersize=6 if method == "MWIS" else 9, markerfacecolor=COLORS[method] if method == "MWIS" else "none", label=f"{method} · {stage}")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, max(data.loc[(key, method), "final_objective"] for key in keys for method in COLORS) * 1.12)
    axes[0].set_ylabel("Total schedule weight")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(.5, -.04))
    fig.suptitle("Schedule objectives before and after MILP")
    fig.tight_layout(rect=(0, .07, 1, .92))
    save(fig, output, "01_objective_comparison", "Schedule weights, not QUBO energies. Final CSP/MWIS objectives coincide for each matched input; hollow CSP markers reveal the overlap. Shared baseline is reused, not replicated.", captions)


def mean_sizes(data, output, captions):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    grouped_bars(axes[0], data, "mean_terminal_variables", "Mean binary variables", "Average terminal QUBO size")
    grouped_bars(axes[1], data, "mean_terminal_couplers", "Mean nonzero couplers", "Average terminal QUBO connectivity")
    axes[0].legend(ncol=2)
    axes[1].set_xlabel("Satellites/gateways (first line) · variable/coupler limits (second line)")
    fig.tight_layout()
    save(fig, output, "02_average_terminal_size", "Arithmetic means across all exported non-empty terminal QUBOs, including alternative branches later discarded. Both axes use common binary-QUBO units, not CSP satellite-variable counts versus MWIS contact-vertex counts.", captions)


def runtimes(data, output, captions):
    sections = [
        ("time_build_reduction", "Load/build/reduction", "#6AAED6"),
        ("time_decomposition", "Separator decomposition", "#2463A0"),
        ("time_classical_verification", "Classical terminal verification", "#9C79B4"),
        ("time_qubo_exports", "Producer QUBO/export", "#76B7B2"),
        ("time_unprofiled_producer", "Other producer time", "#C7CCD1"),
        ("time_annealing_reconstruction_remainder", "Annealing/reconstruction + other notebook time", "#E9AE45"),
        ("time_milp_lp_solver", "MILP/LP solver", "#707780"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    for i, scenario in enumerate(SCENARIOS):
        for offset, method in [(-.2, "MWIS"), (.2, "CSP")]:
            row = data.loc[(scenario.key, method)]
            x = i + offset
            bottom = 0
            for metric, _, color in sections:
                value = row[metric]
                if np.isfinite(value):
                    axes[0].bar(x, value, .37, bottom=bottom, color=color, edgecolor="white", linewidth=.3)
                    bottom += value
            axes[0].text(x, bottom + 18, f"{method}\n{bottom:.1f}s" if row.total_timing_complete else f"{method}\npartial*", ha="center", fontsize=7)
            if row.total_timing_complete:
                axes[1].scatter(x, row.total_notebook_seconds, color=COLORS[method], s=45)
                axes[1].annotate(f"{row.total_notebook_seconds:.1f}", (x, row.total_notebook_seconds), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
            else:
                axes[1].text(x, .12, "N/A", ha="center", va="bottom", fontsize=8, transform=axes[1].get_xaxis_transform())
    axes[0].set_title("Recorded time by pipeline section (linear scale)")
    axes[0].set_ylabel("Recorded seconds")
    axes[0].set_ylim(bottom=0)
    axes[1].set_title("Full notebook sequence time (log scale)")
    axes[1].set_ylabel("Seconds · log scale")
    axes[1].set_yscale("log")
    axes[1].margins(y=.25)
    for ax in axes:
        ax.set_xticks(range(len(SCENARIOS)), [s.label for s in SCENARIOS], fontsize=8)
        ax.grid(axis="x", visible=False)
    fig.legend(handles=[Patch(color=c, label=l) for _, l, c in sections], loc="lower center", bbox_to_anchor=(.5, -.02), ncol=3, fontsize=8)
    fig.suptitle("Recorded runtimes · hardware and sampling budgets were not controlled")
    fig.tight_layout(rect=(0, .09, 1, .95))
    save(fig, output, "03_runtime_breakdown", "CSP hardware/environment provenance is incomplete and sampling budgets differ, so these are recorded timings rather than a controlled speed comparison. Total is the sum of producer and annealing notebook elapsed times; launcher/orchestration overhead is excluded. *CSP 100/50 lacks full notebook clocks: its bar contains only the producer stopwatch and its total is N/A. CSP does not save separate sampling/reconstruction clocks; the annealing notebook remainder combines them with setup/export overhead after subtracting MILP/LP solver time. Producer phases are disjoint; nested component timings and pipeline totals are not added twice. Frozen-input generation time is not measured here.", captions)


def logical_dimensions(terminals, output, captions):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.7))
    for ax, scenario in zip(axes.flat, SCENARIOS):
        for method in COLORS:
            t = terminals[(terminals.scenario == scenario.key) & (terminals.method == method)]
            counts = t.groupby(["variables", "couplers"]).size().reset_index(name="count")
            ax.scatter(counts.variables, counts.couplers, s=14 + 8 * np.sqrt(counts["count"]), color=COLORS[method], alpha=.65, label=method, marker="o" if method == "MWIS" else "^")
        ax.axvline(scenario.variables, color="#555", linestyle="--", linewidth=.8)
        ax.axhline(scenario.couplers, color="#555", linestyle="--", linewidth=.8)
        ax.set_xlim(0, scenario.variables * 1.1)
        ax.set_ylim(0, scenario.couplers * 1.12)
        ax.set_title(f"{scenario.satellites}/{scenario.gateways} · limits {scenario.variables}/{scenario.couplers}", fontsize=10)
        ax.set_xlabel("Binary variables")
        ax.set_ylabel("Nonzero couplers")
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(handles=[Line2D([], [], marker="o" if m == "MWIS" else "^", linestyle="", color=COLORS[m], label=m) for m in COLORS], loc="upper left")
    axes.flat[-1].text(.02, .63, "Logical QUBO dimensions\n\nDashed lines: configured limits\nMarker area reflects repeated sizes\n\nPhysical embedding size:\nnot measured in these runs", transform=axes.flat[-1].transAxes, fontsize=11, va="top")
    fig.suptitle("Logical problem dimensions submitted to simulated annealing")
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, output, "04_logical_qubo_dimensions", "Binary variables and nonzero quadratic couplers for sampled terminals. Repeated (variables, couplers) pairs are aggregated; marker area increases with frequency. These are logical dimensions, not physical qubits, chains, chain lengths or embedding overhead. No hardware embedding was performed.", captions)


def quality_gaps(data, output, captions):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (title, keys, labels, xlabel) in zip(axes, GROUPS):
        for method in COLORS:
            values = [data.loc[(k, method), "pre_milp_gap_to_certified_optimum_percent"] for k in keys]
            ax.plot(labels, values, color=COLORS[method], marker="o" if method == "MWIS" else "^", label=method)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Gap before MILP (%)")
    axes[0].legend()
    fig.suptitle("Pre-MILP schedule gap relative to the certified final optimum")
    fig.tight_layout(rect=(0, 0, 1, .91))
    save(fig, output, "05_pre_milp_quality_gap", "100 × (certified final optimum − reconstructed pre-MILP weight) / certified final optimum. Computed only after confirming final objective equals the saved upper bound. This is the actual retrospective gap for these inputs, distinct from the looser pre-MILP conservative bound. Final certified gaps are zero.", captions)


def provenance_plot(data, output, captions):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    for ax, quantity, title in zip(axes, ["contacts", "objective"], ["Share of selected contacts", "Share of final schedule weight"]):
        for i, scenario in enumerate(SCENARIOS):
            for offset, method in [(-.2, "MWIS"), (.2, "CSP")]:
                row = data.loc[(scenario.key, method)]
                denominator = row.final_selected_contacts if quantity == "contacts" else row.final_objective
                bottom = 0
                for origin, color in zip(ORIGINS, ORIGIN_COLORS):
                    value = 100 * row[f"final_{quantity}_{origin.lower().replace(' ', '_')}"] / denominator
                    ax.bar(i + offset, value, .37, bottom=bottom, color=color)
                    if value >= 12:
                        ax.text(i + offset, bottom + value / 2, f"{value:.0f}%", ha="center", va="center", fontsize=7, color="white" if origin in ["Annealing", "MILP"] else "black")
                    bottom += value
                ax.text(i + offset, 102, method, ha="center", fontsize=7)
        ax.set_ylim(0, 112)
        ax.set_ylabel("Percent")
        ax.set_title(title)
        ax.set_xticks(range(len(SCENARIOS)), [s.label for s in SCENARIOS], fontsize=8)
        ax.grid(axis="x", visible=False)
    fig.legend(handles=[Patch(color=c, label=l) for l, c in zip(ORIGINS, ORIGIN_COLORS)], loc="lower center", ncol=5, bbox_to_anchor=(.5, -.01), fontsize=9)
    fig.suptitle("Final schedule provenance · retained contributions after MILP")
    fig.tight_layout(rect=(0, .05, 1, .94))
    save(fig, output, "06_final_schedule_provenance", "Origins traced through the saved decomposition and sampled selections, then replaced by MILP provenance only for affected components. Exact preprocessing and separator choices are classical. Shares measure retained contacts/weight, not CPU work or quantum advantage. A sampled terminal discarded by reconstruction or MILP contributes no retained share.", captions)


def workload(data, output, captions):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    grouped_bars(axes[0, 0], data, "terminal_qubos", "QUBOs · log scale", "Terminal QUBOs sampled", log=True)
    grouped_bars(axes[0, 1], data, "total_reads", "Reads · log scale", "Total sampling budget", log=True)
    grouped_bars(axes[1, 0], data, "terminal_retention_percent", "Percent", "Sampled terminals retained after MILP")
    # Zero leaves remain visible as labels; use a symlog scale rather than dropping zero.
    grouped_bars(axes[1, 1], data, "incomplete_leaves", "Incomplete leaves · symlog scale", "Unresolved decomposition leaves (all branches)")
    axes[1, 1].set_yscale("symlog", linthresh=1)
    for i, scenario in enumerate(SCENARIOS):
        for offset, method in [(-.19, "MWIS"), (.19, "CSP")]:
            if data.loc[(scenario.key, method), "incomplete_leaves"] == 0:
                axes[1, 1].text(i + offset, .13, "0", ha="center", fontsize=8)
    axes[0, 0].legend(ncol=2)
    fig.suptitle("Workload and decomposition completeness · MWIS 500 / CSP 1,000 reads per QUBO")
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, output, "07_sampling_and_decomposition", "Counts include all alternative branches exported as non-empty terminal QUBOs. Total reads are summed from actual annealing_results.csv rows; per-terminal budgets are 500 for MWIS and 1,000 for CSP. Retention uses final terminal IDs, not contact counts. Incomplete leaves are all unresolved branches, not the count selected in the final schedule; zero is kept explicitly.", captions)


def component_structure(data, output, captions):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    grouped_bars(axes[0, 0], data, "mean_component_contacts", "Mean candidate contacts", "Average independent-component size")
    grouped_bars(axes[0, 1], data, "largest_component_contacts", "Candidate contacts", "Largest independent component")
    grouped_bars(axes[1, 0], data, "components", "Connected components", "Independent components before separator recursion")
    grouped_bars(axes[1, 1], data, "graph_contacts_retained_percent", "Percent of original candidates", "Contacts retained before component decomposition")
    axes[0, 0].legend(ncol=2)
    fig.suptitle("Component structure after each pipeline's initial preprocessing")
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, output, "08_component_structure", "Component sizes use candidate contacts in both methods, rather than CSP satellite variables versus MWIS contact vertices. CSP has already applied global dominance; this MWIS campaign has not. Means include singleton components, so the largest component is shown alongside them. These are root components before recursive separator decomposition, not terminal QUBOs or incomplete leaves.", captions)


def quality_stages(data, output, captions):
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    stage_metrics = ["classical_reference_objective", "pre_milp_objective", "final_objective"]
    labels = ["Classical\nreference", "Sampled\nreconstruction", "Final\nhybrid"]
    # One common, explicitly zoomed percentage scale for all scenarios.
    minimum = min(100 * row[metric] / row.final_upper_bound for _, row in data.iterrows() for metric in stage_metrics)
    lower = np.floor(minimum - .5)
    for ax, scenario in zip(axes.flat, SCENARIOS):
        for method in COLORS:
            row = data.loc[(scenario.key, method)]
            values = [100 * row[metric] / row.final_upper_bound for metric in stage_metrics]
            ax.plot(range(3), values, color=COLORS[method], marker="o" if method == "MWIS" else "s",
                    markersize=5 if method == "MWIS" else 8, markerfacecolor=COLORS[method] if method == "MWIS" else "none", label=method)
        ax.axhline(100, color="#777", linestyle=":", linewidth=.7)
        ax.set_xticks(range(3), labels, fontsize=8)
        ax.set_ylim(lower, 100.8)
        ax.set_ylabel("Certified optimum (%)")
        ax.set_title(f"{scenario.satellites}/{scenario.gateways} · {scenario.variables}/{scenario.couplers}", fontsize=10)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(handles=[Line2D([], [], marker="o" if m == "MWIS" else "s", color=COLORS[m], label=m) for m in COLORS], loc="upper left")
    axes.flat[-1].text(.02, .65, "Zoomed percentage scale\n100% = certified optimum\n\nClassical reference uses exact\nterminal solves plus the same\ngreedy incomplete leaves.\n\nFinal hybrid uses MILP\nwhere fallback is required.", transform=axes.flat[-1].transAxes, va="top", fontsize=10)
    fig.suptitle("Schedule quality through the three computational stages")
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, output, "09_quality_through_stages", "Original schedule weights divided by the certified optimum of each input. The percentage axis is zoomed, with the same limits in every panel. Classical reference comes from the saved producer solution and is independently validated; sampled reconstruction comes from the saved tree/read results. Final hybrid may require no MILP for a complete run. In this campaign reference and sampled objectives match in all 14 records; this is not an annealer-only improvement.", captions)


def pre_post_provenance(data, output, captions):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    for ax, method in zip(axes, COLORS):
        for i, scenario in enumerate(SCENARIOS):
            row = data.loc[(scenario.key, method)]
            for offset, stage, prefix, denominator in [(-.2, "Pre", "pre_milp", row.pre_milp_selected_contacts), (.2, "Final", "final", row.final_selected_contacts)]:
                bottom = 0
                for origin, color in zip(ORIGINS, ORIGIN_COLORS):
                    key = origin.lower().replace(" ", "_")
                    value = 100 * row[f"{prefix}_contacts_{key}"] / denominator
                    ax.bar(i + offset, value, .37, bottom=bottom, color=color)
                    if value >= 15:
                        ax.text(i + offset, bottom + value / 2, f"{value:.0f}%", ha="center", va="center", fontsize=7,
                                color="white" if origin in ["Annealing", "MILP"] else "black")
                    bottom += value
                ax.text(i + offset, 103, f"{stage}\n{int(denominator)}", ha="center", fontsize=7)
        ax.set_ylim(0, 116)
        ax.set_xticks(range(len(SCENARIOS)), [s.label for s in SCENARIOS], fontsize=8)
        ax.set_title(method)
        ax.set_ylabel("Share of selected contacts (%)")
        ax.grid(axis="x", visible=False)
    fig.legend(handles=[Patch(color=c, label=l) for l, c in zip(ORIGINS, ORIGIN_COLORS)], loc="lower center", ncol=5, bbox_to_anchor=(.5, -.01), fontsize=9)
    fig.suptitle("Schedule provenance before and after component-level MILP")
    fig.tight_layout(rect=(0, .05, 1, .94))
    save(fig, output, "10_pre_post_milp_provenance", "Paired bars show provenance of selected contacts before and after MILP; numbers above bars are actual contact counts. Each stage has its own denominator because schedule cardinality may change. These are provenance shares, not identities carried forward: MILP replaces a component solution and can select different contacts. Full pre/post counts and weights are exported in comparison_metrics.csv.", captions)


def sampling_tradeoff(data, output, captions):
    profiles = [("n20_c100", "20/100", "^"), ("baseline", "30/200", "o"), ("n50_c350", "50/350", "s")]
    fig, ax = plt.subplots(figsize=(10, 6))
    offsets = {"MWIS": [(-20, 35), (-15, -35), (35, 10)], "CSP": [(-25, 35), (35, -25), (0, 25)]}
    for method in COLORS:
        ordered = [data.loc[(key, method)] for key, _, _ in profiles]
        ax.plot([r.total_reads for r in ordered], [r.pre_milp_gap_to_certified_optimum_percent for r in ordered], color=COLORS[method], alpha=.45, linewidth=1)
        for index, (key, label, marker) in enumerate(profiles):
            row = data.loc[(key, method)]
            ax.scatter(row.total_reads, row.pre_milp_gap_to_certified_optimum_percent, marker=marker, s=70, color=COLORS[method], edgecolors="white", linewidth=.6, zorder=3)
            ax.annotate(label, (row.total_reads, row.pre_milp_gap_to_certified_optimum_percent), xytext=offsets[method][index], textcoords="offset points", ha="center", fontsize=9,
                        arrowprops={"arrowstyle": "-", "color": COLORS[method], "linewidth": .7})
    ax.set_xscale("log")
    ax.set_xlim(10_000, 850_000)
    ax.set_ylim(-.18, 2.6)
    ax.set_xlabel("Total simulated-annealing reads · log scale")
    ax.set_ylabel("Pre-MILP gap to certified optimum (%)")
    ax.legend(handles=[Line2D([], [], color=COLORS[m], marker="o", label=m) for m in COLORS], loc="upper right")
    ax.set_title("Sampling cost versus quality · same frozen 200-satellite/50-gateway input")
    fig.tight_layout()
    save(fig, output, "11_terminal_profile_cost_quality", "Only the terminal-sensitivity profiles on the identical frozen 200/50 input are compared. Labels are logical variable/coupler limits. Total reads measure sampling workload, excluding classical reduction/decomposition/verification cost. Per-terminal budgets differ (MWIS 500, CSP 1,000), and changing both caps changes the decomposition itself; this is an observed profile tradeoff, not an isolated effect of increasing reads or a runtime Pareto claim.", captions)


def leaf_size_distributions(leaves, output, captions):
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    largest = max(1, leaves.contacts_or_binary_variables.max())
    for ax, scenario in zip(axes.flat, SCENARIOS):
        subset = leaves[leaves.scenario == scenario.key]
        for method in COLORS:
            for kind, linestyle in [("terminal", "-"), ("incomplete", "--")]:
                values = subset[(subset.method == method) & (subset.leaf_kind == kind)].contacts_or_binary_variables.to_numpy()
                if not len(values):
                    continue
                assert (values > 0).all()
                unique, counts = np.unique(values, return_counts=True)
                x = np.r_[.8, unique, largest * 1.3]
                y = np.r_[0, np.cumsum(counts) / len(values), 1]
                ax.step(x, y, where="post", color=COLORS[method], linestyle=linestyle, linewidth=1.5)
        ax.set_xscale("log")
        ax.set_xlim(.8, largest * 1.3)
        ax.set_ylim(-.02, 1.03)
        ax.set_title(f"{scenario.satellites}/{scenario.gateways} · {scenario.variables}/{scenario.couplers}", fontsize=10)
        ax.set_xlabel("Contacts / binary variables · log scale", fontsize=8)
        ax.set_ylabel("Cumulative fraction")
    axes.flat[-1].axis("off")
    handles = [Line2D([], [], color=COLORS[m], linestyle=ls, label=f"{m} · {kind}") for m in COLORS for kind, ls in [("terminal", "-"), ("incomplete", "--")]]
    axes.flat[-1].legend(handles=handles, loc="upper left", fontsize=9)
    axes.flat[-1].text(.02, .52, "All tree leaves / branches\nEmpty fixed-choice leaves excluded\n\nCurrent encodings use one\nbinary variable per remaining\ncontact alternative.\n\nCSP 100/50 has no\nincomplete leaves.", transform=axes.flat[-1].transAxes, va="top", fontsize=10)
    fig.suptitle("Terminal and unresolved-leaf size distributions (ECDF)")
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, output, "12_leaf_size_distributions", "Empirical cumulative distributions over all recorded non-empty terminal leaves and incomplete leaves, including alternative separator states; these are workload distributions, not repeated-run uncertainty. Sizes are remaining candidate contacts/binary variables, excluding fixed choices. In the current one-hot/domain-wall encodings one bit is used per contact alternative. Common logarithmic x limits expose large unresolved pieces concealed by average terminal size. Missing incomplete leaves are not plotted as zero-sized observations. Couplers missing in CSP incomplete-leaf metadata remain blank in leaf_metrics.csv.", captions)


def schedule_gantt(schedules, gateways, scenario, output, captions):
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    chosen = schedules[schedules.scenario == scenario.key]
    station_ids = sorted(gateways[gateways.scenario == scenario.key].station_id.unique())
    positions = {station: index for index, station in enumerate(station_ids)}
    height = max(6, .14 * len(station_ids) + 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(15, height), sharex=True, sharey=True)
    norm, cmap = Normalize(0, 100), plt.get_cmap("viridis")
    assert chosen.weight.between(0, 100).all()
    for ax, method in zip(axes, COLORS):
        method_schedule = chosen[chosen.method == method]
        for station, group in method_schedule.groupby("station_id"):
            y = positions[station]
            ordered = group.sort_values("start_time_hours")
            rows = list(ordered.itertuples(index=False))
            for index, row in enumerate(rows):
                ax.broken_barh([(row.start_time_hours, row.end_time_hours - row.start_time_hours)], (y - .34, .68),
                              facecolors=cmap(norm(row.weight)), edgecolors="none")
                if index + 1 < len(rows):
                    required = max(row.reconfiguration_time_hours, rows[index + 1].reconfiguration_time_hours)
                    available = rows[index + 1].start_time_hours - row.end_time_hours
                    assert available + 1e-9 >= required
                    ax.broken_barh([(row.end_time_hours, min(required, available))], (y - .34, .68),
                                  facecolors="#E1E5E8", edgecolors="#B7BEC4", hatch="////", linewidth=.2)
        ax.set_xlim(0, HORIZON_HOURS)
        ax.set_xticks(np.arange(0, HORIZON_HOURS + .01, .5))
        ax.set_xlabel("Hours from the beginning of the scheduling horizon")
        ax.set_title(f"{method} · {len(method_schedule)} contacts · weight {method_schedule.weight.sum():,.3f}", fontsize=11)
        ax.grid(axis="y", visible=False)
    axes[0].set_ylim(len(station_ids) - .4, -.6)
    axes[0].set_yticks(range(len(station_ids)), station_ids, fontsize=6)
    axes[0].set_ylabel("Gateway / single-antenna resource")
    axes[1].tick_params(axis="y", labelleft=False)
    fig.suptitle(f"Final schedule · {scenario.satellites} satellites / {scenario.gateways} gateways · limits {scenario.variables}/{scenario.couplers}", y=.98)
    fig.subplots_adjust(left=.11, right=.9, bottom=.085, top=.90, wspace=.08)
    colorbar_axis = fig.add_axes([.92, .14, .012, .68])
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_axis, label="Selected contact weight")
    fig.text(.5, .025, "Colored blocks: scheduled contacts. Gray hatching: minimum reconfiguration interval before the next contact. Empty rows: unused gateways.", ha="center", fontsize=9)
    save(fig, output, f"13_schedule_gantt_{scenario.key}", f"Matched final MWIS/CSP schedules for {scenario.satellites}/{scenario.gateways}, terminal limits {scenario.variables}/{scenario.couplers}. Both panels use the same gateway order and frozen four-hour horizon. Contact colors share the 0–100 operational-weight scale. Required setup intervals are drawn only between successive selected contacts, excluding a trailing reservation after the last contact. Empty gateways remain visible. Equal objectives need not imply identical contact choices or timings. This is the simplified single-antenna, at-most-one-contact-per-satellite model.", captions)


def gateway_activity(gateways, scenario, output, captions):
    from matplotlib.ticker import MaxNLocator
    subset = gateways[gateways.scenario == scenario.key]
    stations = sorted(subset.station_id.unique())
    fig, ax = plt.subplots(figsize=(max(11, .15 * len(stations)), 6))
    x = np.arange(len(stations))
    for offset, method in [(-.19, "MWIS"), (.19, "CSP")]:
        rows = subset[subset.method == method].set_index("station_id").loc[stations]
        ax.bar(x + offset, rows.selected_contacts, .36, color=COLORS[method], label=method)
    step = max(1, int(np.ceil(len(stations) / 25)))
    ax.set_xticks(x[::step], [stations[i] for i in x[::step]], rotation=60, ha="right", fontsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Final scheduled contacts")
    ax.set_xlabel("Gateway ID · all gateways plotted; labels thinned when necessary")
    ax.set_title(f"Contacts per gateway · {scenario.satellites}/{scenario.gateways} · limits {scenario.variables}/{scenario.couplers}")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2)
    # All contacts in this campaign last exactly four minutes. Convert the same
    # bars to service occupancy; setup/idle/visibility time is not counted.
    factor = 100 * 4 / (60 * HORIZON_HOURS)
    for _, row in subset.iterrows():
        assert abs(row.contact_minutes - 4 * row.selected_contacts) < 1e-5
    secondary = ax.secondary_yaxis("right", functions=(lambda count: count * factor, lambda percent: percent / factor))
    secondary.set_ylabel("Contact service time / four-hour horizon (%)")
    fig.tight_layout()
    save(fig, output, f"14_gateway_contacts_{scenario.key}", f"Final selected contacts per gateway for {scenario.satellites}/{scenario.gateways}, limits {scenario.variables}/{scenario.couplers}. Zero-contact gateways remain included. Since every selected contact lasts four minutes and each station has one antenna, the right axis converts counts to service occupancy over the four-hour horizon. Occupancy excludes setup, visibility availability and idle time; low occupancy is constrained by the one-contact-per-satellite model and is not a measurement of real Starlink network efficiency. Contact counts, minutes, weight and occupancy for every gateway are exported in gateway_metrics.csv.", captions)


def write_readme(output, frame, captions):
    baseline = frame[(frame.scenario == "baseline") & (frame.method == "MWIS")].iloc[0]
    pre = {origin: int(baseline[f"pre_milp_contacts_{origin.lower().replace(' ', '_')}"]) for origin in ORIGINS}
    figures = "\n\n".join(f"### {name}\n\n![{name}]({name}.png)\n\n{caption}\n\n[Vector PDF]({name}.pdf)" for name, caption in captions.items())
    text = f'''# CSP and MWIS comparison figures

Generated by `results/5_csp_mwis_comparison_plots.py`. This report uses only the completed MWIS campaign `new_pipeline_results_reordered` and the team's extracted CSP runs on `exp_snapshot_2026-10-12`. The dominance-enhanced MWIS baseline is excluded. No experiment is executed or modified.

## Reading the comparison

- All seven configurations are matched by frozen input, satellite/gateway counts, and logical terminal limits. Each of the 14 method/configuration records is loaded once.
- The shared baseline is MWIS `exp1_s200_g50` and CSP `csp_gs_sweep/gs_s200_g50`. It appears in each experiment group as a common reference, not an independent replicate. The other CSP baseline copies are not additional trials.
- CSP uses domain-wall encoding and dominance reduction; this MWIS campaign uses one-hot encoding without dominance. The figures compare the implemented pipelines, not isolated formulation effects.
- MWIS uses 500 reads per terminal and CSP uses 1,000. Total budgets differ further because terminal counts differ. Seeds and one run per setting do not provide statistical error bars.
- Objective means original schedule weight. QUBO energy is not compared across encodings or reward scales.
- Average subproblem size means binary variables/couplers across every non-empty sampled terminal, including alternative branches later discarded. It does not mix native CSP satellite variables with MWIS contact vertices.
- Dimensions are logical QUBO dimensions. Physical embedding qubits, chain lengths and chain breaks are unavailable because these experiments used classical simulated annealing.
- Hardware/runtime environments were not matched, and CSP hardware/environment provenance is incomplete. Runtime plots describe recorded executions; they do not establish a controlled algorithmic speedup.
- The CSP 100/50 run lacks full-notebook timing. Its producer stopwatch is plotted as partial and its total remains missing. No zero, estimate or reconstructed timing is substituted. No frozen data-generation time is included.
- Producer timing partitions use disjoint logged phases plus unprofiled remainder. Total time sums notebook elapsed times and excludes outer launcher/orchestration overhead. CSP does not record sampling separately, so both methods use a combined annealing/reconstruction notebook remainder after subtracting recorded MILP/LP solver time. Nested component clocks are excluded to avoid double counting.

## Why few sampled QUBOs can still give a high pre-MILP objective

Pre-MILP reconstruction combines sampled terminals, exact fixed contacts, separator choices and greedy incomplete leaves. In the original MWIS baseline, the pre-MILP schedule contains {pre['Annealing']} contacts from sampled terminals, {pre['Exact preprocessing']} from exact preprocessing, {pre['Separator assignments']} from separator assignments, and {pre['Greedy fallback']} from greedy fallback. Their combined weight is {baseline.pre_milp_objective:,.3f}; this is not an annealer-only objective. MILP later replaces the incomplete component. A QUBO count therefore cannot by itself explain schedule quality or retained annealing contribution.

## Outputs and regeneration

Run with the project Python interpreter:

```powershell
& 'C:/Users/felip/AppData/Local/Python/pythoncore-3.13-64/python.exe' -B -X utf8 results/5_csp_mwis_comparison_plots.py
```

Use `--output-dir PATH` for another analysis directory. Figures are saved as 200-DPI PNGs and vector PDFs. By default, Gantt and gateway-activity views are generated for all seven configurations; use `--schedule-scenarios baseline` to generate those views only for the baseline, or pass several scenario keys. The gallery and captions list only figures generated by the current invocation; older files in the analysis directory are preserved.

`comparison_metrics.csv` contains run-level measurements and all three quality stages; `terminal_metrics.csv` contains each terminal and its retention flag; `leaf_metrics.csv` and `leaf_size_distribution_summary.csv` contain terminal/unresolved-leaf sizes; `schedule_metrics.csv` contains final selected contacts; `gateway_metrics.csv` contains counts, weights, contact minutes and service occupancy for every gateway. `figure_captions.json` contains definitions/limitations, and `source_manifest.json` records exact source hashes and selected run paths. `validation_report.json` records checks. Missing numeric data remain blank in CSV and null in JSON. Inputs are hash-checked again after plotting to confirm they were not modified.

## Other useful analyses

The newly implemented additions are: three-stage quality (09), pre/post-MILP provenance (10), terminal-profile cost/quality (11), terminal and incomplete-leaf ECDFs (12), final-schedule Gantt charts (13), and contacts/service occupancy per gateway (14). Figures 13 and 14 have one matched CSP/MWIS view for each requested configuration.

ECDFs describe recorded tree workloads, not uncertainty across repeated runs. The cost/quality plot compares only terminal profiles on the same frozen 200/50 dataset and uses total reads rather than uncontrolled runtime. Gantt charts use a common four-hour horizon and include unused gateways. Service occupancy counts scheduled contact time only; the one-contact-per-satellite restriction limits it, so low occupancy cannot be interpreted as real-network inefficiency.

Further useful figures require additional measurements: repeated-run distributions for timing and quality; a matched-hardware/matched-read-budget runtime study; and real-QPU embedding overhead/chain lengths.

## Figures

{figures}
'''
    (output / "README.md").write_text(text, encoding="utf-8")
    import html
    def card(name, caption):
        return f'<article id="{html.escape(name)}"><h2>{html.escape(name)}</h2><a href="{name}.pdf"><img src="{name}.png" alt="{html.escape(name)}"></a><p>{html.escape(caption)}</p></article>'
    comparative = "".join(card(name, caption) for name, caption in captions.items() if not name.startswith(("13_", "14_")))
    operational = "".join(card(name, caption) for name, caption in captions.items() if name.startswith(("13_", "14_")))
    navigation = '<ul>' + ''.join(f'<li><a href="#{html.escape(name)}">{html.escape(name)}</a></li>' for name in captions if not name.startswith(("13_", "14_"))) + '</ul>'
    (output / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>CSP / MWIS experiment comparison</title><style>body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 24px;color:#25313c}article{margin:36px 0;padding:20px;border:1px solid #d4dce3;border-radius:10px}img{max-width:100%;height:auto}p{line-height:1.5}summary{font-size:22px;cursor:pointer;padding:16px;background:#edf2f6}</style><h1>CSP / MWIS experiment comparison</h1><p>Completed reordered MWIS campaign and matched CSP inputs. Dominance-enhanced MWIS excluded. Logical dimensions only; runtime is descriptive with unmatched hardware/environment provenance. Click any figure for its vector PDF. See README.md and the exported tables for full definitions.</p>' + navigation + comparative + '<details><summary>Schedule Gantt charts and gateway activity for each configuration</summary>' + operational + '</details></html>', encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--schedule-scenarios", nargs="+", choices=[s.key for s in SCENARIOS] + ["all"], default=["all"],
                        help="Configurations for Gantt/gateway plots; default: all. Comparative figures always cover all configurations.")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output == ROOT or output == RESULTS or any(base in output.parents or output == base for base in [RESULTS / "runs/mwis", RESULTS / "runs/csp", RESULTS / "runs/mwis-dominance-enhanced"]):
        parser.error("Choose an analysis directory outside the source campaign folders")
    inputs = Inputs()
    records, terminals, leaves, schedules, gateways, selections = [], [], [], [], [], []
    for scenario in SCENARIOS:
        print(f"Loading matched {scenario.key} ({scenario.satellites}/{scenario.gateways}, limits {scenario.variables}/{scenario.couplers})", flush=True)
        for method in COLORS:
            record, terminal_frame, leaf_frame, schedule_frame, gateway_frame = load_run(inputs, scenario, method)
            records.append(record)
            terminals.append(terminal_frame)
            leaves.append(leaf_frame)
            schedules.append(schedule_frame)
            gateways.append(gateway_frame)
            selections.append({"scenario": scenario.key, "method": method, "run": record["run"]})
        a, b = records[-2:]
        assert a["input_sha256"] == b["input_sha256"]
        assert abs(a["final_objective"] - b["final_objective"]) < 1e-6
    frame = pd.DataFrame(records)
    assert len(frame) == 14 and not frame[["scenario", "method"]].duplicated().any()
    terminal_frame = pd.concat(terminals, ignore_index=True)
    leaf_frame = pd.concat(leaves, ignore_index=True)
    schedule_frame = pd.concat(schedules, ignore_index=True)
    gateway_frame = pd.concat(gateways, ignore_index=True)
    data = frame.set_index(["scenario", "method"])
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "comparison_metrics.csv", index=False)
    terminal_frame.to_csv(output / "terminal_metrics.csv", index=False)
    leaf_frame.to_csv(output / "leaf_metrics.csv", index=False)
    leaf_summary = leaf_frame.groupby(["scenario", "method", "leaf_kind"]).contacts_or_binary_variables.agg(
        count="size", minimum="min", q25=lambda x: x.quantile(.25), median="median", q75=lambda x: x.quantile(.75), maximum="max")
    leaf_summary.to_csv(output / "leaf_size_distribution_summary.csv")
    schedule_frame.to_csv(output / "schedule_metrics.csv", index=False)
    gateway_frame.to_csv(output / "gateway_metrics.csv", index=False)
    (output / "comparison_metrics.json").write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")
    style()
    captions = {}
    for plotter, source in [(objectives, data), (mean_sizes, data), (runtimes, data),
                            (logical_dimensions, terminal_frame), (quality_gaps, data),
                            (provenance_plot, data), (workload, data), (component_structure, data),
                            (quality_stages, data), (pre_post_provenance, data),
                            (sampling_tradeoff, data), (leaf_size_distributions, leaf_frame)]:
        print(f"Generating {plotter.__name__}", flush=True)
        plotter(source, output, captions)
    schedule_scenarios = SCENARIOS if "all" in args.schedule_scenarios else [s for s in SCENARIOS if s.key in args.schedule_scenarios]
    for scenario in schedule_scenarios:
        print(f"Generating schedule/gateway views for {scenario.key}", flush=True)
        schedule_gantt(schedule_frame, gateway_frame, scenario, output, captions)
        gateway_activity(gateway_frame, scenario, output, captions)
    write_readme(output, frame, captions)
    inputs.verify_unchanged()
    (output / "source_manifest.json").write_text(json.dumps({"selected_runs": selections, "files_sha256": inputs.manifest,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "python": sys.version,
        "platform": platform.platform(), "libraries": {"pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__}}, indent=2), encoding="utf-8")
    (output / "figure_captions.json").write_text(json.dumps(captions, indent=2), encoding="utf-8")
    report = {"passed": True, "matched_configurations": 7, "method_configuration_records": 14,
              "exported_terminal_records": len(terminal_frame), "figures_png_and_pdf": len(captions),
              "exported_leaf_records": len(leaf_frame), "exported_selected_contact_records": len(schedule_frame),
              "operational_plot_scenarios": [s.key for s in schedule_scenarios],
              "source_hashes_checked_unchanged": len(inputs.manifest), "dominance_variant_included": False,
              "missing_full_runtime_records": frame.loc[~frame.total_timing_complete, ["scenario", "method"]].to_dict("records"),
              "checks": ["matching input hashes and requested active satellite/gateway counts", "terminal caps and read budgets",
                         "sample feasibility flags", "independent pre-MILP reconstruction", "original schedule constraints and objective totals",
                         "final objective/upper-bound certificates", "component-level MILP provenance and retained terminal counts",
                         "matching final optima across formulations", "classical reference schedules and objective totals",
                         "leaf-size units and terminal/incomplete counts", "gateway counts/weights and four-hour service occupancy",
                         "single-antenna resources and required setup intervals", "disjoint phase clocks", "one shared baseline", "unchanged source hashes"]}
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Figures and tables: {output}")
    print(frame[["scenario", "method", "pre_milp_objective", "final_objective", "terminal_qubos", "total_reads"]].to_string(index=False))


if __name__ == "__main__":
    main()
