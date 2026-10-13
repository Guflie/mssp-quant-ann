"""Generate report-size COP/MWIS figures for primary schedules before reference solving.

Run with Python 3.13.14 from any working directory. No experiment is executed.
Only figures and analysis tables are written; source runs are read-only.
Dependencies: numpy, pandas, matplotlib, networkx.
"""
from __future__ import annotations

import argparse
import fnmatch
import gzip
import hashlib
import json
import platform
import subprocess
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass, replace
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
DEFAULT_OUTPUT = RESULTS / "figures/csp_mwis_primary"
CSP_ARCHIVE = RESULTS / "runs/csp_final.zip"
MWIS_VARIANT = "dominance"
COLORS = {"MWIS": "#2463A0", "COP": "#C75C30"}
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
# The final CSP archive supplies every CSP run and its saved notebook clocks.
# The gateway sweep baseline is reused across all experiment groups.
SCENARIOS = [
    Scenario("s100_g50", 100, 50, 30, 200, "exp2_s100_g50", "csp_sat_sweep", "ds_s100_g50"),
    Scenario("baseline", 200, 50, 30, 200, "baseline_s200_g50", "csp_gs_sweep", "gs_s200_g50"),
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
    def __init__(self, source_git_ref=None):
        self.manifest = {}
        self.source_git_ref = None
        if source_git_ref:
            self.source_git_ref = subprocess.check_output(
                ["git", "rev-parse", "--verify", source_git_ref + "^{commit}"],
                cwd=ROOT, text=True).strip()
        from io import BytesIO
        self.csp_zip = zipfile.ZipFile(BytesIO(self.read(CSP_ARCHIVE)))

    def _bytes(self, relative):
        if "!" in relative:
            archive, member = relative.split("!", 1)
            assert archive == CSP_ARCHIVE.relative_to(ROOT).as_posix()
            return self.csp_zip.read(member)
        pinned = {"results/runs/csp_final.zip", "results/figures/csp_mwis_reordered_dominance/comparison_metrics.csv"}
        if self.source_git_ref and relative in pinned:
            return subprocess.check_output(["git", "show", self.source_git_ref + ":" + relative], cwd=ROOT)
        return (ROOT / relative).read_bytes()

    def read(self, path: Path):
        relative = path.relative_to(ROOT).as_posix()
        prefix = "results/runs/csp/"
        if relative.startswith(prefix):
            member = "csp/" + relative[len(prefix):]
            relative = CSP_ARCHIVE.relative_to(ROOT).as_posix() + "!" + member
        data = self._bytes(relative)
        self.manifest[relative] = hashlib.sha256(data).hexdigest()
        return data

    def glob(self, directory, pattern):
        prefix = "results/runs/csp/"
        relative = directory.relative_to(ROOT).as_posix() + "/"
        if relative.startswith(prefix):
            member_dir = "csp/" + relative[len(prefix):]
            return [directory / name[len(member_dir):] for name in self.csp_zip.namelist()
                    if name.startswith(member_dir) and "/" not in name[len(member_dir):]
                    and fnmatch.fnmatch(name[len(member_dir):], pattern)]
        return list(directory.glob(pattern))

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
            if hashlib.sha256(self._bytes(relative)).hexdigest() != expected:
                raise AssertionError(f"Source changed while plotting: {relative}")


def number(metrics, key):
    if key not in metrics:
        raise KeyError(f"Missing required metric {key!r}")
    return float(metrics[key])


def phase_times(inputs, producer):
    files = inputs.glob(producer, "*phase_profile_*.jsonl")
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
        run = scenario.mwis_run
        base = RESULTS / "runs/mwis-dominance-enhanced"
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
    assert ("dominance-enhanced" in str(producer)) == (method == "MWIS" and MWIS_VARIANT == "dominance")
    config = inputs.json(producer / "run_config_log.json")
    config_input = config.get("candidate_slot_csv") if method == "MWIS" else config.get("csp_slot_csv", config.get("slot_csv"))
    assert (ROOT / config_input).resolve() == scenario.input_path.resolve()
    sample_config = inputs.json(annealer / "run_config_log.json")
    assert int(sample_config["max_qpu_terminal_nodes"]) == scenario.variables
    assert int(sample_config["max_qpu_terminal_couplers"]) == scenario.couplers
    metadata = inputs.metadata(producer / "reconstruction_metadata.json.gz")
    if method == "MWIS" and MWIS_VARIANT == "dominance":
        assert "dominance" in metadata["variant"]
        validation = inputs.json(base / "execution_logs" / run / "validation_report.json")
        assert validation["validation_passed"]
    producer_metrics = inputs.metrics(producer / "final_summary.csv")
    final_metrics = inputs.metrics(annealer / "reconstructed_summary.csv")
    terminals = inputs.csv(annealer / "terminal_qubo_summary.csv")
    sampled = inputs.csv(annealer / "annealing_results.csv")
    schedule = inputs.csv(annealer / "reconstructed_schedule.csv")
    provenance = inputs.csv(annealer / "component_provenance_summary.csv").set_index("component_id")
    graph_nodes = inputs.csv(producer / "conflict_graph_nodes.csv")
    graph_edges = inputs.csv(producer / "conflict_graph_edges.csv")
    reads = int(sample_config["num_reads"])
    assert reads == 500, "The final matched campaign uses 500 reads per terminal for both methods"
    assert len(sampled) == len(terminals) == int(number(final_metrics, "Annealed terminal subgraphs available"))
    assert set(sampled.terminal_id) == set(terminals.terminal_id)
    assert (sampled.num_reads == reads).all() and (sampled.is_independent.astype(str).str.lower() == "true").all()
    assert terminals.variables.max() <= scenario.variables and terminals.couplers.max() <= scenario.couplers
    samples = {r["terminal_id"]: r for r in sampled.to_dict("records")}
    before = trace_reconstruction(metadata, samples)
    pre_ids = [s for result in before for s in result["origins"]]
    primary_terminals = set().union(*(r["terminal_ids"] for r in before))
    assert primary_terminals <= set(sampled.terminal_id)
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
        if MWIS_VARIANT == "dominance":
            build += phases["global_dominance_and_export"]
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
        "mwis_variant": MWIS_VARIANT if method == "MWIS" else "not applicable",
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
    terminal_frame["retained_in_primary_schedule"] = terminal_frame.terminal_id.isin(primary_terminals)
    terminal_frame["retained_in_reference_postprocessed_schedule"] = terminal_frame.terminal_id.isin(retained_terminals)
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
    schedule_frame = raw.loc[sorted(pre_ids), ["slot_id", "satellite_id", "station_id", "antenna_id", "start_time_hours", "end_time_hours", "reconfiguration_time_hours", "weight"]].copy().reset_index(drop=True)
    schedule_frame["scenario"] = scenario.key
    schedule_frame["method"] = method
    schedule_frame["origin"] = schedule_frame.slot_id.map(pre_origins)
    schedule_frame["schedule_stage"] = "primary_before_reference_evaluation"
    gateway_records = []
    for station_id in sorted(raw.station_id.unique()):
        selected_here = schedule_frame[schedule_frame.station_id == station_id]
        minutes = 60 * (selected_here.end_time_hours - selected_here.start_time_hours).sum()
        gateway_records.append({"scenario": scenario.key, "method": method, "station_id": station_id,
                                "selected_contacts": len(selected_here), "selected_weight": selected_here.weight.sum(),
                                "contact_minutes": minutes, "service_utilization_percent": 100 * minutes / (60 * HORIZON_HOURS),
                                "horizon_hours": HORIZON_HOURS})
    gateway_frame = pd.DataFrame(gateway_records)
    assert gateway_frame.selected_contacts.sum() == len(pre_ids)
    assert abs(gateway_frame.selected_weight.sum() - pre_objective) < 1e-6
    assert (gateway_frame.service_utilization_percent <= 100 + 1e-6).all()
    record["reduced_slot_graph_sha256"] = hashlib.sha256(json.dumps({
        "nodes": sorted((str(r.slot_id), float(r.weight)) for r in graph_nodes.itertuples()),
        "edges": sorted(tuple(sorted((str(r.source), str(r.target)))) for r in graph_edges.itertuples()),
    }, sort_keys=True).encode()).hexdigest()
    record = primary_record(record, primary_terminals, execution if method == "MWIS" else None)
    return record, terminal_frame, leaf_frame, schedule_frame, gateway_frame


PRIMARY_ORIGINS = ORIGINS[:-1]
ORIGIN_LABELS = ["Sampled terminals", "Exact preprocessing", "Separator choices", "Greedy completion"]
CORE_KEYS = ["s100_g50", "baseline", "s500_g50", "s200_g25", "s200_g100"]
WIDE = 7.05  # 179 mm: full-width figure in an A4 two-column report.
NARROW = 3.4  # 86 mm: single-column report figures.
ORIGINAL_SCRIPT = RESULTS / "5_csp_mwis_comparison_plots.py"
OLD_COMPARISON = RESULTS / "figures/csp_mwis_reordered_dominance/comparison_metrics.csv"


def primary_record(record, retained, execution):
    """Separate the reported schedule, reference results and incomplete clocks."""
    names = {
        "retained_terminal_qubos": "reference_postprocessed_retained_terminal_qubos",
        "terminal_retention_percent": "reference_postprocessed_terminal_retention_percent",
        "classical_reference_objective": "producer_reference_objective",
        "sampling_objective_change_from_reference": "primary_minus_producer_reference_objective",
        "classical_gap_to_certified_optimum_percent": "producer_reference_gap_percent",
        "final_objective": "reference_postprocessed_objective",
        "final_upper_bound": "reference_upper_bound",
        "certified_optimal": "reference_certified_optimal",
        "milp_objective_improvement": "reference_improvement_over_primary",
        "final_selected_contacts": "reference_postprocessed_selected_contacts",
        "selected_conflicts": "reference_postprocessed_conflicts",
        "annealing_notebook_seconds": "sampling_reconstruction_reference_notebook_seconds",
        "total_notebook_seconds": "recorded_full_workflow_seconds",
        "total_timing_complete": "recorded_workflow_timing_complete",
        "time_classical_verification": "time_producer_classical_evaluation",
        "time_annealing_reconstruction_remainder": "combined_notebook_remainder_after_solver_calls_seconds",
    }
    out = {}
    for key, value in record.items():
        if key.startswith("pre_milp_"):
            key = "primary_" + key[len("pre_milp_"):]
        elif key.startswith("final_contacts_"):
            key = "reference_postprocessed_contacts_" + key[len("final_contacts_"):]
        elif key.startswith("final_objective_"):
            key = "reference_postprocessed_objective_" + key[len("final_objective_"):]
        else:
            key = names.get(key, key)
        out[key] = value
    out.update({
        "schedule_stage": "primary_before_reference_evaluation", "primary_feasible": True,
        "primary_conflicts": 0, "primary_retained_terminal_qubos": len(retained),
        "primary_terminal_retention_percent": 100 * len(retained) / out["terminal_qubos"],
        "primary_certified_optimal": abs(out["primary_objective"] - out["reference_upper_bound"]) < 1e-6,
        "primary_absolute_gap_to_reference": max(0, out["reference_upper_bound"] - out["primary_objective"]),
        # Complete reconstruction and reference preparation share a saved cell.
        # Neither solver-only subtraction nor the full sequence isolates this cost.
        "isolated_primary_pipeline_seconds": np.nan,
        "sampling_decoding_export_cell_seconds": np.nan,
    })
    if execution is not None:
        stage = next(s for s in execution["stages"] if s["stage"] == "annealing")
        cells = [c for c in stage["cells"] if c["original_cell"] == 5]
        assert len(cells) == 1
        out["sampling_decoding_export_cell_seconds"] = float(cells[0]["seconds"])
    assert out["primary_contacts_milp"] == 0
    assert sum(out[f"primary_contacts_{o.lower().replace(' ', '_')}"] for o in PRIMARY_ORIGINS) == out["primary_selected_contacts"]
    assert abs(sum(out[f"primary_objective_{o.lower().replace(' ', '_')}"] for o in PRIMARY_ORIGINS) - out["primary_objective"]) < 1e-6
    return out


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "figure.titlesize": 10, "legend.fontsize": 8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18,
        "axes.axisbelow": True, "savefig.facecolor": "white", "pdf.fonttype": 42,
    })


def save(fig, output, name, caption, captions, layouts, *, role="report"):
    assert len(fig.axes) <= 2, "Keep each figure to at most two plots"
    width, height = fig.get_size_inches()
    # Fixed canvas preserves the intended physical font size when included at
    # its documented width. No tight-bbox resizing of the final canvas.
    fig.savefig(output / f"{name}.png", dpi=300)
    fig.savefig(output / f"{name}.pdf")
    captions[name] = caption
    layouts[name] = {"role": role, "width_inches": float(width), "height_inches": float(height),
                     "width_mm": round(float(width) * 25.4, 1), "panels": len(fig.axes),
                     "schedule_stage": "primary_before_reference_evaluation"}
    plt.close(fig)


def legend(fig, axes, *, ncol=2):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False)


def paired_bars(ax, data, keys, metric, labels, ylabel, title, *, logarithmic=False):
    x = np.arange(len(keys))
    for offset, method in [(-.19, "MWIS"), (.19, "COP")]:
        values = [float(data.loc[(k, method), metric]) for k in keys]
        ax.bar(x + offset, values, .36, color=COLORS[method], label=method)
        if logarithmic:
            for i, value in enumerate(values):
                if value == 0:
                    ax.text(i + offset, .16, "0", ha="center", fontsize=7)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    if logarithmic:
        ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)


def primary_quality_table(frame):
    records = []
    data = frame.set_index(["scenario", "method"])
    for s in SCENARIOS:
        record = {"scenario": s.key, "satellites": s.satellites, "gateways": s.gateways,
                  "variable_limit": s.variables, "coupler_limit": s.couplers,
                  "reference_optimum": data.loc[(s.key, "MWIS"), "reference_upper_bound"]}
        for m in COLORS:
            row = data.loc[(s.key, m)]
            record[f"{m}_primary_objective"] = row.primary_objective
            record[f"{m}_primary_contacts"] = row.primary_selected_contacts
            record[f"{m}_primary_gap_percent"] = row.primary_gap_to_certified_optimum_percent
        records.append(record)
    return pd.DataFrame(records)


def quality_table_figure(table, output, captions, layouts):
    fig, ax = plt.subplots(figsize=(WIDE, 3.05))
    ax.axis("off")
    cells = []
    for r in table.itertuples():
        cells.append([f"{r.satellites}/{r.gateways}; {r.variable_limit}/{r.coupler_limit}",
                      f"{r.MWIS_primary_objective:,.3f}", f"{r.COP_primary_objective:,.3f}",
                      f"{r.MWIS_primary_gap_percent:.3f}", f"{r.COP_primary_gap_percent:.3f}"])
    chart = ax.table(cellText=cells, colLabels=["Sat./GW; limits", "MWIS weight", "COP weight", "MWIS gap (%)", "COP gap (%)"],
                     colWidths=[.25, .21, .21, .165, .165], cellLoc="center", bbox=[0, .05, 1, .88])
    chart.auto_set_font_size(False)
    chart.set_fontsize(8)
    for (row, col), cell in chart.get_celld().items():
        cell.set_edgecolor("#CED7DF")
        cell.set_linewidth(.5)
        cell.set_facecolor("#E9EFF4" if row == 0 else ("#F7F9FB" if row % 2 == 0 else "white"))
        if row == 0: cell.set_text_props(weight="bold")
    ax.set_title("Primary schedule quality", pad=6)
    fig.tight_layout(pad=.8)
    save(fig, output, "00_primary_quality_table", "Primary reconstructed schedule weights, before MILP/LP reference evaluation. Limits are logical variables/couplers. Gaps are retrospective distances to certified reference optima; all primary schedules are feasible. The shared baseline is listed once. Full-precision values and reference optima are in primary_quality_summary.csv.", captions, layouts)


def quality_table_latex(table, output):
    """Overleaf table fragment; figures and native table use identical rounded data."""
    lines = [r"% Requires \usepackage{booktabs}. Use table* for a two-column report.",
             r"% Change table* to table for a single-column document.",
             r"\begin{table*}[t]", r"  \centering", r"  \small",
             r"  \caption{Primary schedule quality before MILP/LP reference evaluation. Gaps are measured relative to the certified reference optimum. All schedules are feasible; the shared baseline is listed once. Limits denote logical variables/couplers.}",
             r"  \label{tab:primary-quality}",
             r"  \begin{tabular}{lrrrr}", r"    \toprule",
             r"    Sat./GW; limits & MWIS weight & COP weight & MWIS gap (\%) & COP gap (\%) \\",
             r"    \midrule"]
    for row in table.itertuples():
        lines.append(f"    {row.satellites}/{row.gateways}; {row.variable_limit}/{row.coupler_limit} & "
                     f"{row.MWIS_primary_objective:,.3f} & {row.COP_primary_objective:,.3f} & "
                     f"{row.MWIS_primary_gap_percent:.3f} & {row.COP_primary_gap_percent:.3f}" + r" \\")
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table*}", ""])
    (output / "primary_quality_table.tex").write_text("\n".join(lines), encoding="utf-8")


def quality_scaling(data, output, captions, layouts, *, panel=None, column=False):
    if column:
        assert panel in (0, 1), "Column exports contain one scaling experiment"
    groups = GROUPS[:2] if panel is None else [GROUPS[panel]]
    fig, axes = plt.subplots(1, len(groups), figsize=(NARROW if column else WIDE, 2.65 if column else (2.9 if panel is None else 3.4)), squeeze=False)
    axes = axes.ravel()
    for ax, (title, keys, labels, xlabel) in zip(axes, groups):
        for method in COLORS:
            values = [data.loc[(k, method), "primary_gap_to_certified_optimum_percent"] for k in keys]
            ax.plot(range(3), values, color=COLORS[method], marker="o" if method == "MWIS" else "s",
                    markersize=4 if method == "MWIS" else 6, markerfacecolor=COLORS[method] if method == "MWIS" else "none", label=method)
        ax.set_xticks(range(3), labels)
        ax.set_xlabel(xlabel)
        ax.set_title(["Satellite scaling (50 gateways)", "Gateway scaling (200 satellites)"][panel] if column else title.replace(" · ", "\n"))
        ax.set_ylim(0, max(data.loc[(k, m), "primary_gap_to_certified_optimum_percent"] for k in keys for m in COLORS) * 1.15)
    axes[0].set_ylabel("Gap to optimum (%)" if column else "Gap to reference optimum (%)")
    legend(fig, axes)
    fig.tight_layout(rect=(0, .14 if column else .12, 1, .94 if column else 1), pad=.9, w_pad=1.8)
    name = "01_primary_quality_scaling" if panel is None else ["01a_primary_quality_satellite_scaling", "01b_primary_quality_gateway_scaling"][panel]
    if column: name += "_column"
    scope = "Left: satellite scaling; right: gateway scaling." if panel is None else ["Satellite scaling with 50 gateways and limits 30/200.", "Gateway scaling with 200 satellites and limits 30/200."][panel]
    save(fig, output, name, f"Primary schedule gaps to certified reference optima, excluding MILP-generated contacts. {scope} The baseline is reused. Active resource identities are not fully nested, so differences describe the selected scenarios rather than a strictly isolated resource-count effect. Both methods use the same candidate CSV within each pair.", captions, layouts)


def primary_provenance(data, output, captions, layouts):
    fig, ax = plt.subplots(figsize=(WIDE, 3.65))
    for i, key in enumerate(CORE_KEYS):
        for offset, method in [(-.2, "MWIS"), (.2, "COP")]:
            row = data.loc[(key, method)]
            bottom = 0
            for origin, color in zip(PRIMARY_ORIGINS, ORIGIN_COLORS):
                count = row[f"primary_contacts_{origin.lower().replace(' ', '_')}"]
                value = 100 * count / row.primary_selected_contacts
                ax.bar(i + offset, value, .37, bottom=bottom, color=color)
                if value >= 12:
                    ax.text(i + offset, bottom + value / 2, f"{value:.0f}%", ha="center", va="center", fontsize=7,
                            color="white" if origin == "Annealing" else "black")
                bottom += value
            assert abs(bottom - 100) < 1e-6
            ax.text(i + offset, 103, f"{method}\nn={int(row.primary_selected_contacts)}", ha="center", fontsize=7)
    labels = [f"{data.loc[(k, 'MWIS'), 'satellites']}/{data.loc[(k, 'MWIS'), 'gateways']}" for k in CORE_KEYS]
    ax.set_xticks(range(5), labels)
    ax.set_ylabel("Primary selected contacts (%)")
    ax.set_xlabel("Satellites / gateways; limits 30/200")
    ax.set_ylim(0, 121)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("Primary schedule provenance before reference evaluation")
    ax.grid(axis="x", visible=False)
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(ORIGIN_COLORS, ORIGIN_LABELS)],
               loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .17, 1, 1), pad=.9)
    save(fig, output, "02_primary_schedule_provenance", "Origins of contacts retained in the selected pre-MILP reconstruction, with paired MWIS/COP bars for the five distinct datasets. n is each primary schedule's contact count. MILP is not a category. These are contact shares, not objective shares, execution time or a causal measure of sampling advantage. Sampled alternative branches discarded by reconstruction do not contribute. All seven profiles' contact and weight provenance are in comparison_metrics.csv.", captions, layouts)


def terminal_profiles(data, output, captions, layouts, *, panel=None, column=False):
    if column:
        assert panel == 0, "Column export is the terminal-profile quality panel"
    keys = ["n20_c100", "baseline", "n50_c350"]
    labels = ["20/100", "30/200", "50/350"]
    fig, drawn_axes = plt.subplots(1, 2 if panel is None else 1, figsize=(NARROW if column else WIDE, 2.65 if column else (2.95 if panel is None else 3.4)), squeeze=False)
    drawn_axes = list(drawn_axes.ravel())
    axes = drawn_axes if panel is None else [drawn_axes[0] if panel == 0 else None, drawn_axes[0] if panel == 1 else None]
    for method in COLORS:
        if axes[0] is None: break
        values = [data.loc[(k, method), "primary_gap_to_certified_optimum_percent"] for k in keys]
        axes[0].plot(range(3), values, color=COLORS[method], marker="o" if method == "MWIS" else "s", markersize=4, label=method)
        for i, v in enumerate(values):
            offset = (0, 7) if method == "MWIS" else (0, 18)
            alignment = "left" if column and i == 0 else ("right" if column and i == 2 else "center")
            axes[0].annotate(f"{v:.3f}", (i, v), xytext=offset, textcoords="offset points", ha=alignment, fontsize=7, color=COLORS[method])
    if axes[0] is not None:
        axes[0].set_ylabel("Gap to optimum (%)" if column else "Gap to reference optimum (%)")
        axes[0].set_title("Terminal-size sensitivity (200/50)" if column else "Primary schedule quality")
        axes[0].set_ylim(0, 1.65)
    x = np.arange(3)
    for offset, method in [(-.19, "MWIS"), (.19, "COP")]:
        if axes[1] is None: break
        values = [data.loc[(k, method), "total_reads"] / 1000 for k in keys]
        bars = axes[1].bar(x + offset, values, .36, color=COLORS[method], label=method)
        axes[1].bar_label(bars, labels=[f"{v:g}" for v in values], padding=3, fontsize=7)
    if axes[1] is not None:
        axes[1].set_ylabel("Total reads (thousands)")
        axes[1].set_title("Sampling workload")
        axes[1].set_ylim(0, 750)
    for ax in drawn_axes:
        ax.set_xticks(range(3), labels)
        ax.set_xlabel("Logical limits (variables/couplers)" if column else "Logical variable/coupler limits")
        ax.grid(axis="x", visible=False)
    legend(fig, drawn_axes)
    fig.tight_layout(rect=(0, .14 if column else .12, 1, .94 if column else 1), pad=.9, w_pad=1.8)
    name = "03_terminal_profile_tradeoff" if panel is None else ["03a_terminal_profile_primary_quality", "03b_terminal_profile_sampling_workload"][panel]
    if column: name += "_column"
    scope = "Left: primary quality gap; right: total sampled reads over all exported alternative terminal branches." if panel is None else ["Primary schedule gap to the certified reference optimum, before MILP evaluation.", "Total sampled reads over all exported alternative terminal branches, including branches discarded in reconstruction."][panel]
    save(fig, output, name, f"Same frozen 200/50 input for all three combined limit profiles. {scope} Both methods use 500 reads per terminal. Read counts are sampling workload, not total runtime. Both logical limits change together; changing them also changes the decomposition.", captions, layouts)


def logical_dimensions(terminals, output, captions, layouts):
    fig, ax = plt.subplots(figsize=(NARROW, 3.05))
    for method in COLORS:
        t = terminals[(terminals.scenario == "baseline") & (terminals.method == method)]
        counts = t.groupby(["variables", "couplers"]).size().reset_index(name="count")
        ax.scatter(counts.variables, counts.couplers, s=12 + 4 * np.sqrt(counts["count"]),
                   facecolors=COLORS[method] if method == "MWIS" else "none", edgecolors=COLORS[method],
                   linewidths=.7, alpha=.7, marker="o" if method == "MWIS" else "s", label=method)
    ax.axvline(30, linestyle="--", color="#777", linewidth=.7)
    ax.axhline(200, linestyle="--", color="#777", linewidth=.7)
    ax.set(xlim=(0, 33), ylim=(-4, 220), xlabel="Logical binary variables", ylabel="Nonzero quadratic couplers",
           title="Baseline terminal QUBOs (30/200)")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(pad=.85)
    save(fig, output, "04_baseline_logical_dimensions", "All sampled baseline terminal QUBOs, including alternative branches. Marker area increases with repeated dimension pairs; dashed lines are logical limits. One-hot MWIS and domain-wall COP counts use common binary-variable/coupler units. These are not physical qubits, chains or embedding overhead; no hardware embedding was performed. Include this PDF at 86 mm width.", captions, layouts)


def producer_profile(data, output, captions, layouts):
    fig, ax = plt.subplots(figsize=(WIDE, 3.65))
    sections = [("time_build_reduction", "Load / build / reduction", "#6AAED6"),
                ("time_decomposition", "Separator decomposition", "#2463A0"),
                ("time_producer_classical_evaluation", "Classical producer evaluation", "#9C79B4"),
                ("time_qubo_exports", "QUBO / producer exports", "#76B7B2")]
    for i, s in enumerate(SCENARIOS):
        for offset, method in [(-.2, "MWIS"), (.2, "COP")]:
            row = data.loc[(s.key, method)]
            total = sum(row[k] for k, _, _ in sections)
            assert total > 0
            bottom = 0
            for key, _, color in sections:
                value = 100 * row[key] / total
                ax.bar(i + offset, value, .37, bottom=bottom, color=color)
                if value >= 18:
                    ax.text(i + offset, bottom + value / 2, f"{value:.0f}%", ha="center", va="center", fontsize=6.8, color="white")
                bottom += value
            ax.text(i + offset, 103, method, ha="center", fontsize=6.8)
    ax.set_xticks(range(7), [s.label for s in SCENARIOS], fontsize=7)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylim(0, 114)
    ax.set_ylabel("Share of profiled producer time (%)")
    ax.set_title("Producer computation profile")
    ax.grid(axis="x", visible=False)
    fig.legend(handles=[Patch(color=c, label=l) for _, l, c in sections], loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .18, 1, 1), pad=.9)
    save(fig, output, "05_producer_stage_profile", "Shares of four disjoint logged producer phases, excluding unprofiled notebook overhead, sampling, reconstruction and MILP/LP solver calls. Classical producer evaluation includes exact terminal reference evaluation and incomplete-leaf evaluation; it is displayed separately and is not sampling. Recursive reductions lie within decomposition. Nested component/operation clocks are not added again. Absolute seconds are in timing_records.csv; shares do not establish a controlled speed comparison. This is not a complete primary-pipeline runtime.", captions, layouts)


def supplementary_workload(data, output, captions, layouts, *, panel=None):
    keys = [s.key for s in SCENARIOS]
    labels = [s.label for s in SCENARIOS]
    specs = [("primary_terminal_retention_percent", "Percent", "Sampled terminal IDs retained in primary reconstruction", False),
             ("incomplete_leaves", "Leaves (symlog)", "Incomplete leaves across all recorded branches", True)]
    chosen = specs if panel is None else [specs[panel]]
    fig, axes = plt.subplots(len(chosen), 1, figsize=(WIDE, 4.8 if panel is None else 3.6), squeeze=False)
    axes = axes.ravel()
    for ax, (metric, ylabel, title, logarithmic) in zip(axes, chosen):
        paired_bars(ax, data, keys, metric, labels, ylabel, title, logarithmic=logarithmic)
    legend(fig, axes)
    fig.tight_layout(rect=(0, .06 if panel is None else .12, 1, 1), pad=.9, h_pad=1.4)
    name = "S01_primary_retention_and_incomplete_leaves" if panel is None else ["S01a_primary_terminal_retention", "S01b_incomplete_leaves"][panel]
    caption = "Retention is evaluated before MILP: terminal IDs present in the selected reconstruction divided by all sampled terminal IDs. It is not a contact share. Incomplete leaves count all recorded alternatives, not just the selected branch. Zero incomplete leaves are shown explicitly. The shared baseline appears once."
    if panel == 0: caption = "Fraction of sampled terminal IDs retained in the selected primary reconstruction, before MILP replacement. It is not a contact or objective share, and does not measure sampling workload saved. The shared baseline appears once."
    if panel == 1: caption = "Incomplete leaves across all recorded decomposition branches, including alternatives discarded in primary reconstruction. The symlog scale shows zero leaves explicitly. Counts are bounded-tree workload, not counts of natural connected components. The shared baseline appears once."
    save(fig, output, name, caption, captions, layouts, role="supplementary")


def supplementary_means(data, output, captions, layouts, *, panel=None):
    keys, labels = [s.key for s in SCENARIOS], [s.label for s in SCENARIOS]
    specs = [("mean_terminal_variables", "Binary variables", "Average sampled terminal size"),
             ("mean_terminal_couplers", "Nonzero couplers", "Average sampled terminal connectivity")]
    chosen = specs if panel is None else [specs[panel]]
    fig, axes = plt.subplots(len(chosen), 1, figsize=(WIDE, 4.8 if panel is None else 3.6), squeeze=False)
    axes = axes.ravel()
    for ax, (metric, ylabel, title) in zip(axes, chosen):
        paired_bars(ax, data, keys, metric, labels, ylabel, title)
    legend(fig, axes)
    fig.tight_layout(rect=(0, .06 if panel is None else .12, 1, 1), pad=.9, h_pad=1.4)
    name = "S02_average_terminal_dimensions" if panel is None else ["S02a_average_terminal_variables", "S02b_average_terminal_couplers"][panel]
    dimension = "Binary-variable and coupler units are common to both encodings." if panel is None else ["Sizes count logical binary variables, not COP satellite variables or physical qubits.", "Connectivity counts nonzero logical quadratic couplers, not physical hardware couplers."][panel]
    save(fig, output, name, f"Arithmetic means across all non-empty sampled terminals, including alternatives discarded during reconstruction. These are workload dimensions, not sizes of the full unresolved problem or exclusively of retained terminals. {dimension}", captions, layouts, role="supplementary")


def supplementary_components(data, output, captions, layouts, *, panel=None):
    labels = [f"{data.loc[(k,'MWIS'),'satellites']}/{data.loc[(k,'MWIS'),'gateways']}" for k in CORE_KEYS]
    specs = [("retained_graph_contacts", "Contacts after global dominance"),
             ("largest_component_contacts", "Largest reduced component")]
    chosen = specs if panel is None else [specs[panel]]
    fig, axes = plt.subplots(1, len(chosen), figsize=(WIDE, 2.9 if panel is None else 3.4), squeeze=False)
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, chosen):
        values = [data.loc[(k, "MWIS"), metric] for k in CORE_KEYS]
        ax.bar(range(5), values, color="#647E91")
        ax.set_xticks(range(5), labels, rotation=25, ha="right")
        ax.set_ylabel("Candidate contacts")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    fig.tight_layout(pad=.9, w_pad=1.8)
    name = "S03_common_reduced_component_structure" if panel is None else ["S03a_contacts_after_global_dominance", "S03b_largest_reduced_component"][panel]
    save(fig, output, name, "One bar per dataset represents both methods: retained weighted vertices and edge sets are independently hash-checked identical after global dominance. Sizes are candidate contacts, not COP satellite variables. Only the five distinct datasets are shown; terminal profiles reuse the same reduced 200/50 graph.", captions, layouts, role="supplementary")


def supplementary_leaves(leaves, output, captions, layouts, *, panel=None):
    specs = [("baseline", "Baseline 200/50"), ("s500_g50", "Satellite case 500/50")]
    chosen = specs if panel is None else [specs[panel]]
    fig, axes = plt.subplots(1, len(chosen), figsize=(WIDE, 3.1 if panel is None else 3.6), squeeze=False)
    axes = axes.ravel()
    for ax, (key, title) in zip(axes, chosen):
        for method in COLORS:
            for kind, ls in [("terminal", "-"), ("incomplete", "--")]:
                values = np.sort(leaves[(leaves.scenario == key) & (leaves.method == method) & (leaves.leaf_kind == kind)].contacts_or_binary_variables)
                if len(values):
                    ax.step(values, np.arange(1, len(values)+1)/len(values), where="post", color=COLORS[method], linestyle=ls, label=f"{method}: {kind}")
        ax.set(xscale="log", ylim=(0, 1.04), xlabel="Remaining contacts / binary variables", title=title)
    axes[0].set_ylabel("Cumulative fraction of leaves")
    legend(fig, axes, ncol=2)
    fig.tight_layout(rect=(0, .19, 1, 1), pad=.9, w_pad=1.8)
    name = "S04_terminal_and_incomplete_leaf_distributions" if panel is None else ["S04a_leaf_distributions_baseline", "S04b_leaf_distributions_s500_g50"][panel]
    scope = "the baseline and largest satellite case" if panel is None else ["the baseline 200/50 case", "the largest satellite case, 500/50"][panel]
    save(fig, output, name, f"Empirical distributions of all recorded terminal and incomplete leaves for {scope}. They describe bounded-tree workloads, not repeated-run uncertainty. Empty fixed-choice leaves are excluded. Means of sampled terminal sizes alone can conceal large unresolved pieces. Current encodings use one bit per retained contact alternative.", captions, layouts, role="supplementary")


def selected_gateways(gateways, scenario, limit):
    rows = gateways[gateways.scenario == scenario.key]
    totals = rows.groupby("station_id").selected_contacts.sum()
    return sorted(totals.index, key=lambda station: (-totals[station], station))[:limit]


def gantt(schedules, gateways, scenario, output, captions, layouts, limit):
    stations = selected_gateways(gateways, scenario, limit)
    fig, axes = plt.subplots(2, 1, figsize=(WIDE, 1.6 + .26 * len(stations)), sharex=True)
    origin_colors = dict(zip(PRIMARY_ORIGINS, ORIGIN_COLORS))
    for ax, method in zip(axes, COLORS):
        rows = schedules[(schedules.scenario == scenario.key) & (schedules.method == method)]
        for i, station in enumerate(stations):
            ordered = rows[rows.station_id == station].sort_values("start_time_hours")
            contacts = list(ordered.itertuples())
            for j, r in enumerate(contacts):
                ax.broken_barh([(r.start_time_hours, r.end_time_hours-r.start_time_hours)], (i-.32,.64), facecolors=origin_colors[r.origin])
                if j + 1 < len(contacts):
                    required = max(r.reconfiguration_time_hours, contacts[j+1].reconfiguration_time_hours)
                    available = contacts[j+1].start_time_hours-r.end_time_hours
                    assert available + 1e-9 >= required
                    ax.broken_barh([(r.end_time_hours, min(required, available))], (i-.32,.64), facecolors="#D8DCE0")
        ax.set_yticks(range(len(stations)), [s.replace("GATEWAY_", "GW ") for s in stations], fontsize=7)
        ax.set_ylim(-.7, len(stations)-.3)
        ax.invert_yaxis()
        ax.set_title(f"{method}: primary schedule, {scenario.satellites}/{scenario.gateways}; {scenario.variables}/{scenario.couplers}")
        ax.grid(axis="y", visible=False)
    axes[-1].set(xlim=(0,HORIZON_HOURS), xlabel="Hours from horizon start (UTC)")
    fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(ORIGIN_COLORS,ORIGIN_LABELS)], loc="lower center",ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.09,1,1),pad=.9,h_pad=1.4)
    save(fig, output, f"schedule_gantt_{scenario.key}_primary_excerpt", f"Primary schedule excerpt before reference solving: the {len(stations)} gateways with highest combined MWIS/COP primary contact counts, ties by station ID. Both panels use the same gateway order and four-hour horizon. Contacts are coloured by primary provenance; grey setup reservations occur only between successive selected contacts. This is an excerpt, not the full schedule. All primary contacts and all gateways are exported in schedule_metrics.csv and gateway_metrics.csv.", captions, layouts, role="operational supplement")


def activity(gateways, scenario, output, captions, layouts, limit):
    stations = selected_gateways(gateways, scenario, limit)
    fig, ax = plt.subplots(figsize=(WIDE, 1.2 + .19 * len(stations)))
    rows = gateways[gateways.scenario == scenario.key].set_index(["station_id","method"])
    y = np.arange(len(stations))
    for offset, method in [(-.18,"MWIS"),(.18,"COP")]:
        ax.barh(y+offset,[rows.loc[(s,method),"selected_contacts"] for s in stations],.34,color=COLORS[method],label=method)
    ax.set_yticks(y,[s.replace("GATEWAY_","GW ") for s in stations],fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Primary scheduled contacts")
    ax.set_title(f"Primary gateway activity: {scenario.satellites}/{scenario.gateways}; {scenario.variables}/{scenario.couplers}")
    ax.grid(axis="y",visible=False)
    ax.legend(loc="lower right",frameon=False)
    fig.tight_layout(pad=.9)
    save(fig, output, f"gateway_activity_{scenario.key}_primary_excerpt", f"Primary contact counts for the {len(stations)} busiest gateways ranked by combined MWIS/COP counts, with deterministic station-ID ties. Remaining and zero-contact gateways are retained in gateway_metrics.csv. Counts concern the simplified single-antenna, one-contact-per-satellite model, not real network efficiency. Contact minutes and service occupancy (excluding setup) are available for every gateway in the exported table.", captions, layouts, role="operational supplement")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_artifacts():
    roots = [RESULTS / "figures/csp_mwis_reordered", RESULTS / "figures/csp_mwis_reordered_dominance"]
    files = [p for base in roots for p in base.rglob("*") if p.is_file()]
    files += [ORIGINAL_SCRIPT, RESULTS / "_config.json", RESULTS / "02.A_mwis_pipeline.ipynb",
              RESULTS / "02.A_mwis_pipeline_dominance_enhanced.ipynb", RESULTS / "02.B_CSP_pipeline.ipynb",
              RESULTS / "3_qubo_annealing.ipynb"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def verify_protected(manifest):
    for relative, expected in manifest.items():
        assert sha(ROOT / relative) == expected, f"Protected artifact changed: {relative}"


def verify_previous_metrics(inputs, record):
    original = inputs.csv(OLD_COMPARISON)
    method = "CSP" if record["method"] == "COP" else "MWIS"
    row = original[(original.scenario == record["scenario"]) & (original.method == method)]
    assert len(row) == 1
    row = row.iloc[0]
    fields = {"primary_objective": "pre_milp_objective", "primary_selected_contacts": "pre_milp_selected_contacts",
              "primary_gap_to_certified_optimum_percent": "pre_milp_gap_to_certified_optimum_percent",
              "reference_upper_bound": "final_upper_bound", "producer_reference_objective": "classical_reference_objective",
              "terminal_qubos": "terminal_qubos", "total_reads": "total_reads", "incomplete_leaves": "incomplete_leaves"}
    for origin in ORIGINS:
        suffix = origin.lower().replace(" ", "_")
        fields[f"primary_contacts_{suffix}"] = f"pre_milp_contacts_{suffix}"
        fields[f"primary_objective_{suffix}"] = f"pre_milp_objective_{suffix}"
    for new, old in fields.items():
        assert abs(float(record[new]) - float(row[old])) < 1e-6, (record["scenario"], method, new)


def export_tables(output, frame, terminals, leaves, schedules, gateways):
    frame.to_csv(output / "comparison_metrics.csv", index=False)
    (output / "comparison_metrics.json").write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")
    terminals.to_csv(output / "terminal_metrics.csv", index=False)
    leaves.to_csv(output / "leaf_metrics.csv", index=False)
    leaf_summary = leaves.groupby(["scenario", "method", "leaf_kind"]).contacts_or_binary_variables.agg(
        count="size", minimum="min", q25=lambda x: x.quantile(.25), median="median", q75=lambda x: x.quantile(.75), maximum="max")
    leaf_summary.to_csv(output / "leaf_size_distribution_summary.csv")
    schedules.to_csv(output / "schedule_metrics.csv", index=False)
    gateways.to_csv(output / "gateway_metrics.csv", index=False)
    nodes = {}
    for (scenario, method), rows in schedules.groupby(["scenario", "method"]):
        nodes.setdefault(scenario, {})[method] = sorted(rows.slot_id.tolist())
    (output / "primary_selected_nodes.json").write_text(json.dumps(nodes, indent=2), encoding="utf-8")
    timing_fields = [c for c in frame if c.startswith("time_") or c.endswith("_seconds") or "timing_complete" in c]
    timing = frame[["scenario", "method", "run", "timing_source", *timing_fields]].copy()
    timing["timing_interpretation"] = "Producer phases are separate; complete primary pipeline timing unavailable. Full workflow includes reference work."
    timing.to_csv(output / "timing_records.csv", index=False)
    reference_fields = [c for c in frame if c.startswith(("reference_", "producer_reference_"))]
    frame[["scenario", "method", "primary_objective", *reference_fields]].to_csv(output / "reference_evaluation_metrics.csv", index=False)
    table = primary_quality_table(frame)
    table.to_csv(output / "primary_quality_summary.csv", index=False)
    return table


def write_readme(output, frame, captions, layouts):
    matches = int(np.isclose(frame.primary_objective, frame.producer_reference_objective, rtol=0, atol=1e-6).sum())
    primary_optimal = int(frame.primary_certified_optimal.sum())
    text = f'''# Primary COP / dominance-enhanced MWIS results

Generated by `results/5_csp_mwis_comparison_plots_primary.py`, a separate copy of
the validated loading/reconstruction logic in `5_csp_mwis_comparison_plots.py`.
Only `mwis-dominance-enhanced` is used for MWIS. COP is read directly from the
final tracked archive `results/runs/csp_final.zip`, without requiring extraction;
the figure legend follows the report's COP terminology. No experiment, solver,
annealing sampler, source notebook or global configuration is executed or edited.

## Endpoint and reference separation

The **primary result** is the selected schedule reconstructed before MILP/LP
reference solving. It includes sampled terminals, exact fixed choices, separator
assignments and greedy incomplete leaves. It contains no MILP-generated contact.
The primary schedules are independently recovered from saved tree metadata and
selected samples, and validated against the original candidate table, satellite
exclusivity, antenna setup and objective totals. Removed dominated alternatives
are not restored. Primary terminal retention is measured before MILP replacement.

All 14 primary records are feasible. {primary_optimal}/14 reach their certified
reference optimum. Primary objectives match the producer's classical tree
reference in {matches}/14 records; that tree reference is not the global optimum
when incomplete leaves remain. The lone mismatch is MWIS at 50/350 (0.205 below
the producer reference). Sampling is classical simulated annealing, not quantum
hardware. Provenance describes retained contacts, not causal advantage or CPU work.

Reference upper bounds are independently checked against the saved feasible,
certified postprocessed schedules. Since these bounds are tight for this campaign,
the primary gaps are retrospective distances to the certified optimum. The saved
MILP evaluation solves incomplete reduced components, not the entire original
instance as a standalone MILP benchmark. Its schedules, improvements and
certificates are separate in `reference_evaluation_metrics.csv`; they are not the
main result and are not inserted into primary Gantt or gateway plots.

## Five report figures and quality table

- `00_primary_quality_table`: seven distinct configurations; baseline counted once.
- `01_primary_quality_scaling`: satellite and gateway quality, two panels.
- `02_primary_schedule_provenance`: paired primary contact shares, one panel.
- `03_terminal_profile_tradeoff`: primary quality and total reads, two panels.
- `04_baseline_logical_dimensions`: baseline scatter, one column.
- `05_producer_stage_profile`: logged producer stage shares, one panel.

For separate experiment subsections, use the full-width standalone panels:
- `01a_primary_quality_satellite_scaling` and `01b_primary_quality_gateway_scaling`.
- `03a_terminal_profile_primary_quality` and `03b_terminal_profile_sampling_workload`.
- `S01a`/`S01b`: retained terminal fraction and incomplete leaves.
- `S02a`/`S02b`: average logical variables and average nonzero couplers.
- `S03a`/`S03b`: contacts after dominance and largest reduced component.
- `S04a`/`S04b`: leaf-size distributions for baseline and 500/50.

These 12 figures redraw the same data on separate 179 mm canvases, with their
own axes, legend (where needed) and captions. The combined figures remain
available; do not include both versions of the same plot in the report.

For the two-column main report, three additional `_column` exports redraw
`01a`, `01b` and `03a` on 86 mm canvases, with readable fonts, compact titles
and shorter axis labels. Use these PDFs with `figure` and `width=\\columnwidth`.
The existing full-width variants remain available. Figure 04 is already
column-sized; figure 02 remains full-width. The gap axis in the compact figures
uses the same certified reference optimum as the full-width versions.
`primary_quality_table.tex` is an editable Overleaf table fragment matching
the image table. It uses `booktabs` and `table*` for a two-column report; use
`table` for a single-column document. Prefer this native table in the report.

Full-width figures use 179 mm canvases; figure 04 and the three `_column`
exports use 86 mm canvases. PDFs preserve
vector lines/text with embedded TrueType fonts; PNGs are 300 DPI. Include them at
their documented width rather than shrinking an old dashboard. `figure_layouts.json`
records exact sizes and panel counts. No figure contains more than two plots.
Captions provide scope and limitations; titles are kept short for printed size.

`S01`-`S04` are supplementary workload, mean dimensions, shared component structure
and two-case leaf-distribution views. Operational plots show ranked gateway
**excerpts** (15 in Gantt, 20 in activity by default), using primary contacts and
the same ranking for both methods. Full schedules and all gateways, including
unused ones, remain in `schedule_metrics.csv` and `gateway_metrics.csv`.

## Timing interpretation

Both final campaigns use 500 reads per terminal. The final CSP archive replaces
the earlier 1,000-read CSP runs and supplies updated timings, including 100/50.
Objectives and selected-contact provenance are independently checked against
the team's updated comparison. Earlier report drafts describing unequal read
budgets need revision. These remain single matched runs on different computers.

Figure 05 uses shares of four disjoint logged **producer** phases, not a total
primary-pipeline runtime. It excludes unprofiled overhead and all later notebook
work. Classical producer evaluation includes exact terminal reference evaluation
and incomplete-leaf evaluation, and is distinguished from sampling.

`timing_records.csv` keeps full recorded workflow time, producer clocks, MILP/LP
solver-call time, and the later notebook remainder separately. Subtracting solver
time alone does not isolate main reconstruction: reference graph preparation,
validation/export and reconstruction share saved timing blocks. Therefore
`isolated_primary_pipeline_seconds` remains unavailable, rather than estimated.
MWIS also has a separately logged sampling/decoding/CSV-export code-cell time;
this is not a pure sampler-call time, and a corresponding COP clock is unavailable.
No missing value is substituted with zero.

Recorded timings are descriptive: hardware/environment provenance was not matched,
although both methods now use 500 reads per terminal. Total reads
include discarded alternative branches. Leaf counts are recorded tree workloads,
not repeated-run uncertainty. Active resource identities are not fully nested in
scaling scenarios; only terminal sensitivity uses the identical input throughout.

## Regeneration and audit

From the repository root in the project Python 3.13.14 environment:

```powershell
python -B -X utf8 results/5_csp_mwis_comparison_plots_primary.py
```

The default folder is `results/figures/csp_mwis_primary`. Options are `--output-dir`,
`--schedule-scenarios baseline` (or several keys, default all), `--gantt-gateways`
and `--activity-gateways`. Only a directory belonging to this primary analysis can
be regenerated. Existing other comparison folders and source run folders are protected.
`--source-git-ref COMMIT` reads the final CSP archive and comparison reference
directly from that commit when working with an older checkout; no extracted CSP
directory is changed. Archive members use `archive.zip!member` manifest keys.

Every source actually used is hash-checked before completion. Source selection,
copied-script provenance, software versions, preservation checks and validation
are in `source_manifest.json`, `preserved_artifacts_manifest.json` and
`validation_report.json`. `comparison_metrics.csv` uses explicit primary/reference
field names. `primary_quality_summary.csv` is the compact report table;
`terminal_metrics.csv` distinguishes primary and postprocessed retention.

## Figure captions

'''
    for name, caption in captions.items():
        text += f"\n### {name}\n\n![{name}]({name}.png)\n\n{caption}\n\n[Vector PDF]({name}.pdf)\n"
    (output / "README.md").write_text(text, encoding="utf-8")
    import html
    def cards(role):
        result = []
        for name, caption in captions.items():
            if layouts[name]["role"] != role: continue
            result.append(f'<article id="{html.escape(name)}"><h2>{html.escape(name)}</h2><a href="{name}.pdf"><img src="{name}.png" alt="{html.escape(name)}"></a><p>{html.escape(caption)}</p></article>')
        return "".join(result)
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Primary COP / MWIS results</title>
<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 24px;color:#25313c}article{padding:20px;margin:24px 0;border:1px solid #cbd6df;border-radius:8px}img{max-width:100%;height:auto}p{line-height:1.55}summary{padding:16px;background:#edf2f6;font-size:20px;cursor:pointer}</style>
<h1>Primary COP / dominance-enhanced MWIS results</h1>
<p>All schedules shown are reconstructed before MILP/LP reference solving. Certified optima are quality references. PDF links preserve report-size vector figures. Recorded producer costs do not isolate a complete primary-pipeline runtime. Read README.md for definitions and limitations.</p>'''
    page += cards("report")
    for role in ["supplementary", "operational supplement"]:
        page += f'<details><summary>{html.escape(role.title())}</summary>{cards(role)}</details>'
    (output / "index.html").write_text(page + "</html>", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-git-ref", help="Read the final CSP archive and its comparison reference from an immutable Git commit; local run folders remain untouched")
    parser.add_argument("--schedule-scenarios", nargs="+", choices=[s.key for s in SCENARIOS] + ["all"], default=["all"])
    parser.add_argument("--gantt-gateways", type=int, default=15)
    parser.add_argument("--activity-gateways", type=int, default=20)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    figures_root = (RESULTS / "figures").resolve()
    if output == figures_root or figures_root not in output.parents:
        parser.error("Choose a new primary analysis directory inside results/figures")
    for old in [figures_root / "csp_mwis_reordered", figures_root / "csp_mwis_reordered_dominance"]:
        if output == old or old in output.parents:
            parser.error("Primary analysis must stay outside the existing comparison folders")
    if output.exists():
        manifest_file = output / "source_manifest.json"
        if not manifest_file.is_file() or json.loads(manifest_file.read_text()).get("analysis_kind") != "primary_before_reference_evaluation":
            parser.error("Refusing to overwrite an existing directory belonging to another analysis")
    if not 1 <= args.gantt_gateways <= 100 or not 1 <= args.activity_gateways <= 100:
        parser.error("Gateway excerpt limits must be between 1 and 100")
    preserved = protected_artifacts()
    inputs = Inputs(args.source_git_ref)
    inputs.read(Path(__file__))
    inputs.read(ORIGINAL_SCRIPT)
    records, terminals, leaves, schedules, gateways, selections = [], [], [], [], [], []
    for scenario in SCENARIOS:
        print(f"Loading primary {scenario.key}", flush=True)
        for method in COLORS:
            record, terminal, leaf, schedule, gateway = load_run(inputs, scenario, method)
            verify_previous_metrics(inputs, record)
            records.append(record); terminals.append(terminal); leaves.append(leaf)
            schedules.append(schedule); gateways.append(gateway)
            selections.append({"scenario": scenario.key, "method": method, "run": record["run"]})
        a, b = records[-2:]
        assert a["input_sha256"] == b["input_sha256"]
        assert a["reduced_slot_graph_sha256"] == b["reduced_slot_graph_sha256"]
        assert abs(a["reference_upper_bound"] - b["reference_upper_bound"]) < 1e-6
    frame = pd.DataFrame(records)
    assert len(frame) == 14 and not frame[["scenario", "method"]].duplicated().any()
    assert frame.primary_feasible.all() and (frame.primary_conflicts == 0).all()
    assert frame.isolated_primary_pipeline_seconds.isna().all()
    assert set(frame.method) == {"MWIS", "COP"}
    assert (frame.loc[frame.method == "MWIS", "mwis_variant"] == "dominance").all()
    terminal_frame = pd.concat(terminals, ignore_index=True)
    leaf_frame = pd.concat(leaves, ignore_index=True)
    schedule_frame = pd.concat(schedules, ignore_index=True)
    gateway_frame = pd.concat(gateways, ignore_index=True)
    assert not (schedule_frame.origin == "MILP").any()
    data = frame.set_index(["scenario", "method"])
    for (scenario, method), part in schedule_frame.groupby(["scenario", "method"]):
        row = data.loc[(scenario, method)]
        assert len(part) == row.primary_selected_contacts
        assert abs(part.weight.sum() - row.primary_objective) < 1e-6
        assert part.slot_id.is_unique and part.satellite_id.is_unique
        t = terminal_frame[(terminal_frame.scenario == scenario) & (terminal_frame.method == method)]
        assert t.retained_in_primary_schedule.sum() == row.primary_retained_terminal_qubos
    output.mkdir(parents=True, exist_ok=True)
    table = export_tables(output, frame, terminal_frame, leaf_frame, schedule_frame, gateway_frame)
    style()
    captions, layouts = {}, {}
    quality_table_figure(table, output, captions, layouts)
    quality_table_latex(table, output)
    for plotter, source in [(quality_scaling, data), (primary_provenance, data), (terminal_profiles, data),
                            (logical_dimensions, terminal_frame), (producer_profile, data),
                            (supplementary_workload, data), (supplementary_means, data),
                            (supplementary_components, data), (supplementary_leaves, leaf_frame)]:
        print(f"Generating {plotter.__name__}", flush=True)
        plotter(source, output, captions, layouts)
    for plotter, source in [(quality_scaling, data), (terminal_profiles, data),
                            (supplementary_workload, data), (supplementary_means, data),
                            (supplementary_components, data), (supplementary_leaves, leaf_frame)]:
        print(f"Generating standalone panels: {plotter.__name__}", flush=True)
        for panel in (0, 1):
            plotter(source, output, captions, layouts, panel=panel)
    print("Generating single-column report quality figures", flush=True)
    for panel in (0, 1):
        quality_scaling(data, output, captions, layouts, panel=panel, column=True)
    terminal_profiles(data, output, captions, layouts, panel=0, column=True)
    chosen = SCENARIOS if "all" in args.schedule_scenarios else [s for s in SCENARIOS if s.key in args.schedule_scenarios]
    for scenario in chosen:
        print(f"Generating primary operational excerpts: {scenario.key}", flush=True)
        gantt(schedule_frame, gateway_frame, scenario, output, captions, layouts, args.gantt_gateways)
        activity(gateway_frame, scenario, output, captions, layouts, args.activity_gateways)
    write_readme(output, frame, captions, layouts)
    inputs.verify_unchanged()
    verify_protected(preserved)
    manifest = {"analysis_kind": "primary_before_reference_evaluation", "mwis_variant": "dominance",
                "selected_runs": selections, "files_sha256": inputs.manifest,
                "csp_source_archive": CSP_ARCHIVE.relative_to(ROOT).as_posix(),
                "source_git_ref": inputs.source_git_ref,
                "script_sha256": sha(Path(__file__)), "copied_from": ORIGINAL_SCRIPT.relative_to(ROOT).as_posix(),
                "copied_from_sha256": sha(ORIGINAL_SCRIPT), "python": sys.version, "platform": platform.platform(),
                "libraries": {"pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__, "networkx": nx.__version__}}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "preserved_artifacts_manifest.json").write_text(json.dumps(preserved, indent=2), encoding="utf-8")
    (output / "figure_captions.json").write_text(json.dumps(captions, indent=2), encoding="utf-8")
    (output / "figure_layouts.json").write_text(json.dumps(layouts, indent=2), encoding="utf-8")
    report = {"passed": True, "matched_configurations": 7, "method_configuration_records": 14,
              "primary_feasible_records": int(frame.primary_feasible.sum()), "primary_certified_optimal_records": int(frame.primary_certified_optimal.sum()),
              "reference_certified_records": int(frame.reference_certified_optimal.sum()),
              "primary_matches_classical_tree_reference": int(np.isclose(frame.primary_objective, frame.producer_reference_objective, rtol=0, atol=1e-6).sum()),
              "primary_selected_contact_records": len(schedule_frame), "sampled_terminal_records": len(terminal_frame),
              "figures_png_and_pdf": len(captions), "maximum_panels_per_figure": max(v["panels"] for v in layouts.values()),
              "standalone_split_figures": 12, "native_latex_quality_table": True,
              "additional_single_column_quality_figures": 3,
              "source_hashes_checked_unchanged": len(inputs.manifest), "protected_artifact_hashes_checked_unchanged": len(preserved),
              "mwis_variant": "dominance", "primary_contains_milp_contacts": False,
              "operational_plot_scenarios": [s.key for s in chosen],
              "checks": ["matching frozen inputs and active resource counts", "identical reduced weighted slot graphs",
                         "read budgets and terminal logical caps", "independent primary reconstruction and original scheduling constraints",
                         "primary objective/contact/weight provenance totals", "primary retained terminal IDs before reference replacement",
                         "certified bounds used only as quality references", "all 14 primary records match previous validated pre-MILP metrics",
                         "primary gateway totals and four-hour service occupancy", "no MILP-generated contact in primary schedules",
                         "missing isolated primary timings left unavailable", "one shared baseline", "at most two panels per figure",
                         "old plotting script, notebooks, configuration and comparison folders unchanged"]}
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Primary figures and tables: {output}")
    print(frame[["scenario", "method", "primary_objective", "primary_selected_contacts", "primary_retained_terminal_qubos", "primary_gap_to_certified_optimum_percent"]].to_string(index=False))


if __name__ == "__main__":
    main()
