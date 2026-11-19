"""Run the large COP components saved by cop_components_export.py on a D-Wave QPU, keeping everything.

Input: a <stem>_components.pkl from scripts/cop_components_export.py (the components and their QUBOs,
the same csp_to_qubo models used by the plot scripts and the CSP pipeline).

For each selected component (all of them by default):
    1. minor embedding (minorminer) on the solver's working graph, chain strength from
       uniform_torque_compensation (as in results/3_qubo_annealing.ipynb), the coefficient scale the
       QPU's auto-scaling will apply, and the estimated QPU access time
    2. only with --submit: the embedded problem is sent to the QPU (answer_mode raw, so every read is
       kept in order, solver warnings saved). Kept from the D-Wave API:
         - the sampleset as returned (physical samples, one column per qubit, energies, occurrences)
           and its info (timing breakdown, problem id, label, warnings)
         - the server's record of the problem (SAPI problems/<id>/info): the submitted data, every
           parameter the solver applied, defaults included (thermalizations, flux drift
           compensation, per-qubit flux biases and anneal offsets, auto_scale...), status,
           submitted / solved timestamps, solver graph id, server messages and the raw answer
         - client-side submit / resolve wall-clock times
       and derived from it: the logical samples (majority vote on broken chains), per-chain break
       rates, and a steepest-descent polish of every QPU read

Every read (QPU and polished QPU) is decoded into slots and scored: objective (raw slot weights),
feasibility (antenna conflicts, two slots of one satellite), broken domain-wall chains (or one-hot
violations), and the objective after a greedy repair + fill.

Without --submit nothing is sent to the QPU (the solver is contacted only for its working graph and
the time estimate, which uses no QPU access time): run it first to check the embedding and estimate.
The run stops before submitting anything if the estimated total exceeds --budget-seconds.

Self-contained: besides src/ it imports only the D-Wave Ocean SDK, numpy and pandas.

Output, in --out-dir/<pickle stem>/<run name>/ (default scripts/qpu_component_run/...; the run name
defaults to <timestamp>_qpu or <timestamp>_dryrun, so QPU data is never overwritten):
    run_config.json         arguments, source pickle metadata, solver, git commit
    solver_properties.json  the solver's full property dict (ranges, timing model, working graph)
    summary.csv             one row per component: sizes, embedding, chain strength, scale, estimated
                            and charged QPU time, timing breakdown, server status / timestamps /
                            applied scalar parameters, and for every source the best / repaired
                            objective, feasible and broken-encoding fractions
    reads.csv               one row per read and source (qpu, qpu_polished)
    chains.csv              one row per logical variable: satellite, slot, qubits, chain break rate
    sapi/<component>.json   the server's full record of the problem (problems/<id>/info), as returned
    qpu_run.pkl             everything above plus the samplesets (physical, logical, polished),
                            embeddings, embedded BQMs and the sampleset info; saved after every component

QPU settings: the defaults of --solver, --num-reads, --annealing-time, --chain-strength-prefactor and
--budget-seconds are read from --config (default results/_config.json, or the SWE2_CONFIG environment
variable, as in results/3_qubo_annealing.ipynb): qpu_solver, num_reads, qpu_annealing_time_us,
qpu_chain_strength_prefactor, qpu_time_budget_seconds. A flag given on the command line wins over
the config; a key missing from the config falls back to the value in CONFIG_FALLBACKS.

Examples:
    .venv/Scripts/python.exe scripts/qpu_component_run.py "scripts/cop_components_export/<stem>_components.pkl"
    .venv/Scripts/python.exe scripts/qpu_component_run.py "<pkl>" --components component_001 --num-reads 1000 --submit
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dimod
import minorminer
import networkx as nx
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[0]
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # the pickle holds src/ classes (CSP, Timeslot)

DEFAULT_OUT_DIR = HERE / Path(__file__).stem  # scripts/<script name>
DEFAULT_CONFIG = Path(os.environ.get("SWE2_CONFIG", PROJECT_ROOT / "results" / "_config.json"))
# argparse destination -> (config key, value used when the key is missing from the config)
CONFIG_FALLBACKS = {
    "solver": ("qpu_solver", "Advantage_system6"),
    "num_reads": ("num_reads", 100),
    "annealing_time": ("qpu_annealing_time_us", 20.0),
    "chain_strength_prefactor": ("qpu_chain_strength_prefactor", 1.414),
    "budget_seconds": ("qpu_time_budget_seconds", 5.0),
}
# solver parameters that change the QPU access time estimate
ESTIMATE_PARAMS = ("readout_thermalization", "programming_thermalization", "reduce_intersample_correlation")


# ---------------------------------------------------------------- loading
def load_export(pkl_path: Path) -> dict[str, Any]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip() != ""
        return {"git_commit": commit, "src_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "src_dirty": None}


# ---------------------------------------------------------------- problem structure
class Problem:
    """Everything needed to decode and score a sample of one component's QUBO, as numpy arrays."""

    def __init__(self, component, qubo: dict[str, Any], slot_weight: dict[str, float], encoding: str):
        self.encoding = encoding
        self.Q = qubo["Q"]
        self.index_map = qubo["index_map"]
        self.owner = {t.slot_id: v.name for v in component.vars for t in v.domain}
        self.slots = sorted(self.index_map)
        self.slot_pos = {s: k for k, s in enumerate(self.slots)}
        self.weight = np.array([float(slot_weight[s]) for s in self.slots])
        self.satellites = sorted({v.name for v in component.vars})
        sat_pos = {name: k for k, name in enumerate(self.satellites)}
        self.slot_sat = np.array([sat_pos[self.owner[s]] for s in self.slots])

        # QUBO variables (bits), all of them, also those a cancelled coefficient left out of Q
        self.bits = sorted({b for b, n in self.index_map.values()} | {n for b, n in self.index_map.values()
                                                                       if n is not None})
        self.bit_pos = {b: k for k, b in enumerate(self.bits)}
        self.slot_bit = np.array([self.bit_pos[self.index_map[s][0]] for s in self.slots])
        self.slot_next = np.array([self.bit_pos[n] if (n := self.index_map[s][1]) is not None else -1
                                   for s in self.slots])
        self.bit_slot = {self.index_map[s][0]: s for s in self.slots}   # every bit leads one slot

        # conflicts between slots of different satellites (each pair once)
        pairs = set()
        for v in component.vars:
            for t in v.domain:
                for c in component.conflicting_values(v, t):
                    a, b = sorted((t.slot_id, c.slot_id))
                    pairs.add((self.slot_pos[a], self.slot_pos[b]))
        self.conflicts = np.array(sorted(pairs), dtype=int).reshape(-1, 2)
        self.neighbours: dict[int, set[int]] = {k: set() for k in range(len(self.slots))}
        for a, b in self.conflicts:
            self.neighbours[a].add(b)
            self.neighbours[b].add(a)

        # bits of each satellite, in chain order (domain wall: consecutive pairs must not read 0 1)
        sat_bits: dict[int, list[int]] = {}
        for k, s in enumerate(self.slots):
            sat_bits.setdefault(int(self.slot_sat[k]), []).append(int(self.slot_bit[k]))
        self.sat_bits = {sat: sorted(bits) for sat, bits in sat_bits.items()}
        chain_pairs = [(p, c) for bits in self.sat_bits.values() for p, c in zip(bits, bits[1:])]
        self.chain_pairs = np.array(chain_pairs, dtype=int).reshape(-1, 2)

        self.bqm = dimod.BinaryQuadraticModel.from_qubo(self.Q)
        self.bqm.add_variables_from((b, 0.0) for b in self.bits if b not in self.bqm.variables)

    def matrix(self, sampleset: dimod.SampleSet) -> np.ndarray:
        """Reads x bits (columns in self.bits order)."""
        order = [sampleset.variables.index(b) for b in self.bits]
        return np.asarray(sampleset.record.sample[:, order], dtype=np.int8)

    def evaluate(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Decode and score every read of X (reads x bits)."""
        nxt = np.where(self.slot_next >= 0, X[:, np.maximum(self.slot_next, 0)], 0)
        chosen = (X[:, self.slot_bit] - nxt) == 1                       # reads x slots
        counts = np.zeros((X.shape[0], len(self.satellites)), dtype=int)
        np.add.at(counts, (slice(None), self.slot_sat), chosen.astype(int))
        same_satellite = np.maximum(counts - 1, 0).sum(axis=1)
        conflict = (chosen[:, self.conflicts[:, 0]] & chosen[:, self.conflicts[:, 1]]).sum(axis=1) \
            if len(self.conflicts) else np.zeros(X.shape[0], dtype=int)
        if self.encoding == "domain_wall":
            encoding_violations = ((X[:, self.chain_pairs[:, 0]] == 0) & (X[:, self.chain_pairs[:, 1]] == 1)).sum(axis=1) \
                if len(self.chain_pairs) else np.zeros(X.shape[0], dtype=int)
        else:   # one-hot: satellites with more than one bit set
            set_bits = np.zeros_like(counts)
            np.add.at(set_bits, (slice(None), self.slot_sat), X[:, self.slot_bit].astype(int))
            encoding_violations = (set_bits > 1).sum(axis=1)
        return {
            "chosen": chosen,
            "objective": chosen.astype(float) @ self.weight,
            "conflict_violations": conflict,
            "same_satellite_violations": same_satellite,
            "encoding_violations": encoding_violations,
            "feasible": (conflict == 0) & (same_satellite == 0),
            "satellites_scheduled": (counts > 0).sum(axis=1),
        }

    def repair(self, chosen: np.ndarray) -> float:
        """Objective after a greedy repair (drop the slot with most clashes, lightest first) and fill
        (add the heaviest free slots that clash with nothing)."""
        selected = set(np.flatnonzero(chosen).tolist())
        by_sat: dict[int, set[int]] = {}
        for k in selected:
            by_sat.setdefault(int(self.slot_sat[k]), set()).add(k)

        def clashes(k: int) -> int:
            return len(self.neighbours[k] & selected) + len(by_sat[int(self.slot_sat[k])]) - 1

        while True:
            worst = max(selected, key=lambda k: (clashes(k), -self.weight[k], -k), default=None)
            if worst is None or clashes(worst) == 0:
                break
            selected.discard(worst)
            by_sat[int(self.slot_sat[worst])].discard(worst)
        for k in np.argsort(-self.weight, kind="stable"):
            k = int(k)
            if not by_sat.get(int(self.slot_sat[k])) and not (self.neighbours[k] & selected):
                selected.add(k)
                by_sat.setdefault(int(self.slot_sat[k]), set()).add(k)
        return float(self.weight[sorted(selected)].sum()) if selected else 0.0


# ---------------------------------------------------------------- embedding and scaling
def embed(bqm: dimod.BinaryQuadraticModel, T: nx.Graph, seed: int, tries: int, timeout: float) -> dict:
    """minorminer embedding of the QUBO graph into the working graph; isolated variables get a free qubit."""
    from dwave.embedding import is_valid_embedding
    S = nx.Graph()
    S.add_nodes_from(sorted(bqm.variables))
    S.add_edges_from(sorted(tuple(sorted(e)) for e in bqm.quadratic))
    chains = minorminer.find_embedding(sorted(S.edges), T, random_seed=seed, tries=tries, timeout=timeout)
    if S.number_of_edges() and not chains:
        raise RuntimeError("minorminer found no embedding (raise --timeout / --tries)")
    chains = {b: [int(q) for q in qs] for b, qs in chains.items()}
    used = {q for qs in chains.values() for q in qs}
    free = (q for q in sorted(T) if q not in used)
    for b in S:
        if b not in chains:
            chains[b] = [next(free)]
    if not is_valid_embedding(chains, S, T):
        raise RuntimeError("invalid embedding")
    return chains


def autoscale(embedded: dimod.BinaryQuadraticModel, properties: dict[str, Any], chain_strength: float) -> dict[str, float]:
    """Factor the QPU's auto-scaling divides the Ising problem by, and the scaled extremes."""
    h, J, _ = embedded.to_ising()
    h_lo, h_hi = properties["h_range"]
    j_lo, j_hi = properties["j_range"]

    def ratio(value: float, lo: float, hi: float) -> float:
        return value / hi if value > 0 else value / lo if value < 0 else 0.0

    factor = max([ratio(v, h_lo, h_hi) for v in h.values()] + [ratio(v, j_lo, j_hi) for v in J.values()] + [1e-12])
    return {
        "autoscale_factor": factor,
        "scaled_max_abs_h": max((abs(v) for v in h.values()), default=0.0) / factor,
        "scaled_max_abs_j": max((abs(v) for v in J.values()), default=0.0) / factor,
        "scaled_min_abs_j": min((abs(v) for v in J.values() if abs(v) > 1e-12), default=0.0) / factor,
        "scaled_chain_coupling": -chain_strength / factor,   # Ising coupling of every chain edge on the chip
    }


def chain_break_rates(physical: dimod.SampleSet, chains: dict) -> dict[int, float]:
    """Fraction of reads (weighted by occurrences) in which each chain is broken."""
    col = {q: k for k, q in enumerate(physical.variables)}
    X = physical.record.sample
    weights = physical.record.num_occurrences
    rates = {}
    for b, qubits in chains.items():
        if len(qubits) == 1:
            rates[b] = 0.0
            continue
        values = X[:, [col[q] for q in qubits]]
        broken = values.min(axis=1) != values.max(axis=1)
        rates[b] = float(np.average(broken, weights=weights))
    return rates


# ---------------------------------------------------------------- scoring
def score_reads(problem: Problem, sampleset: dimod.SampleSet, source: str,
                component_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """reads.csv rows of one sampleset, and the per-source summary columns."""
    X = problem.matrix(sampleset)
    scores = problem.evaluate(X)
    occurrences = sampleset.record.num_occurrences
    energies = sampleset.record.energy
    breaks = sampleset.record.chain_break_fraction if "chain_break_fraction" in sampleset.record.dtype.names else None
    repaired_cache: dict[bytes, float] = {}
    rows = []
    for r in range(X.shape[0]):
        key = scores["chosen"][r].tobytes()
        if key not in repaired_cache:
            repaired_cache[key] = problem.repair(scores["chosen"][r])
        objective = float(scores["objective"][r])
        rows.append({
            "component_id": component_id,
            "source": source,
            "read": r,
            "energy": float(energies[r]),
            "num_occurrences": int(occurrences[r]),
            "chain_break_fraction": None if breaks is None else float(breaks[r]),
            "objective": objective,
            "feasible": bool(scores["feasible"][r]),
            "satellites_scheduled": int(scores["satellites_scheduled"][r]),
            "conflict_violations": int(scores["conflict_violations"][r]),
            "same_satellite_violations": int(scores["same_satellite_violations"][r]),
            "encoding_violations": int(scores["encoding_violations"][r]),
            "repaired_objective": repaired_cache[key],
        })
    reads = pd.DataFrame(rows)
    w = reads["num_occurrences"].to_numpy()
    feasible = reads["feasible"].to_numpy()
    best_feasible = reads.loc[feasible, "objective"].max() if feasible.any() else None
    best_feasible_energy = reads.loc[feasible, "energy"].min() if feasible.any() else None
    summary = {
        f"{source}_reads": int(w.sum()),
        f"{source}_distinct_samples": int(len({scores['chosen'][r].tobytes() for r in range(X.shape[0])})),
        f"{source}_min_energy": float(reads["energy"].min()),
        f"{source}_mean_energy": float(np.average(reads["energy"], weights=w)),
        f"{source}_min_feasible_energy": None if best_feasible_energy is None else float(best_feasible_energy),
        f"{source}_feasible_fraction": float(np.average(feasible, weights=w)),
        f"{source}_broken_encoding_fraction": float(np.average(reads["encoding_violations"] > 0, weights=w)),
        f"{source}_best_feasible_objective": None if best_feasible is None else float(best_feasible),
        f"{source}_best_repaired_objective": float(reads["repaired_objective"].max()),
        f"{source}_mean_repaired_objective": float(np.average(reads["repaired_objective"], weights=w)),
    }
    if breaks is not None:
        summary[f"{source}_chain_break_fraction_mean"] = float(np.average(breaks, weights=w))
        summary[f"{source}_chain_break_fraction_max"] = float(breaks.max())
    return rows, summary


def fetch_problem_info(problems_api, problem_id: str) -> dict[str, Any]:
    """The server's full record of a problem (GET problems/<id>/info), as raw JSON.

    Problems.get_problem_info would do the same request, but its pydantic model requires
    answer.num_occurrences, which the server leaves out for answer_mode="raw" (every read listed
    once), so it rejects a valid record. The JSON is kept as the server sent it."""
    response = problems_api.session.get(
        f"{problem_id}/info", headers={"Accept": "application/vnd.dwave.sapi.problem-data+json; version=3.0.0"})
    return json.loads(response.content)


def server_summary(record: dict[str, Any]) -> dict[str, Any]:
    """summary.csv columns from the server's record: status, timestamps, applied parameters
    (scalars as they are, per-qubit lists as the number of non-zero entries)."""
    metadata = record.get("metadata") or {}
    row = {
        "sapi_status": metadata.get("status"),
        "sapi_submitted_on": metadata.get("submitted_on"),
        "sapi_solved_on": metadata.get("solved_on"),
        "sapi_solver_graph_id": ((metadata.get("solver") or {}).get("version") or {}).get("graph_id"),
        "sapi_messages": json.dumps(metadata.get("messages") or []),
    }
    for key, value in (record.get("params") or {}).items():
        if isinstance(value, list):
            row[f"param_{key}_nonzero"] = sum(1 for v in value if v not in (0, 0.0, None))
        else:
            row[f"param_{key}"] = value
    return row


# ---------------------------------------------------------------- run
def parse_params(items: list[str]) -> dict[str, Any]:
    params = {}
    for item in items:
        key, _, value = item.partition("=")
        try:
            params[key] = json.loads(value)
        except json.JSONDecodeError:
            params[key] = value
    return params


def save(run_dir: Path, state: dict[str, Any]) -> None:
    """Write every output (called after each component, so finished QPU jobs are never lost)."""
    pd.DataFrame(state["summary_rows"]).to_csv(run_dir / "summary.csv", index=False)
    pd.DataFrame(state["read_rows"]).to_csv(run_dir / "reads.csv", index=False)
    pd.DataFrame(state["chain_rows"]).to_csv(run_dir / "chains.csv", index=False)
    with open(run_dir / "qpu_run.pkl", "wb") as f:
        pickle.dump({"metadata": state["metadata"], "solver_properties": state["solver_properties"],
                     "components": state["components"]}, f, protocol=pickle.HIGHEST_PROTOCOL)


def run(args: argparse.Namespace) -> Path:
    from dwave.embedding import embed_bqm, unembed_sampleset
    from dwave.embedding.chain_breaks import majority_vote
    from dwave.embedding.chain_strength import uniform_torque_compensation
    from dwave.cloud.api import Problems
    from dwave.samplers import SteepestDescentSolver
    from dwave.system import DWaveSampler
    from dwave.system.warnings import WarningAction

    data = load_export(args.pkl)
    components = args.components or sorted(data["components"])
    missing = [c for c in components if c not in data["components"]]
    if missing:
        raise SystemExit(f"components not in the pickle: {missing} (available: {sorted(data['components'])})")
    for cid in components:
        if args.encoding not in data["qubos"][cid]:
            raise SystemExit(f"{cid} has no {args.encoding} QUBO (rerun cop_components_export.py with that encoding)")
    extra = parse_params(args.qpu_param)

    qpu = DWaveSampler(solver=args.solver) if args.solver else DWaveSampler()
    properties = dict(qpu.properties)
    T = nx.Graph()
    T.add_nodes_from(qpu.nodelist)
    T.add_edges_from(qpu.edgelist)
    print(f"solver {qpu.solver.name}: {properties['topology']['type']} {properties['topology']['shape']}, "
          f"{len(qpu.nodelist)} working qubits, {len(qpu.edgelist)} couplers", flush=True)

    mode = "qpu" if args.submit else "dryrun"
    run_name = args.run_name or f"{datetime.now():%Y%m%d-%H%M%S}_{mode}"
    run_dir = args.out_dir / args.pkl.stem.removesuffix("_components") / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "source_pickle": str(args.pkl),
        "export_metadata": data["metadata"],
        "components": components,
        "encoding": args.encoding,
        "solver": qpu.solver.name,
        "topology": properties["topology"],
        "num_reads": args.num_reads,
        "annealing_time_us": args.annealing_time,
        "chain_strength_prefactor": args.chain_strength_prefactor,
        "qpu_params": extra,
        "budget_seconds": args.budget_seconds,
        "seed": args.seed,
        "embedding_tries": args.tries,
        "embedding_timeout": args.timeout,
        "config_file": str(args.config),
        "settings_from_config": args.settings_from_config,
        **git_state(),
    }
    (run_dir / "run_config.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    (run_dir / "solver_properties.json").write_text(json.dumps(properties, default=str), encoding="utf-8")
    state = {"metadata": metadata, "solver_properties": properties, "components": {},
             "summary_rows": [], "read_rows": [], "chain_rows": []}

    # 1) everything classical first: embedding, scaling, estimate
    plans = []
    estimate_kwargs = {k: extra[k] for k in ESTIMATE_PARAMS if k in extra}
    for index, cid in enumerate(components):
        component = data["components"][cid]
        problem = Problem(component, data["qubos"][cid][args.encoding], data["slot_weight"], args.encoding)
        row = {"component_id": cid, "satellites": len(problem.satellites), "slots": len(problem.slots),
               "qubo_variables": len(problem.bits), "qubo_couplers": problem.bqm.num_interactions,
               "conflict_pairs": len(problem.conflicts)}
        print(f"{cid}: {row['satellites']} satellites, {row['qubo_variables']} QUBO variables, "
              f"{row['qubo_couplers']} couplers", flush=True)

        record = {"problem": {"slots": problem.slots, "bits": problem.bits, "satellite_of_slot": problem.owner}}

        started = time.perf_counter()
        try:
            chains = embed(problem.bqm, T, args.seed + index, args.tries, args.timeout)
        except RuntimeError as exc:
            print(f"  embedding failed: {exc}", flush=True)
            row["embedding_status"] = str(exc)
            state["summary_rows"].append(row)
            state["components"][cid] = record
            continue
        lengths = [len(qs) for qs in chains.values()]
        chain_strength = float(uniform_torque_compensation(problem.bqm, chains, args.chain_strength_prefactor))
        # linear biases spread over the chain in spin space, as FixedEmbeddingComposite does (the notebook)
        embedded = embed_bqm(problem.bqm, chains, qpu.adjacency, chain_strength=chain_strength,
                             smear_vartype=dimod.SPIN)
        physical_qubits = sum(lengths)
        estimate_us = float(qpu.solver.estimate_qpu_access_time(
            physical_qubits, num_reads=args.num_reads, annealing_time=args.annealing_time, **estimate_kwargs))
        row.update({
            "embedding_status": "ok",
            "embedding_seconds": time.perf_counter() - started,
            "physical_qubits": physical_qubits,
            "physical_couplers": embedded.num_interactions,
            "chain_length_max": max(lengths),
            "chain_length_mean": float(np.mean(lengths)),
            "chain_strength": chain_strength,
            **autoscale(embedded, properties, chain_strength),
            "estimated_qpu_access_time_us": estimate_us,
        })
        print(f"  embedding: {physical_qubits} qubits, chain length max {max(lengths)} mean {np.mean(lengths):.2f}, "
              f"chain strength {chain_strength:.3f}, auto-scale /{row['autoscale_factor']:.2f}, "
              f"estimated QPU access {estimate_us / 1e3:.1f} ms", flush=True)
        record.update({"embedding": chains, "chain_strength": chain_strength, "embedded_bqm": embedded})

        state["components"][cid] = record
        plans.append({"cid": cid, "problem": problem, "row": row, "record": record, "embedded": embedded,
                      "chains": chains, "estimate_us": estimate_us})
        state["summary_rows"].append(row)
        save(run_dir, state)

    estimated_total = sum(p["estimate_us"] for p in plans) / 1e6
    print(f"estimated QPU access time for {len(plans)} components: {estimated_total:.3f} s "
          f"(budget {args.budget_seconds} s)", flush=True)
    if not args.submit:
        print(f"dry run: nothing sent to the QPU. Outputs in {run_dir}", flush=True)
        return run_dir
    if estimated_total > args.budget_seconds:
        raise SystemExit(f"estimated {estimated_total:.3f} s exceeds --budget-seconds {args.budget_seconds}; "
                         "nothing was submitted (raise the budget or lower --num-reads)")

    # 2) QPU, one component at a time
    problems_api = Problems.from_config()   # same credentials as DWaveSampler
    sapi_dir = run_dir / "sapi"
    sapi_dir.mkdir()
    spent_us = 0.0
    for plan in plans:
        cid, problem, row, record = plan["cid"], plan["problem"], plan["row"], plan["record"]
        if (spent_us + plan["estimate_us"]) / 1e6 > args.budget_seconds:
            print(f"budget reached ({spent_us / 1e6:.3f} s used); {cid} and later components not submitted", flush=True)
            break
        label = f"{run_dir.parent.name}/{cid}/{run_name}"[-500:]
        submitted_at = datetime.now(timezone.utc)
        physical = qpu.sample(plan["embedded"], num_reads=args.num_reads, annealing_time=args.annealing_time,
                              answer_mode="raw", label=label, warnings=WarningAction.SAVE, **extra)
        physical.resolve()
        resolved_at = datetime.now(timezone.utc)
        info = dict(physical.info)
        timing = info.get("timing", {})
        spent_us += float(timing.get("qpu_access_time", 0.0))
        record.update({"physical_sampleset": physical, "info": info})
        save(run_dir, state)   # the QPU answer is on disk before anything else can fail

        # the server's own record: submitted data, applied parameters (defaults included), timestamps
        try:
            sapi_record = fetch_problem_info(problems_api, info["problem_id"])
            (sapi_dir / f"{cid}.json").write_text(json.dumps(sapi_record), encoding="utf-8")
            row.update(server_summary(sapi_record))
        except Exception as exc:   # never lose the run because of the follow-up query
            sapi_record = {"error": repr(exc)}
            row["sapi_status"] = f"query failed: {exc!r}"
        record["sapi_problem"] = sapi_record

        logical = unembed_sampleset(physical, plan["chains"], problem.bqm, chain_break_method=majority_vote,
                                    chain_break_fraction=True)
        polished = SteepestDescentSolver().sample(problem.bqm, initial_states=logical)

        warnings_saved = info.get("warnings") or []
        row.update({
            "problem_id": info.get("problem_id"),
            "problem_label": info.get("problem_label"),
            "client_submitted_at": submitted_at.isoformat(),
            "client_resolved_at": resolved_at.isoformat(),
            "client_wall_seconds": (resolved_at - submitted_at).total_seconds(),
            "warning_count": len(warnings_saved),
            "warning_types": json.dumps(sorted({str(w.get("type")) for w in warnings_saved if isinstance(w, dict)})),
            **{f"timing_{k}": v for k, v in timing.items()},
        })
        for name, sampleset in (("qpu", logical), ("qpu_polished", polished)):
            rows, summary = score_reads(problem, sampleset, name, cid)
            state["read_rows"] += rows
            row.update(summary)

        rates = chain_break_rates(physical, plan["chains"])
        for b, qubits in sorted(plan["chains"].items()):
            slot = problem.bit_slot.get(b)
            state["chain_rows"].append({
                "component_id": cid, "bit": b, "slot_id": slot,
                "satellite": problem.owner.get(slot) if slot else None,
                "chain_length": len(qubits), "qubits": json.dumps(qubits), "break_rate": rates[b],
            })
        record.update({"physical_sampleset": physical, "logical_sampleset": logical,
                       "polished_sampleset": polished, "info": info})
        save(run_dir, state)
        print(f"{cid}: QPU access {timing.get('qpu_access_time', 0) / 1e3:.1f} ms, best feasible "
              f"{row['qpu_best_feasible_objective']}, feasible fraction "
              f"{row['qpu_feasible_fraction']:.2f}, chain breaks {row['qpu_chain_break_fraction_mean']:.4f}, "
              f"polished best {row['qpu_polished_best_feasible_objective']}, warnings {row['warning_count']}",
              flush=True)

    metadata["qpu_access_time_us"] = spent_us
    (run_dir / "run_config.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    save(run_dir, state)
    print(f"QPU access time used: {spent_us / 1e6:.3f} s (estimated {estimated_total:.3f} s). Outputs in {run_dir}")
    return run_dir


def config_defaults(config_path: Path) -> tuple[dict[str, Any], list[str]]:
    """QPU defaults from the config file (fallbacks for missing keys), and the keys actually read."""
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    defaults, found = {}, []
    for dest, (key, fallback) in CONFIG_FALLBACKS.items():
        if key in config:
            defaults[dest] = config[key]
            found.append(key)
        else:
            defaults[dest] = fallback
    if defaults["solver"] is None:   # null in the config: D-Wave's default solver
        defaults["solver"] = ""
    return defaults, found


def main(argv: list[str] | None = None) -> None:
    # the config path is needed before the other defaults can be set
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    config_path = pre.parse_known_args(argv)[0].config
    defaults, found = config_defaults(config_path)
    from_config = f"default from {config_path.name}"

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pkl", type=Path, help="<stem>_components.pkl from scripts/cop_components_export.py")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="JSON config with the QPU defaults (default results/_config.json or $SWE2_CONFIG)")
    parser.add_argument("--components", nargs="+", help="component ids (default: all in the pickle)")
    parser.add_argument("--encoding", default="domain_wall", choices=("domain_wall", "one_hot"))
    parser.add_argument("--submit", action="store_true", help="send the problems to the QPU (uses QPU access time)")
    parser.add_argument("--solver", help=f"solver name, '' for D-Wave's default ({from_config}: qpu_solver)")
    parser.add_argument("--num-reads", type=int, help=f"QPU reads ({from_config}: num_reads)")
    parser.add_argument("--annealing-time", type=float, help=f"microseconds ({from_config}: qpu_annealing_time_us)")
    parser.add_argument("--chain-strength-prefactor", type=float,
                        help=f"({from_config}: qpu_chain_strength_prefactor)")
    parser.add_argument("--qpu-param", action="append", default=[], metavar="KEY=VALUE",
                        help="extra solver parameter, value parsed as JSON (repeatable), "
                             "e.g. readout_thermalization=100")
    parser.add_argument("--budget-seconds", type=float,
                        help=f"cap on the total QPU access time ({from_config}: qpu_time_budget_seconds)")
    parser.add_argument("--seed", type=int, default=0, help="embedding seed of the first component (+1 each)")
    parser.add_argument("--tries", type=int, default=10, help="minorminer restarts")
    parser.add_argument("--timeout", type=float, default=120.0, help="minorminer time limit per component, seconds")
    parser.add_argument("--run-name", help="output folder name (default <timestamp>_qpu / _dryrun)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.pkl.suffix != ".pkl":
        parser.error(f"expected the .pkl written by cop_components_export.py, got {args.pkl.name}")
    # which settings came from the config (a command-line flag overrides it)
    given = {a.split("=")[0] for a in (argv if argv is not None else sys.argv[1:]) if a.startswith("--")}
    args.settings_from_config = {
        CONFIG_FALLBACKS[dest][0]: getattr(args, dest) for dest in CONFIG_FALLBACKS
        if CONFIG_FALLBACKS[dest][0] in found and f"--{dest.replace('_', '-')}" not in given
    }
    print(f"settings from {config_path}: {args.settings_from_config or 'none'}", flush=True)
    run(args)


if __name__ == "__main__":
    main()
