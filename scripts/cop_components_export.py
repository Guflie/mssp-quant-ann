"""Store the large connected components of a COP instance, with their exact decomposition.

Steps, as in results/02.B_CSP_pipeline.ipynb (sections 3-5), using only src/:
    1. build the COP of the slot CSV (one variable per satellite, its domain the satellite's slots)
    2. dominance_reduction (exact), then split_connected_components (largest first, in slots)
    3. keep the components with at least --min-qubo-vars QUBO variables (one per slot left after
       dominance, the same in both encodings): the "relevant" ones
    4. csp_to_qubo on each kept component, whole (before any decomposition), in every encoding of
       --encodings, with the default penalties of src/CSP_to_QUBO.py
    5. decompose_csp on each kept component (skip with --skip-decomposition); budgets and QPU limits
       default to the values of results/_config.json but are CLI flags, so the config is never read

Component ids are numbered over all the components of the instance (component_001 is the largest),
so they match results/02.B_CSP_pipeline.ipynb and cop_qubo_components.py.

Self-contained: besides src/, it depends on no other script (build_csp and component_row are
copies of the ones in scripts/visualization/cop_conflict_graph.py and cop_qubo_components.py).

Output, in --out-dir (default scripts/cop_components_export/):
    <stem>_components.csv      one row per kept component: sizes (as in cop_qubo_components.py), QUBO
                               couplers per encoding and, when decomposed, status, tree nodes,
                               terminals, incomplete leaves
    <stem>_components.pkl      fast cache: {"metadata", "slot_weight", "components": {id: CSP},
                               "qubos": {id: {encoding: {"Q": {(i, j): coeff}, "index_map":
                               {slot_id: (bit, next_bit)}}}}, "decompositions": {id: decompose_csp
                               output}}. A slot is chosen in a sample when sample[bit] - sample[next_bit]
                               is 1 (next_bit None: just sample[bit]), see decode_sample in
                               src/CSP_to_QUBO.py. All components are dumped together, so they keep
                               sharing one ConflictIndex. Loading needs
                               the src/ classes unchanged: use load_components() from this module.
                               The metadata (source CSV, git commit, settings) say which instance
                               and code version produced it: rerun the script if it no longer loads

Example:
    .venv/Scripts/python.exe scripts/cop_components_export.py "data/exp_datasets/slots_g50_strat_seed65416843/starlink_scheduling_slots_s1000_g50_ deg60_seed65416843.csv"
"""

from __future__ import annotations

import argparse
import inspect
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[0]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from CSP import CSP, CSP_Variable, Timeslot  # noqa: E402
from CSP_decomposition import decompose_csp, dominance_reduction, split_connected_components  # noqa: E402
from CSP_to_QUBO import csp_to_qubo  # noqa: E402

DEFAULT_OUT_DIR = HERE / Path(__file__).stem  # scripts/<script name>
COLUMNS = ["slot_id", "satellite_name", "antenna_id", "start_time_hours", "end_time_hours",
           "reconfiguration_time_hours", "weight"]
ENCODINGS = ("domain_wall", "one_hot")

# defaults of the decomposition settings: the values of results/_config.json
DECOMPOSITION_DEFAULTS = {
    "max_qubits": 30,                # max_qpu_terminal_nodes
    "max_couplers": 200,             # max_qpu_terminal_couplers
    "max_separator_size": 3,
    "max_candidate_pool": 14,        # max_separator_candidate_pool
    "max_separator_states": 10_000,
    "max_recursion_depth": 12,
    "max_decomposition_nodes": 10_000,
}


# ---------------------------------------------------------------- COP model and component sizes
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


def conflict_pair_count(component: CSP) -> int:
    """Conflicting slot pairs between different satellites of the component (each pair once)."""
    return sum(len(component.conflicting_values(v, t)) for v in component.vars for t in v.domain) // 2


def component_qubos(component: CSP, weight: dict[str, float], encodings: tuple[str, ...]) -> dict[str, dict]:
    """QUBO of the whole component in each encoding (csp_to_qubo, default penalties):
    {encoding: {"Q": {(i, j): coeff}, "index_map": {slot_id: (bit, next_bit)}}}."""
    qubos = {}
    for encoding in encodings:
        Q, index_map = csp_to_qubo(component, weight, encoding=encoding)
        qubos[encoding] = {"Q": Q, "index_map": index_map}
    return qubos


def component_row(component_id: str, component: CSP, qubos: dict[str, dict]) -> dict:
    """Size of one component: COP variables, constraints, domains, QUBO variables and couplers."""
    domains = [len(v.domain) for v in component.vars]
    row = {
        "component_id": component_id,
        "satellites": len(component.vars),
        "constraints": component.num_constraints(),
        "conflict_pairs": conflict_pair_count(component),
        "avg_domain": sum(domains) / len(domains),
        "max_domain": max(domains),
        "qubo_variables": sum(domains),
    }
    for encoding, qubo in qubos.items():
        assert len(qubo["index_map"]) == row["qubo_variables"]
        row[f"couplers_{encoding}"] = sum(1 for i, j in qubo["Q"] if i != j)
    return row


# ---------------------------------------------------------------- helpers for the reading scripts
def load_components(pkl_path: Path) -> dict[str, Any]:
    """Load a <stem>_components.pkl (src/ is already on sys.path through this module)."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- export
def git_state() -> dict[str, Any]:
    """Commit of the repository and whether src/ has uncommitted changes (None if git fails)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip() != ""
        return {"git_commit": commit, "src_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "src_dirty": None}


def decomposition_row(decomposition: dict[str, Any], seconds: float) -> dict[str, Any]:
    terminals = [n for n in decomposition["terminals"] if n["qubits"] > 0]
    return {
        "decomposition_status": decomposition["status"],
        "root_type": decomposition["root"]["node_type"],
        "tree_nodes": len(decomposition["nodes"]),
        "terminal_subproblems": len(terminals),
        "largest_terminal_qubits": max((n["qubits"] for n in terminals), default=0),
        "incomplete_nodes": len(decomposition["incomplete"]),
        "largest_incomplete_qubits": max((n["qubits"] for n in decomposition["incomplete"]), default=0),
        "separator_states": decomposition["separator_state_count"],
        "cache_hits": decomposition["cache_hits"],
        "max_depth": decomposition["max_depth"],
        "decomposition_seconds": round(seconds, 2),
    }


def run(slots_path: Path, min_qubo_vars: int, decompose: bool, settings: dict[str, Any],
        encodings: tuple[str, ...], out_dir: Path) -> pd.DataFrame:
    stem = slots_path.stem
    print(f"\n{stem}", flush=True)
    started = time.perf_counter()
    slots = pd.read_csv(slots_path, usecols=COLUMNS)
    weight = dict(zip(slots["slot_id"], slots["weight"].astype(float)))

    # 1-2. COP, dominance, connected components (largest first, in slots)
    csp = build_csp(slots)
    input_slots = sum(len(v.domain) for v in csp.vars)
    dominated = dominance_reduction(csp, weight)
    components = split_connected_components(csp)
    print(f"  {len(csp.vars)} satellites, {input_slots} slots; dominance removed {dominated} slots; "
          f"{len(components)} components ({time.perf_counter() - started:.1f} s)", flush=True)

    # 3. relevant components: at least min_qubo_vars QUBO variables (one per slot)
    kept = {f"component_{i:03d}": c for i, c in enumerate(components, start=1)
            if sum(len(v.domain) for v in c.vars) >= min_qubo_vars}
    if not kept:
        largest = sum(len(v.domain) for v in components[0].vars) if components else 0
        print(f"  no component with >= {min_qubo_vars} QUBO variables (largest: {largest}); nothing saved")
        return pd.DataFrame()
    print(f"  kept {len(kept)} components with >= {min_qubo_vars} QUBO variables", flush=True)

    # 4. QUBO of each kept component, its sizes and its exact decomposition
    rows, qubos, decompositions = [], {}, {}
    for index, (component_id, component) in enumerate(kept.items(), start=1):
        qubo_started = time.perf_counter()
        qubos[component_id] = component_qubos(component, weight, encodings)
        row = component_row(component_id, component, qubos[component_id])
        row["qubo_seconds"] = round(time.perf_counter() - qubo_started, 2)
        couplers = ", ".join(f"{row[f'couplers_{e}']} couplers ({e})" for e in encodings)
        print(f"  [{index}/{len(kept)}] {component_id}: {row['satellites']} satellites, "
              f"{row['qubo_variables']} QUBO variables, {couplers}", end="", flush=True)
        if decompose:
            decomposition_started = time.perf_counter()
            decompositions[component_id] = decompose_csp(component, weight, component_id,
                                                         encoding=settings["encoding"],
                                                         **settings["budgets"])
            row.update(decomposition_row(decompositions[component_id],
                                         time.perf_counter() - decomposition_started))
            print(f" -> {row['decomposition_status']}, {row['tree_nodes']} nodes, "
                  f"{row['terminal_subproblems']} terminals, {row['incomplete_nodes']} incomplete "
                  f"({row['decomposition_seconds']:.1f} s)", end="")
        print(flush=True)
        rows.append(row)
    table = pd.DataFrame(rows)

    metadata = {
        "source_csv": slots_path.resolve().relative_to(PROJECT_ROOT).as_posix()
        if slots_path.resolve().is_relative_to(PROJECT_ROOT) else str(slots_path.resolve()),
        "created": datetime.now().isoformat(timespec="seconds"),
        **git_state(),
        "satellites": len(csp.vars),
        "input_slots": input_slots,
        "dominated_slots": dominated,
        "components_total": len(components),
        "min_qubo_vars": min_qubo_vars,
        "qubo_encodings": list(encodings),
        # csp_to_qubo is called with its defaults: record them (penalty_chain None = per satellite)
        "qubo_parameters": {name: p.default for name, p in inspect.signature(csp_to_qubo).parameters.items()
                            if p.default is not inspect.Parameter.empty and name not in ("slot_weight", "encoding")},
        "decomposed": decompose,
        "decomposition_settings": settings if decompose else None,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"{stem}_components.csv", index=False)

    kept_slots = {t.slot_id for c in kept.values() for v in c.vars for t in v.domain}
    with open(out_dir / f"{stem}_components.pkl", "wb") as f:
        pickle.dump({
            "metadata": metadata,
            "slot_weight": {s: weight[s] for s in sorted(kept_slots)},
            "components": kept,                 # one dump: the components keep sharing one ConflictIndex
            "qubos": qubos,
            "decompositions": decompositions,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"  saved {stem}_components.csv and .pkl to {out_dir} "
          f"({time.perf_counter() - started:.1f} s)")
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("slots", type=Path, help="path to the slot CSV")
    parser.add_argument("--min-qubo-vars", type=int, default=2000,
                        help="smallest component kept, in QUBO variables (slots after dominance); default 2000")
    parser.add_argument("--skip-decomposition", action="store_true",
                        help="only dominance and connected components, no decompose_csp")
    parser.add_argument("--encoding", choices=ENCODINGS, default="domain_wall",
                        help="encoding the decomposition sizes the terminals with (csp_qubo_encoding)")
    for name, default in DECOMPOSITION_DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=default)
    parser.add_argument("--encodings", nargs="+", choices=ENCODINGS, default=list(ENCODINGS),
                        help="encodings of the component QUBOs built and stored (default: both)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="default: scripts/cop_components_export/")
    args = parser.parse_args(argv)
    if not args.slots.is_file():
        parser.error(f"slot CSV not found: {args.slots}")
    settings = {"encoding": args.encoding, "budgets": {name: getattr(args, name) for name in DECOMPOSITION_DEFAULTS}}
    run(args.slots, args.min_qubo_vars, not args.skip_decomposition, settings, tuple(args.encodings), args.out_dir)


if __name__ == "__main__":
    main()
