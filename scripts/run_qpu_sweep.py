"""Run the MWIS and QUBO notebooks across a grid of QPU-size limits.

Each point gets a temporary config copied from ``results/_config.json``. The
notebooks write their normal artifacts under ``results/runs/mwis`` using a
unique ``qpu_...`` run name; the sweep summary is kept separately in
``results/runs/mwis/qpu_sweep``.
Existing matching outputs are preserved unless ``--overwrite`` is supplied.

Run from the project root, for example:
    python scripts/run_qpu_sweep.py
    python scripts/run_qpu_sweep.py --nodes 10 20 30 50 --couplers 50 100 200
    python scripts/run_qpu_sweep.py --nodes 30 --couplers 200 --stages mwis
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import nbformat
import pandas as pd
from nbclient import NotebookClient

warnings.filterwarnings("ignore", message="Proactor event loop does not implement add_reader")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
BASE_CONFIG_PATH = RESULTS_DIR / "_config.json"
SWEEP_ROOT = RESULTS_DIR / "runs" / "mwis" / "qpu_sweep"
FROZEN_CANDIDATE_SLOT_CSV = Path(
    "data/experiment_inputs/exp1_3_snapshot_2026-09-27/scenario_inputs/slots_s500_g50.csv"
)
EXACT_CLIQUE_NODE_LIMIT = 50

STAGES = {
    "mwis": RESULTS_DIR / "02.A_mwis_pipeline.ipynb",
    "qubo": RESULTS_DIR / "3_qubo_annealing_reconstruction.ipynb",
}

# MWIS summary label -> (group, key). Unlisted labels are kept in "other".
MWIS_METRIC_LAYOUT: dict[str, tuple[str, str] | None] = {
    "Number of components (total)": ("graph", "num subgraphs"),
    "Number of couplers (total)": ("graph", "num conflicts"),
    "Total number of solution subgraphs": ("qubo split", "subproblems"),
    "Subgraphs solved directly": None,
    "Subgraphs solved exactly with separator decomposition": ("qubo split", "subprobs from exact split"),
    "Subgraphs solved heuristically": ("qubo split", "subprobs from heur split"),
    "subpros decomposed": ("qubo split", "subpros decomposed"),
    "Separator states": ("qubo split", "separator states"),
    "Max decomposition depth": ("qubo split", "max decomposition depth"),
    "Exact components objective sum": ("objective", "obj from exact split"),
    "Incomplete/heuristic components feasible sum": ("objective", "obj from heur split"),
    "Total feasible objective / lower bound": ("objective", "total objective"),
    "Exact global objective": None,
    "Incomplete/heuristic components conservative upper bound": ("optimality gap", "ub heur subprob"),
    "Total conservative upper bound": ("optimality gap", "ub tot obj"),
    "Absolute optimality-gap bound": ("optimality gap", "absolute optimality-gap bound"),
    "Relative optimality-gap bound (%)": ("optimality gap", "relative optimality-gap bound (%)"),
}

QUBO_METRIC_NAMES: dict[str, str | None] = {
    "Producer run": None,
    "Status": "anneal success",
    "Selected contacts": "contact slot selected",
    "Total feasible objective": "final mwis objective",
    "Annealed terminal subgraphs available": "subprobs annealed",
}


def execute_notebook(notebook_path: Path, timeout: int | None) -> None:
    """Run a notebook in memory from its own folder without changing the file."""
    notebook = nbformat.read(notebook_path, as_version=4)
    kernel_name = notebook.metadata.get("kernelspec", {}).get("name")
    if not kernel_name:
        raise ValueError(f"Notebook has no kernel name in its metadata: {notebook_path}")
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()


def parse_value(value: Any) -> Any:
    """Convert summary values to JSON-friendly numbers or strings."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return int(number) if number.is_integer() else number


def read_metric_table(csv_path: Path) -> dict[str, Any]:
    """Read a (metric, value, ...) summary CSV into a dictionary."""
    if not csv_path.exists():
        return {}
    table = pd.read_csv(csv_path, dtype=str)
    return {str(row.metric): parse_value(row.value) for row in table.itertuples(index=False)}


def group_metrics(
    metrics: dict[str, Any], layout: dict[str, tuple[str, str] | None]
) -> dict[str, dict[str, Any]]:
    """Group summary labels for compact sweep output; preserve unknown labels."""
    grouped: dict[str, dict[str, Any]] = {}
    for target in layout.values():
        if target is not None:
            grouped.setdefault(target[0], {})
    for label, value in metrics.items():
        target = layout.get(label, ("other", label))
        if target is not None:
            grouped.setdefault(target[0], {})[target[1]] = value
    return {group: values for group, values in grouped.items() if values}


def collect_metrics(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Read the outputs written by the current MWIS and reconstruction notebooks."""
    run_name = Path(config["run_name_mwis"])
    mwis_dir = RESULTS_DIR / "runs" / "mwis" / "pipeline_results" / run_name
    qubo_dir = RESULTS_DIR / "runs" / "mwis" / "qubo_annealing" / run_name

    mwis = read_metric_table(mwis_dir / "final_summary.csv")
    qubo = read_metric_table(qubo_dir / "reconstructed_summary.csv")
    if "Status" in qubo:
        qubo["Status"] = qubo["Status"] != "annealed_infeasible"

    renamed_qubo: dict[str, Any] = {}
    for label, value in qubo.items():
        key = QUBO_METRIC_NAMES.get(label, label)
        if key is not None:
            renamed_qubo[key] = value
    qubo = renamed_qubo

    decomposition_csv = mwis_dir / "decomposition_summary.csv"
    if decomposition_csv.exists():
        decomposition = pd.read_csv(decomposition_csv)
        mwis["subpros decomposed"] = int(
            decomposition["root_type"].isin(["separator", "independent_sum"]).sum()
        )
        mwis["Separator states"] = int(decomposition["separator_states"].sum())
        mwis["Max decomposition depth"] = int(decomposition["max_depth"].max())

    qubo_summary_csv = qubo_dir / "terminal_qubo_summary.csv"
    if qubo_summary_csv.exists():
        qubo_summary = pd.read_csv(qubo_summary_csv)
        qubo["max subprob var"] = int(qubo_summary["variables"].max())
        qubo["max subprob coup"] = int(qubo_summary["couplers"].max())

    return {"mwis": group_metrics(mwis, MWIS_METRIC_LAYOUT), "qubo": qubo}


def run_point(
    base_config: dict[str, Any],
    nodes: int,
    couplers: int,
    stages: list[str],
    timeout: int | None,
    prefix: str,
) -> dict[str, Any]:
    run_name = f"{prefix}_n{nodes}_c{couplers}"
    point_dir = SWEEP_ROOT / run_name
    # A unique flat name keeps both stages paired in the current MWIS run folders.
    stage_run_name = run_name
    point_dir.mkdir(parents=True)

    config = dict(base_config)
    config.update({
        "candidate_slot_csv": FROZEN_CANDIDATE_SLOT_CSV.as_posix(),
        "max_qpu_terminal_nodes": nodes,
        "max_qpu_terminal_couplers": couplers,
        "exact_clique_node_limit": EXACT_CLIQUE_NODE_LIMIT,
        "run_name_mwis": stage_run_name,
        "qubo_input_run": f"runs/mwis/pipeline_results/{stage_run_name}",
    })
    config_path = point_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Notebook kernels inherit this path and read only this point's config.
    os.environ["SWE2_CONFIG"] = str(config_path)

    row: dict[str, Any] = {
        "run_name": run_name,
        "candidate_slot_csv": FROZEN_CANDIDATE_SLOT_CSV.as_posix(),
        "max_qpu_terminal_nodes": nodes,
        "max_qpu_terminal_couplers": couplers,
        "exact_clique_node_limit": EXACT_CLIQUE_NODE_LIMIT,
        "status": "ok",
        "runtime_seconds": {},
    }
    for stage in stages:
        notebook_path = STAGES[stage]
        print(f"[{run_name}] running {notebook_path.name} ...", flush=True)
        start = time.perf_counter()
        try:
            execute_notebook(notebook_path, timeout)
        except Exception as exc:
            row["status"] = f"failed in {stage}: {type(exc).__name__}"
            (point_dir / f"error_{stage}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[{run_name}] FAILED in {stage}: {type(exc).__name__} (see {point_dir})", flush=True)
            break
        finally:
            row["runtime_seconds"][stage] = round(time.perf_counter() - start, 2)
        print(f"[{run_name}] {stage} done in {row['runtime_seconds'][stage]} s", flush=True)

    row.update(collect_metrics(config))
    return row


def prepare_sweep_outputs(run_names: list[str], summary_path: Path, overwrite: bool) -> None:
    """Refuse to replace existing outputs unless the caller explicitly opts in."""
    targets = [summary_path]
    for run_name in run_names:
        targets.extend((
            SWEEP_ROOT / run_name,
            RESULTS_DIR / "runs" / "mwis" / "pipeline_results" / run_name,
            RESULTS_DIR / "runs" / "mwis" / "qubo_annealing" / run_name,
        ))

    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        examples = "\n".join(f"  {path}" for path in existing[:5])
        more = "\n  ..." if len(existing) > 5 else ""
        raise FileExistsError(
            "Sweep output already exists; choose a new --prefix or pass --overwrite "
            "to replace only this sweep's matching outputs:\n" + examples + more
        )

    for path in existing:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes", type=int, nargs="+", default=[10, 20, 30, 50],
                        help="values for max_qpu_terminal_nodes")
    parser.add_argument("--couplers", type=int, nargs="+", default=[50, 100, 200, 400],
                        help="values for max_qpu_terminal_couplers")
    parser.add_argument("--stages", nargs="+", choices=list(STAGES), default=list(STAGES),
                        help="notebooks to run for each point (default: both)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="per-cell timeout in seconds (default: none)")
    parser.add_argument("--prefix", default="qpu", help="run-name prefix")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing outputs for these sweep points and summary")
    args = parser.parse_args()

    if not args.prefix or not all(char.isascii() and (char.isalnum() or char in "_-") for char in args.prefix):
        parser.error("--prefix may contain only ASCII letters, digits, underscores, and hyphens")
    if any(value <= 0 for value in args.nodes + args.couplers):
        parser.error("node and coupler limits must be positive integers")
    if len(set(args.nodes)) != len(args.nodes) or len(set(args.couplers)) != len(args.couplers):
        parser.error("node and coupler values must not contain duplicates")

    base_config = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    candidate_slot_csv = PROJECT_ROOT / FROZEN_CANDIDATE_SLOT_CSV
    if not candidate_slot_csv.is_file():
        raise FileNotFoundError(f"Frozen QPU-sweep input not found: {candidate_slot_csv}")

    summary_path = SWEEP_ROOT / f"{args.prefix}_sweep_summary.json"
    run_names = [f"{args.prefix}_n{nodes}_c{couplers}"
                 for nodes, couplers in itertools.product(args.nodes, args.couplers)]
    prepare_sweep_outputs(run_names, summary_path, args.overwrite)
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []
    for nodes, couplers in itertools.product(args.nodes, args.couplers):
        rows.append(run_point(base_config, nodes, couplers, args.stages, args.timeout, args.prefix))
        # Save after each point so completed results survive an interrupted sweep.
        summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"\nSweep summary written to {summary_path}")
    for row in rows:
        timings = ", ".join(f"{stage} {seconds} s" for stage, seconds in row["runtime_seconds"].items())
        print(f"  {row['run_name']}: {row['status']} ({timings})")


if __name__ == "__main__":
    main()
