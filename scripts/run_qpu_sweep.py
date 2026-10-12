"""Run a producer pipeline (MWIS or CSP) and the QUBO notebook across a grid of QPU-size limits.

Each point gets a temporary config copied from ``results/_config.json``. The
notebooks write their normal artifacts under ``results/runs/<pipeline>`` using a
unique ``qpu_...`` run name; the sweep summary is kept separately in
``results/runs/<pipeline>/qpu_sweep``. Both pipelines read the same frozen
input, so MWIS and CSP sweeps are comparable point by point.
Existing matching outputs are preserved unless ``--overwrite`` is supplied.

Run from the project root, for example:
    python scripts/run_qpu_sweep.py
    python scripts/run_qpu_sweep.py --nodes 10 20 30 50 --couplers 50 100 200
    python scripts/run_qpu_sweep.py --points 20/100 30/200 50/350
    python scripts/run_qpu_sweep.py --nodes 30 --couplers 200 --stages producer
    python scripts/run_qpu_sweep.py --pipeline csp --nodes 30 --couplers 200
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
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

warnings.filterwarnings("ignore", message="Proactor event loop does not implement add_reader")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
BASE_CONFIG_PATH = RESULTS_DIR / "_config.json"
FROZEN_CANDIDATE_SLOT_CSV = Path(
    "data/experiment_inputs/exp_snapshot_2026-10-12/slots_s200_g50.csv"
)
EXACT_CLIQUE_NODE_LIMIT = 50

# Producer pipelines: notebook and the config keys of its run name and input dataset.
PIPELINES = {
    "mwis": {
        "notebook": RESULTS_DIR / "02.A_mwis_pipeline.ipynb",
        "run_name_key": "run_name_mwis",
        "slot_csv_key": "candidate_slot_csv",
    },
    "csp": {
        "notebook": RESULTS_DIR / "02.B_CSP_pipeline.ipynb",
        "run_name_key": "run_name_csp",
        "slot_csv_key": "csp_slot_csv",
    },
}
QUBO_NOTEBOOK = RESULTS_DIR / "3_qubo_annealing.ipynb"
STAGES = ("producer", "qubo")


def sweep_root(pipeline: str) -> Path:
    return RESULTS_DIR / "runs" / pipeline / "qpu_sweep"


def stage_notebook(pipeline: str, stage: str) -> Path:
    return PIPELINES[pipeline]["notebook"] if stage == "producer" else QUBO_NOTEBOOK


# Producer summary label -> (group, key). Unlisted labels are kept in "other".
# The objective and gap labels are shared by both pipelines.
COMMON_METRIC_LAYOUT: dict[str, tuple[str, str] | None] = {
    "subpros decomposed": ("qubo split", "subpros decomposed"),
    "Separator states": ("qubo split", "separator states"),
    "Max decomposition depth": ("qubo split", "max decomposition depth"),
    "Conflict graph edges": ("graph", "num conflicts"),
    "Exact components objective sum": ("objective", "obj from exact split"),
    "Incomplete/heuristic components feasible sum": ("objective", "obj from heur split"),
    "Total feasible objective / lower bound": ("objective", "total objective"),
    "Exact global objective": None,
    "Incomplete/heuristic components conservative upper bound": ("optimality gap", "ub heur subprob"),
    "Total conservative upper bound": ("optimality gap", "ub tot obj"),
    "Absolute optimality-gap bound": ("optimality gap", "absolute optimality-gap bound"),
    "Relative optimality-gap bound (%)": ("optimality gap", "relative optimality-gap bound (%)"),
}

MWIS_METRIC_LAYOUT: dict[str, tuple[str, str] | None] = {
    **COMMON_METRIC_LAYOUT,
    "Number of components (total)": ("graph", "num subgraphs"),
    "Number of couplers (total)": None,     # same value as "Conflict graph edges"
    "Total number of solution subgraphs": ("qubo split", "subproblems"),
    "Subgraphs solved directly": None,
    "Subgraphs solved exactly with separator decomposition": ("qubo split", "subprobs from exact split"),
    "Subgraphs solved heuristically": ("qubo split", "subprobs from heur split"),
}

CSP_METRIC_LAYOUT: dict[str, tuple[str, str] | None] = {
    **COMMON_METRIC_LAYOUT,
    "Connected components": ("graph", "num subgraphs"),
    "Slots removed by dominance": ("graph", "slots removed by dominance"),
    "Terminal sub-problems": ("qubo split", "terminal subprobs"),
    "Incomplete leaves": ("qubo split", "incomplete leaves"),
}

METRIC_LAYOUTS = {"mwis": MWIS_METRIC_LAYOUT, "csp": CSP_METRIC_LAYOUT}

QUBO_METRIC_NAMES: dict[str, str | None] = {
    "Producer": None,
    "Producer run": None,
    "Status": "final status",
    "Selected contacts": "contact slot selected",
    "Total feasible objective": "final objective",
    "Annealed terminal subgraphs available": "subprobs annealed",
}


def resolve_kernel(notebook: Any, notebook_path: Path, override: str | None) -> str:
    """Kernel to run a notebook with: the override, else the notebook's own kernel if it
    is installed, else the default `python3` kernel (the notebooks may be saved with
    different kernels on different machines)."""
    if override:
        return override
    kernel_name = notebook.metadata.get("kernelspec", {}).get("name")
    installed = KernelSpecManager().find_kernel_specs()
    if kernel_name in installed:
        return kernel_name
    if "python3" not in installed:
        raise ValueError(f"Kernel {kernel_name!r} of {notebook_path.name} is not installed; pass --kernel "
                         f"(installed: {', '.join(sorted(installed)) or 'none'})")
    print(f"note: kernel {kernel_name!r} of {notebook_path.name} is not installed; using 'python3'", flush=True)
    return "python3"


def cell_labels(notebook: Any) -> dict[int, str]:
    """Label each code cell with the closest markdown heading above it
    (or its first line of code when there is none)."""
    labels: dict[int, str] = {}
    heading = None
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown":
            headings = [line.lstrip("#").strip() for line in cell.source.splitlines() if line.startswith("#")]
            if headings:
                heading = headings[0]
        elif cell.cell_type == "code" and cell.source.strip():
            first_line = next((line.strip() for line in cell.source.splitlines() if line.strip()), "")
            labels[index] = heading or first_line[:60]
    return labels


def execute_notebook(notebook_path: Path, timeout: int | None, kernel: str | None = None, log_prefix: str = "") -> str:
    """Run a notebook in memory from its own folder without changing the file,
    printing progress per code cell. Returns the name of the kernel used."""
    notebook = nbformat.read(notebook_path, as_version=4)
    kernel_name = resolve_kernel(notebook, notebook_path, kernel)
    labels = cell_labels(notebook)
    positions = {index: position for position, index in enumerate(labels, start=1)}
    cell_start = {}

    def on_cell_start(cell: Any, cell_index: int) -> None:
        if cell_index not in labels:  # also called for markdown and empty cells
            return
        cell_start[cell_index] = time.perf_counter()
        print(f"{log_prefix}  cell {positions[cell_index]}/{len(labels)}: {labels[cell_index]} ...", flush=True)

    def on_cell_executed(cell: Any, cell_index: int, execute_reply: Any) -> None:
        elapsed = time.perf_counter() - cell_start[cell_index]
        failed = execute_reply.get("content", {}).get("status") == "error"
        print(f"{log_prefix}  cell {positions[cell_index]}/{len(labels)} "
              f"{'raised an error' if failed else 'done'} in {elapsed:.1f} s", flush=True)

    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(notebook_path.parent)}},
        on_cell_start=on_cell_start,
        on_cell_executed=on_cell_executed,
    )
    print(f"{log_prefix}  starting kernel {kernel_name!r} ...", flush=True)
    client.execute()
    return kernel_name


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


def collect_metrics(config: dict[str, Any], pipeline: str) -> dict[str, dict[str, Any]]:
    """Read the outputs written by the producer (MWIS or CSP) and reconstruction notebooks."""
    run_name = Path(config[PIPELINES[pipeline]["run_name_key"]])
    producer_dir = RESULTS_DIR / "runs" / pipeline / "pipeline_results" / run_name
    qubo_name = config.get("qubo_output_run_name", run_name.name)
    qubo_dir = RESULTS_DIR / "runs" / pipeline / "qubo_annealing" / qubo_name

    producer = read_metric_table(producer_dir / "final_summary.csv")
    qubo = read_metric_table(qubo_dir / "reconstructed_summary.csv")
    # annealing succeeded when every terminal returned a feasible sample
    if "Infeasible terminal annealing samples" in qubo:
        qubo["anneal success"] = qubo["Infeasible terminal annealing samples"] == 0

    renamed_qubo: dict[str, Any] = {}
    for label, value in qubo.items():
        key = QUBO_METRIC_NAMES.get(label, label)
        if key is not None:
            renamed_qubo[key] = value
    qubo = renamed_qubo

    # the same files and columns are written by both producers
    edges_csv = producer_dir / "conflict_graph_edges.csv"
    if edges_csv.exists():
        producer["Conflict graph edges"] = len(pd.read_csv(edges_csv, usecols=["source"]))

    decomposition_csv = producer_dir / "decomposition_summary.csv"
    if decomposition_csv.exists():
        decomposition = pd.read_csv(decomposition_csv)
        producer["subpros decomposed"] = int(
            decomposition["root_type"].isin(["separator", "independent_sum"]).sum()
        )
        producer["Separator states"] = int(decomposition["separator_states"].sum())
        producer["Max decomposition depth"] = int(decomposition["max_depth"].max())

    qubo_summary_csv = qubo_dir / "terminal_qubo_summary.csv"
    if qubo_summary_csv.exists():
        qubo_summary = pd.read_csv(qubo_summary_csv)
        if not qubo_summary.empty:
            qubo["max subprob var"] = int(qubo_summary["variables"].max())
            qubo["max subprob coup"] = int(qubo_summary["couplers"].max())

    return {"producer": group_metrics(producer, METRIC_LAYOUTS[pipeline]), "qubo": qubo}


def run_point(
    base_config: dict[str, Any],
    pipeline: str,
    nodes: int,
    couplers: int,
    stages: list[str],
    timeout: int | None,
    prefix: str,
    qubo_input_run: str | None = None,
    kernel: str | None = None,
    point_label: str = "",
) -> dict[str, Any]:
    run_name = f"{prefix}_n{nodes}_c{couplers}"
    print(f"\n=== {point_label}{run_name} (nodes={nodes}, couplers={couplers}) "
          f"started at {time.strftime('%H:%M:%S')} ===", flush=True)
    point_dir = sweep_root(pipeline) / run_name
    # A unique flat name keeps both stages paired in the pipeline's run folders.
    stage_run_name = run_name
    point_dir.mkdir(parents=True)

    config = dict(base_config)
    config.update({
        # both producers read the same frozen input
        "candidate_slot_csv": FROZEN_CANDIDATE_SLOT_CSV.as_posix(),
        "csp_slot_csv": FROZEN_CANDIDATE_SLOT_CSV.as_posix(),
        "max_qpu_terminal_nodes": nodes,
        "max_qpu_terminal_couplers": couplers,
        "exact_clique_node_limit": EXACT_CLIQUE_NODE_LIMIT,
        PIPELINES[pipeline]["run_name_key"]: stage_run_name,
        "qubo_input_run": qubo_input_run or f"runs/{pipeline}/pipeline_results/{stage_run_name}",
        "qubo_output_run_name": stage_run_name,
    })
    config_path = point_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Notebook kernels inherit this path and read only this point's config.
    os.environ["SWE2_CONFIG"] = str(config_path)

    row: dict[str, Any] = {
        "run_name": run_name,
        "pipeline": pipeline,
        "candidate_slot_csv": FROZEN_CANDIDATE_SLOT_CSV.as_posix(),
        "max_qpu_terminal_nodes": nodes,
        "max_qpu_terminal_couplers": couplers,
        "exact_clique_node_limit": EXACT_CLIQUE_NODE_LIMIT,
        "status": "ok",
        "runtime_seconds": {},
        "kernels": {},
    }
    for stage in stages:
        notebook_path = stage_notebook(pipeline, stage)
        print(f"[{run_name}] running {notebook_path.name} ...", flush=True)
        start = time.perf_counter()
        try:
            row["kernels"][stage] = execute_notebook(notebook_path, timeout, kernel, log_prefix=f"[{run_name}]")
        except Exception as exc:
            row["status"] = f"failed in {stage}: {type(exc).__name__}"
            (point_dir / f"error_{stage}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[{run_name}] FAILED in {stage}: {type(exc).__name__} (see {point_dir})", flush=True)
            break
        finally:
            row["runtime_seconds"][stage] = round(time.perf_counter() - start, 2)
        print(f"[{run_name}] {stage} done in {row['runtime_seconds'][stage]} s", flush=True)

    print(f"[{run_name}] collecting metrics ...", flush=True)
    row.update(collect_metrics(config, pipeline))
    return row


def prepare_sweep_outputs(pipeline: str, run_names: list[str], summary_path: Path, overwrite: bool) -> None:
    """Refuse to replace existing outputs unless the caller explicitly opts in."""
    targets = [summary_path]
    for run_name in run_names:
        targets.extend((
            sweep_root(pipeline) / run_name,
            RESULTS_DIR / "runs" / pipeline / "pipeline_results" / run_name,
            RESULTS_DIR / "runs" / pipeline / "qubo_annealing" / run_name,
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


DEFAULT_NODES = [10, 20, 30, 50]
DEFAULT_COUPLERS = [50, 100, 200, 400]


def parse_point(text: str) -> tuple[int, int]:
    """Parse one 'nodes/couplers' pair, e.g. '30/200'."""
    try:
        nodes, couplers = (int(part) for part in text.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected NODES/COUPLERS (e.g. 30/200), got {text!r}") from None
    return nodes, couplers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes", type=int, nargs="+", default=None,
                        help=f"values for max_qpu_terminal_nodes, crossed with --couplers (default: {DEFAULT_NODES})")
    parser.add_argument("--couplers", type=int, nargs="+", default=None,
                        help=f"values for max_qpu_terminal_couplers, crossed with --nodes (default: {DEFAULT_COUPLERS})")
    parser.add_argument("--points", type=parse_point, nargs="+", default=None, metavar="NODES/COUPLERS",
                        help="explicit (nodes, couplers) pairs instead of the full grid, e.g. 20/100 30/200 50/350")
    parser.add_argument("--pipeline", choices=list(PIPELINES), default="mwis",
                        help="producer pipeline whose QUBOs are swept (default: mwis)")
    parser.add_argument("--stages", nargs="+", choices=list(STAGES), default=list(STAGES),
                        help="notebooks to run for each point: the producer pipeline and/or the QUBO notebook (default: both)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="per-cell timeout in seconds (default: none)")
    parser.add_argument("--kernel", default=None,
                        help="Jupyter kernel for every notebook (default: each notebook's own kernel, "
                             "or 'python3' when that one is not installed)")
    parser.add_argument("--qubo-input-run", default=None,
                        help="run QUBO stage only from this existing path relative to results/")
    parser.add_argument("--prefix", default="qpu", help="run-name prefix")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing outputs for these sweep points and summary")
    args = parser.parse_args()

    if not args.prefix or not all(char.isascii() and (char.isalnum() or char in "_-") for char in args.prefix):
        parser.error("--prefix may contain only ASCII letters, digits, underscores, and hyphens")
    if args.points is not None:
        if args.nodes is not None or args.couplers is not None:
            parser.error("use either --points or --nodes/--couplers, not both")
        points = args.points
    else:
        nodes_values = args.nodes or DEFAULT_NODES
        couplers_values = args.couplers or DEFAULT_COUPLERS
        if len(set(nodes_values)) != len(nodes_values) or len(set(couplers_values)) != len(couplers_values):
            parser.error("node and coupler values must not contain duplicates")
        points = list(itertools.product(nodes_values, couplers_values))
    if any(value <= 0 for point in points for value in point):
        parser.error("node and coupler limits must be positive integers")
    if len(set(points)) != len(points):
        parser.error("--points must not contain duplicates")
    if args.qubo_input_run and args.stages != ["qubo"]:
        parser.error("--qubo-input-run can only be used with --stages qubo")
    if args.qubo_input_run and not args.qubo_input_run.replace("\\", "/").startswith(f"runs/{args.pipeline}/"):
        parser.error(f"--qubo-input-run must be a run of the {args.pipeline} pipeline (runs/{args.pipeline}/...); "
                     "use --pipeline to select the other one")

    base_config = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    candidate_slot_csv = PROJECT_ROOT / FROZEN_CANDIDATE_SLOT_CSV
    if not candidate_slot_csv.is_file():
        raise FileNotFoundError(f"Frozen QPU-sweep input not found: {candidate_slot_csv}")

    summary_path = sweep_root(args.pipeline) / f"{args.prefix}_sweep_summary.json"
    run_names = [f"{args.prefix}_n{nodes}_c{couplers}" for nodes, couplers in points]
    prepare_sweep_outputs(args.pipeline, run_names, summary_path, args.overwrite)
    sweep_root(args.pipeline).mkdir(parents=True, exist_ok=True)

    print(f"QPU sweep: pipeline={args.pipeline}, stages={', '.join(args.stages)}, {len(points)} point(s)", flush=True)
    sweep_start = time.perf_counter()
    rows = []
    for position, (nodes, couplers) in enumerate(points, start=1):
        rows.append(run_point(
            base_config, args.pipeline, nodes, couplers, args.stages, args.timeout, args.prefix,
            qubo_input_run=args.qubo_input_run, kernel=args.kernel, point_label=f"point {position}/{len(points)}: ",
        ))
        # Save after each point so completed results survive an interrupted sweep.
        summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        elapsed = time.perf_counter() - sweep_start
        remaining = elapsed / position * (len(points) - position)
        print(f"point {position}/{len(points)} finished: {rows[-1]['status']} "
              f"(sweep elapsed {elapsed / 60:.1f} min, ~{remaining / 60:.1f} min left)", flush=True)

    print(f"\nSweep summary written to {summary_path}")
    for row in rows:
        timings = ", ".join(f"{stage} {seconds} s" for stage, seconds in row["runtime_seconds"].items())
        print(f"  {row['run_name']}: {row['status']} ({timings})")


if __name__ == "__main__":
    main()
