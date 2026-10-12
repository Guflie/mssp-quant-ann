"""Run a producer pipeline (MWIS or CSP) and the QUBO notebook once per dataset,
varying the number of satellites (same ground stations).

The datasets are listed by name in ``DATASETS`` below: edit that table to choose
what to run. Every other setting (QPU limits, decomposition budgets, annealing
reads, MILP budgets) comes from ``results/_config.json``, so all datasets are
run with the same parameters.

Each dataset gets a temporary config copied from ``results/_config.json``. The
notebooks write their normal artifacts under ``results/runs/<pipeline>`` using
the run name ``<prefix>_<dataset name>``; the sweep summary and the per-dataset
configs are kept separately in ``results/runs/<pipeline>/dataset_sat_sweep``.
Existing matching outputs are preserved unless ``--overwrite`` is supplied.

Run from the project root, for example:
    python scripts/run_sat_dataset_sweep.py --pipeline csp
    python scripts/run_sat_dataset_sweep.py --pipeline csp --datasets s100_g50 s200_g50
    python scripts/run_sat_dataset_sweep.py --pipeline csp --stages qubo --overwrite
    python scripts/run_sat_dataset_sweep.py --pipeline mwis --prefix ds_mwis
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

# notebook execution and metric collection are shared with the QPU-size sweep
from run_qpu_sweep import (
    BASE_CONFIG_PATH,
    PIPELINES,
    PROJECT_ROOT,
    RESULTS_DIR,
    STAGES,
    collect_metrics,
    execute_notebook,
    stage_notebook,
)

# Datasets to run: name -> candidate-slot CSV, relative to the project root.
# The name becomes part of the run folders (<prefix>_<name>), so use only
# letters, digits, underscores and hyphens.
DATASETS: dict[str, str] = {
    "s100_g50": "data/experiment_inputs/exp_snapshot_2026-10-12/slots_s100_g50.csv",
    "s200_g50": "data/experiment_inputs/exp_snapshot_2026-10-12/slots_s200_g50.csv",
    "s500_g50": "data/experiment_inputs/exp_snapshot_2026-10-12/slots_s500_g50.csv",
}


def sweep_root(pipeline: str) -> Path:
    return RESULTS_DIR / "runs" / pipeline / "dataset_sat_sweep"


def producer_dir(pipeline: str, run_name: str) -> Path:
    return RESULTS_DIR / "runs" / pipeline / "pipeline_results" / run_name


def qubo_dir(pipeline: str, run_name: str) -> Path:
    return RESULTS_DIR / "runs" / pipeline / "qubo_annealing" / run_name


def is_valid_name(text: str) -> bool:
    return bool(text) and all(char.isascii() and (char.isalnum() or char in "_-") for char in text)


def run_dataset(
    base_config: dict[str, Any],
    pipeline: str,
    dataset: str,
    slot_csv: str,
    stages: list[str],
    timeout: int | None,
    prefix: str,
    kernel: str | None = None,
    point_label: str = "",
) -> dict[str, Any]:
    run_name = f"{prefix}_{dataset}"
    print(f"\n=== {point_label}{run_name} ({slot_csv}) started at {time.strftime('%H:%M:%S')} ===", flush=True)
    point_dir = sweep_root(pipeline) / run_name
    point_dir.mkdir(parents=True, exist_ok=True)

    config = dict(base_config)
    config.update({
        PIPELINES[pipeline]["slot_csv_key"]: slot_csv,
        PIPELINES[pipeline]["run_name_key"]: run_name,
        # the QUBO stage reads this dataset's producer run and writes under the same name
        "qubo_input_run": f"runs/{pipeline}/pipeline_results/{run_name}",
        "qubo_output_run_name": run_name,
    })
    config_path = point_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Notebook kernels inherit this path and read only this dataset's config.
    os.environ["SWE2_CONFIG"] = str(config_path)

    row: dict[str, Any] = {
        "run_name": run_name,
        "dataset": dataset,
        "pipeline": pipeline,
        "slot_csv": slot_csv,
        "max_qpu_terminal_nodes": config.get("max_qpu_terminal_nodes"),
        "max_qpu_terminal_couplers": config.get("max_qpu_terminal_couplers"),
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


def prepare_sweep_outputs(
    pipeline: str, run_names: list[str], stages: list[str], summary_path: Path, overwrite: bool
) -> None:
    """Refuse to replace existing outputs unless the caller explicitly opts in.
    Only the outputs of the stages being run are replaced, so a QUBO-only rerun
    keeps the producer results it reads."""
    targets = [summary_path]
    for run_name in run_names:
        targets.append(sweep_root(pipeline) / run_name)
        if "producer" in stages:
            targets.append(producer_dir(pipeline, run_name))
        if "qubo" in stages:
            targets.append(qubo_dir(pipeline, run_name))

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
    parser.add_argument("--datasets", nargs="+", default=None, metavar="NAME",
                        help=f"datasets to run, by name (default: all of {', '.join(DATASETS)})")
    parser.add_argument("--pipeline", choices=list(PIPELINES), default="csp",
                        help="producer pipeline to run on every dataset (default: csp)")
    parser.add_argument("--stages", nargs="+", choices=list(STAGES), default=list(STAGES),
                        help="notebooks to run for each dataset: the producer pipeline and/or the QUBO notebook (default: both)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="per-cell timeout in seconds (default: none)")
    parser.add_argument("--kernel", default=None,
                        help="Jupyter kernel for every notebook (default: each notebook's own kernel, "
                             "or 'python3' when that one is not installed)")
    parser.add_argument("--prefix", default="ds", help="run-name prefix")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing outputs for these datasets and summary")
    args = parser.parse_args()

    if not is_valid_name(args.prefix):
        parser.error("--prefix may contain only ASCII letters, digits, underscores, and hyphens")
    invalid = [name for name in DATASETS if not is_valid_name(name)]
    if invalid:
        parser.error(f"dataset names may contain only ASCII letters, digits, underscores, and hyphens: {invalid}")
    datasets = args.datasets or list(DATASETS)
    unknown = [name for name in datasets if name not in DATASETS]
    if unknown:
        parser.error(f"unknown dataset(s) {unknown}; edit DATASETS in {Path(__file__).name} "
                     f"(available: {', '.join(DATASETS)})")
    if len(set(datasets)) != len(datasets):
        parser.error("--datasets must not contain duplicates")

    missing = [DATASETS[name] for name in datasets if not (PROJECT_ROOT / DATASETS[name]).is_file()]
    if missing:
        raise FileNotFoundError("Dataset CSV(s) not found:\n" + "\n".join(f"  {path}" for path in missing))

    run_names = [f"{args.prefix}_{name}" for name in datasets]
    # a QUBO-only run reads the producer outputs of a previous run with the same names
    if "producer" not in args.stages:
        absent = [name for name in run_names if not producer_dir(args.pipeline, name).is_dir()]
        if absent:
            raise FileNotFoundError("--stages qubo needs the producer outputs of a previous run:\n"
                                    + "\n".join(f"  {producer_dir(args.pipeline, name)}" for name in absent))

    base_config = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    summary_path = sweep_root(args.pipeline) / f"{args.prefix}_sweep_summary.json"
    prepare_sweep_outputs(args.pipeline, run_names, args.stages, summary_path, args.overwrite)
    sweep_root(args.pipeline).mkdir(parents=True, exist_ok=True)

    print(f"Dataset sweep: pipeline={args.pipeline}, stages={', '.join(args.stages)}, "
          f"{len(datasets)} dataset(s): {', '.join(datasets)}", flush=True)
    sweep_start = time.perf_counter()
    rows = []
    for position, dataset in enumerate(datasets, start=1):
        rows.append(run_dataset(
            base_config, args.pipeline, dataset, DATASETS[dataset], args.stages, args.timeout, args.prefix,
            kernel=args.kernel, point_label=f"dataset {position}/{len(datasets)}: ",
        ))
        # Save after each dataset so completed results survive an interrupted sweep.
        summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        elapsed = time.perf_counter() - sweep_start
        remaining = elapsed / position * (len(datasets) - position)
        print(f"dataset {position}/{len(datasets)} finished: {rows[-1]['status']} "
              f"(sweep elapsed {elapsed / 60:.1f} min, ~{remaining / 60:.1f} min left)", flush=True)

    print(f"\nSweep summary written to {summary_path}")
    for row in rows:
        timings = ", ".join(f"{stage} {seconds} s" for stage, seconds in row["runtime_seconds"].items())
        print(f"  {row['run_name']}: {row['status']} ({timings})")


if __name__ == "__main__":
    main()
