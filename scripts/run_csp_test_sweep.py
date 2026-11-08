"""Run only the CSP pipeline notebook (02.B_CSP_pipeline.ipynb) once per dataset in ``data/test``.

Every ``slots_<name>.csv`` in the folder is one instance; its outputs go to
``results/runs/csp/pipeline_results/<prefix>/<name>`` (e.g. ``csp_test1/s500_g50``),
the same layout as a manual run. Every other setting (QPU limits, decomposition
budgets) comes from ``results/_config.json``, so all instances use the same parameters.

Each instance gets its own copy of the config (``sweep_config.json`` in its output
folder), passed to the notebook kernel through ``SWE2_CONFIG``; the notebook file
itself is not modified. The notebook overwrites the files of an existing run with
the same name. A summary of all instances is saved after each one in
``<prefix>/sweep_summary.json``, so an interrupted sweep keeps what it finished.

Run from the project root, for example:
    python scripts/run_csp_test_sweep.py
    python scripts/run_csp_test_sweep.py --datasets s200_g50 no-usa_s200_g50
    python scripts/run_csp_test_sweep.py --prefix csp_test2 --timeout 3600
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

# notebook execution and metric collection are shared with the other sweeps
from run_qpu_sweep import BASE_CONFIG_PATH, PIPELINES, PROJECT_ROOT, RESULTS_DIR, collect_metrics, execute_notebook

TEST_DIR = PROJECT_ROOT / "data" / "test"
CSP = PIPELINES["csp"]


def find_datasets() -> dict[str, Path]:
    """{name: csv} for every slots_<name>.csv in the test folder, sorted by name."""
    return {path.stem.removeprefix("slots_"): path for path in sorted(TEST_DIR.glob("slots_*.csv"))}


def run_dataset(base_config: dict[str, Any], name: str, csv_path: Path, prefix: str,
                timeout: int | None, kernel: str | None, label: str) -> dict[str, Any]:
    run_name = f"{prefix}/{name}"
    run_dir = RESULTS_DIR / "runs" / "csp" / "pipeline_results" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {label}{run_name} started at {time.strftime('%H:%M:%S')} ===", flush=True)

    config = {**base_config,
              CSP["slot_csv_key"]: csv_path.relative_to(PROJECT_ROOT).as_posix(),
              CSP["run_name_key"]: run_name}
    config_path = run_dir / "sweep_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    os.environ["SWE2_CONFIG"] = str(config_path)    # the notebook kernel inherits it

    row: dict[str, Any] = {"run_name": run_name, "dataset": name, "slot_csv": config[CSP["slot_csv_key"]],
                           "status": "ok"}
    start = time.perf_counter()
    try:
        row["kernel"] = execute_notebook(CSP["notebook"], timeout, kernel, log_prefix=f"[{name}]")
    except Exception as exc:
        row["status"] = f"failed: {type(exc).__name__}"
        (run_dir / "sweep_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"[{name}] FAILED: {type(exc).__name__} (see {run_dir / 'sweep_error.txt'})", flush=True)
    row["runtime_seconds"] = round(time.perf_counter() - start, 1)
    row.update(collect_metrics(config, "csp")["producer"])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", metavar="NAME",
                        help="instances to run, by name without 'slots_' (default: every CSV in data/test)")
    parser.add_argument("--prefix", default="csp_test1", help="run-name folder (default: csp_test1)")
    parser.add_argument("--timeout", type=int, default=None, help="per-cell timeout in seconds (default: none)")
    parser.add_argument("--kernel", default=None, help="Jupyter kernel (default: the notebook's own)")
    args = parser.parse_args()

    available = find_datasets()
    if not available:
        raise FileNotFoundError(f"no slots_*.csv found in {TEST_DIR}")
    names = args.datasets or list(available)
    unknown = [n for n in names if n not in available]
    if unknown:
        parser.error(f"unknown dataset(s) {unknown} (available: {', '.join(available)})")

    base_config = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    summary_path = RESULTS_DIR / "runs" / "csp" / "pipeline_results" / args.prefix / "sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"CSP sweep over {len(names)} instance(s) into {args.prefix}/: {', '.join(names)}", flush=True)

    rows, sweep_start = [], time.perf_counter()
    for position, name in enumerate(names, start=1):
        rows.append(run_dataset(base_config, name, available[name], args.prefix, args.timeout, args.kernel,
                                label=f"{position}/{len(names)}: "))
        summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        elapsed = time.perf_counter() - sweep_start
        print(f"{position}/{len(names)} {name}: {rows[-1]['status']} in {rows[-1]['runtime_seconds'] / 60:.1f} min "
              f"(sweep elapsed {elapsed / 60:.1f} min)", flush=True)

    print(f"\nSummary written to {summary_path}")
    for row in rows:
        print(f"  {row['run_name']}: {row['status']} ({row['runtime_seconds'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
