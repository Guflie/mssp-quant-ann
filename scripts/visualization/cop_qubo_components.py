"""QUBO size of every sub-problem left by the first step of the COP decomposition.

First step, as in results/02.B_CSP_pipeline.ipynb (sections 3-4): build the
COP (one variable per satellite, its domain the satellite's candidate slots), apply the exact
dominance_reduction, split into connected components. Each component is then translated to a QUBO
with csp_to_qubo (src/CSP_to_QUBO.py, default penalties), in each requested encoding.

Components with fewer than --min-satellites satellites (default 2: a satellite isolated from the
rest, which simply takes its heaviest slot) are trivial and only counted in the summary.

One row per significant component:
    satellites          COP variables
    constraints         COP constraints: satellite pairs with at least one conflicting slot pair
    conflict_pairs      conflicting slot pairs between the component's satellites (same antenna,
                        overlapping or closer than the reconfiguration time)
    avg_domain          mean domain size (slots per satellite, after dominance)
    max_domain          largest domain
    qubo_variables      QUBO variables (one per slot in both encodings)
    couplers_<encoding> logical couplings: non-zero off-diagonal QUBO coefficients
Written to <stem>_components_qubo.csv in --out-dir (default scripts/visualization/plot_cop_qubo_components/)
and printed.

The input is the path to one slot CSV.

Example:
    .venv/Scripts/python.exe scripts/visualization/cop_qubo_components.py data/test/slots_s500_g50.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(HERE))
from CSP import CSP  # noqa: E402
from CSP_decomposition import dominance_reduction, split_connected_components  # noqa: E402
from CSP_to_QUBO import csp_to_qubo  # noqa: E402
from cop_conflict_graph import COLUMNS, build_csp  # noqa: E402

DEFAULT_OUT_DIR = HERE / f"plot_{Path(__file__).stem}"  # scripts/visualization/plot_<script name>
ENCODINGS = ("domain_wall", "one_hot")


def conflict_pair_count(component: CSP) -> int:
    """Conflicting slot pairs between different satellites of the component (each pair once)."""
    return sum(len(component.conflicting_values(v, t)) for v in component.vars for t in v.domain) // 2


def component_row(index: int, component: CSP, weight: dict[str, float], encodings: tuple[str, ...]) -> dict:
    domains = [len(v.domain) for v in component.vars]
    row = {
        "component_id": f"component_{index:03d}",
        "satellites": len(component.vars),
        "constraints": component.num_constraints(),
        "conflict_pairs": conflict_pair_count(component),
        "avg_domain": sum(domains) / len(domains),
        "max_domain": max(domains),
        "qubo_variables": sum(domains),
    }
    for encoding in encodings:
        Q, index_map = csp_to_qubo(component, weight, encoding=encoding)
        assert len(index_map) == row["qubo_variables"]
        row[f"couplers_{encoding}"] = sum(1 for i, j in Q if i != j)
    return row


def run(slots_path: Path, encodings: tuple[str, ...], min_satellites: int, out_dir: Path) -> pd.DataFrame:
    print(f"\n{slots_path.stem}", flush=True)
    started = time.perf_counter()
    slots = pd.read_csv(slots_path, usecols=COLUMNS)
    weight = dict(zip(slots["slot_id"], slots["weight"].astype(float)))
    csp = build_csp(slots)
    input_slots = sum(len(v.domain) for v in csp.vars)
    dominated = dominance_reduction(csp, weight)
    components = split_connected_components(csp)    # largest first, as in the pipeline
    rows = [component_row(i, c, weight, encodings) for i, c in enumerate(components, start=1)
            if len(c.vars) >= min_satellites]
    trivial = [c for c in components if len(c.vars) < min_satellites]
    table = pd.DataFrame(rows)

    print(f"  {len(csp.vars)} satellites, {input_slots} slots; dominance removed {dominated} slots")
    print(f"  {len(components)} components: {len(rows)} significant (>= {min_satellites} satellites), "
          f"{len(trivial)} trivial ({sum(len(v.domain) for c in trivial for v in c.vars)} slots)")
    if not table.empty:
        with pd.option_context("display.width", 200, "display.max_rows", 200):
            print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{slots_path.stem}_components_qubo.csv"
        table.to_csv(out_path, index=False)
        print(f"  saved {out_path} ({time.perf_counter() - started:.1f} s)")
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("slots", type=Path, help="path to the slot CSV")
    parser.add_argument("--encodings", nargs="+", choices=ENCODINGS, default=list(ENCODINGS))
    parser.add_argument("--min-satellites", type=int, default=2,
                        help="smallest component reported (smaller ones are trivial)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="default: scripts/visualization/plot_cop_qubo_components/")
    args = parser.parse_args(argv)
    if not args.slots.is_file():
        parser.error(f"slot CSV not found: {args.slots}")
    run(args.slots, tuple(args.encodings), args.min_satellites, args.out_dir)


if __name__ == "__main__":
    main()
