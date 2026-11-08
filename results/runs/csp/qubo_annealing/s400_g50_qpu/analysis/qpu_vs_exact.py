"""Exact optimum of every terminal (MILP, HiGHS) vs the QPU reads of the s400_g50_qpu run."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path(r"d:/Projects/mmsp-quant-ann/results/runs/csp")
qubos = {t["terminal_id"]: t for t in json.loads((ROOT / "pipeline_results/csp_test1/s400_g50/csp_terminal_qubos.json").read_text())}
subs = {t["terminal_id"]: t for t in json.loads((ROOT / "pipeline_results/csp_test1/s400_g50/csp_terminal_subproblems.json").read_text())}
run = ROOT / "qubo_annealing/s400_g50_qpu"
results = pd.read_csv(run / "annealing_results.csv")
reads = pd.read_csv(run / "annealing_reads.csv")


def exact_optimum(terminal_id: str) -> float:
    q = qubos[terminal_id]
    slots = sorted(q["slot_weights"])
    index = {s: i for i, s in enumerate(slots)}
    rows = [(index[a], index[b]) for a, b in q["conflicts"] if a in index and b in index]
    # at most one slot per satellite (also implied by the conflicts, added for safety)
    groups = [[index[str(s)] for s in dom if str(s) in index] for dom in subs[terminal_id]["domains"].values()]
    groups = [g for g in groups if len(g) > 1]
    A = lil_matrix((len(rows) + len(groups), len(slots)))
    for r, (i, j) in enumerate(rows):
        A[r, i] = A[r, j] = 1
    for r, g in enumerate(groups, start=len(rows)):
        for i in g:
            A[r, i] = 1
    c = -np.array([q["slot_weights"][s] for s in slots])
    constraints = [LinearConstraint(A.tocsr(), -np.inf, 1)] if A.shape[0] else []
    res = milp(c, constraints=constraints, integrality=np.ones(len(slots)), bounds=Bounds(0, 1))
    assert res.status == 0, (terminal_id, res.message)
    return -res.fun


start = time.perf_counter()
optimum = {t: exact_optimum(t) for t in results.terminal_id}
print(f"exact optima of {len(optimum)} terminals in {time.perf_counter() - start:.1f} s")

reads["optimum"] = reads.terminal_id.map(optimum)
reads["is_optimal"] = reads.is_independent & np.isclose(reads.objective, reads.optimum, rtol=0, atol=1e-6)
per = reads.groupby("terminal_id").apply(lambda g: pd.Series({
    "reads": g.num_occurrences.sum(),
    "p_feasible": g.num_occurrences[g.is_independent].sum() / g.num_occurrences.sum(),
    "p_optimal": g.num_occurrences[g.is_optimal].sum() / g.num_occurrences.sum(),
    "best_feasible": g.objective[g.is_independent].max() if g.is_independent.any() else np.nan,
}), include_groups=False).reset_index()
qsum = pd.read_csv(run / "terminal_qubo_summary.csv")[["terminal_id", "variables", "couplers"]]
per = per.merge(qsum, on="terminal_id").merge(results[["terminal_id", "chain_break_fraction_mean", "physical_qubits", "max_chain_length", "qpu_access_time_us"]], on="terminal_id")
per["optimum"] = per.terminal_id.map(optimum)
per["best_ratio"] = per.best_feasible / per.optimum
per["found_optimum"] = per.p_optimal > 0
per["size"] = pd.cut(per.variables, [0, 30, 100, 150, 250], labels=["<=30", "31-100", "101-150", "151-250"])

pd.set_option("display.width", 200)
print(per.groupby("size", observed=True).agg(
    terminals=("terminal_id", "size"), feasible_terminals=("best_feasible", lambda s: s.notna().sum()),
    optimum_found=("found_optimum", "sum"), p_feasible=("p_feasible", "mean"), p_optimal=("p_optimal", "mean"),
    best_over_opt_med=("best_ratio", "median"), best_over_opt_min=("best_ratio", "min"),
    chain_breaks=("chain_break_fraction_mean", "mean"), phys_q=("physical_qubits", "median"),
    chain_max=("max_chain_length", "max")).round(4).to_string())
print("\noverall: terminals", len(per), "| feasible", int(per.best_feasible.notna().sum()), "| optimum found", int(per.found_optimum.sum()),
      "| QPU access s", round(per.qpu_access_time_us.sum() / 1e6, 3),
      "| sum best/sum opt", round(per.best_feasible.sum() / per.optimum.sum(), 4))
per.to_csv(Path(__file__).with_name("qpu_vs_exact_per_terminal.csv"), index=False)
