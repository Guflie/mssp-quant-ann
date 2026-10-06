## Project stucture

| File          | Description                                                   |
| ------------- | ------------------------------------------------------------- |
| `├── dataset` | Collection of dataset used to run experiments.                |
| `├── docs`    | Collection of documentation about the project.                |
| `├── results` | Collection of eperimental results (mainly Jupyter notebooks). |
| `└── src`     | Python module implementation.                                 |

## MWIS improvements branch handoff

This branch manually integrates the useful changes from `qubo_sweep_test` and
`data-wrangling-refactor-2`. The files are integrated in the working branch;
there is no formal Git merge commit from either source branch.

### What changed

- **Data wrangling:** Layers A–D keep intermediate files in
  `data/dataset/starlink/intermediate`. Layer E reads from there and writes the
  weighted candidate-slot file to `data/dataset/starlink`, preserving the
  downstream input path. The shared `results/01_data_wrangling/data_config.json`
  uses the agreed June 18, 2026 horizon and four-minute minimum contact.
- **Frozen inputs:** The Layer A satellite CSV and Layer B gateway CSV in
  `intermediate` are byte-for-byte the same as the frozen experiment snapshot.
  To keep experiments reproducible, run Layer C onward without rerunning A or
  B. Rerun A/B only when intentionally refreshing the source data.
- **MWIS model and decomposition:** The conflict graph includes the
  one-contact-per-satellite constraint. Natural connected components are
  independent; oversized components use exact separator-state decomposition
  where the configured search completes. If a budget is reached, the unresolved
  part is clearly marked and receives a feasible heuristic result plus a
  conservative upper bound; this is not certified optimal.
- **Memory and runtime:** Decomposition nodes store node IDs and compact
  metadata instead of a full NetworkX graph copy. Needed subgraphs are viewed
  from the original graph. Separator search avoids rescanning the graph for
  each articulation candidate; memoization reuses recursive results. Runtime
  and operation profiling are saved with each MWIS run.

- **Bounded MILP fallback after annealing:** `02.A_mwis_pipeline.ipynb` saves
  per-component objectives and bounds. After simulated annealing and schedule
  reconstruction, `3_qubo_annealing_reconstruction.ipynb` sends only natural
  components whose separator decomposition was incomplete to a classical
  binary MWIS solver (MILP). It has a 5-second per-component and 30-second total
  solver-time budget. A proven optimum replaces that component's contribution;
  if the time limit is reached, a valid better schedule may still be kept and
  the best valid MILP/triangle-LP bound is reported, falling back to the saved
  conservative bound if needed. All other components keep their annealing
  results. The comparison reports which method supplied each
  result, proof counts, timing, and the final objective and gap. This fallback
  complements simulated annealing; it is not a QPU run.
- **Why the fallback was chosen:** Isolated tests showed that triangle-LP cuts
  can tighten an upper bound but do not by themselves prove an MWIS optimum.
  The bounded MILP proved all unresolved components in the tested saved runs,
  so it is used only for separator-incomplete components. The tests and
  rationale are summarized in
  `results/archive/mwis_gap_reduction_findings.md`.

- **QPU-size-limit sweeps:** `scripts/run_qpu_sweep.py` runs the MWIS and QUBO/
  reconstruction notebooks at selected node/coupler limits, using the project's
  simulated annealer rather than a physical QPU. It supplies a
  per-point config through `SWE2_CONFIG`, fixes the exact-clique limit at 50,
  and preserves results in paired folders:
  `results/runs/mwis/pipeline_results/<run-name>` and
  `results/runs/mwis/qubo_annealing/<run-name>`. The notebooks use the
  `python313` kernel metadata.
- **Results organization:** Final experiment results and the archived previous
  model remain separate. The current optimality gap is still an open analysis
  item; this branch does not claim a quantum advantage.

### Checks and running

The QPU-sweep integration was smoke-tested at two parameter points. All pilot
run outputs and temporary sweep configs were removed after testing; the
reusable runner remains. The data-wrangling notebooks were checked for valid
notebook/config structure and consistent paths, but Layers A–E were not rerun,
so no source data was downloaded or regenerated.

The integrated simulated-annealing and reconstruction notebook was also run
end-to-end on the saved `exp2_s500_g50` producer input. It annealed 182 terminal
subgraphs (100 contributed to the schedule), then proved the one
separator-incomplete 2,419-node component optimal with MILP in 0.76 seconds.
The final 953-contact schedule had no graph conflicts, and its objective and
upper bound both equaled 64,329.426. The temporary smoke-test outputs and
config override were removed; frozen experiment results were not overwritten.

From the project root, a small sweep can be run with:

```powershell
python scripts/run_qpu_sweep.py --nodes 10 20 --couplers 100 --prefix qpusmoke
```

The script uses the Python kernel specified in the notebooks. A full default
sweep runs many more parameter combinations and can take substantially longer.
