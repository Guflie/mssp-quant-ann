# Split MWIS Pipeline

This folder contains the MWIS notebook workflow derived from `ground_station_pipeline.ipynb`.

## New files

- `mwis_pipeline.ipynb` - main MWIS scheduling pipeline.
- `pipeline_graphics.ipynb` - reads saved MWIS artifacts and creates figures.
- `qubo_annealing_pipeline.ipynb` - reads saved terminal subgraphs, builds QUBOs, and runs D-Wave simulated annealing as the standard final solver step.
- `README_mwis_pipeline.md` - this guide.

## Why the split was made

The original `ground_station_pipeline.ipynb` had grown too long. It mixed data loading, MWIS decomposition, small test cases, plotting code, QUBO construction, and annealing execution in one place.

The new workflow separates responsibilities:

```text
mwis_pipeline.ipynb
    -> creates standardized run artifacts

pipeline_graphics.ipynb
    -> reads those artifacts and creates figures

qubo_annealing_pipeline.ipynb
    -> reads terminal subgraphs and builds/runs QUBOs
```

The notebooks communicate through files, not by re-running each other's logic. They are kept together in `results/mwis/`, while generated run artifacts remain in `results/runs/mwis_pipeline/`.

## Default run folder

By default, the notebooks use:

```text
results/runs/mwis_pipeline/latest_mwis_run/
```

The main MWIS notebook creates this folder and writes the artifacts used by the other notebooks.

## How to run

Run the notebooks in this order from the `results/mwis` folder:

1. `mwis_pipeline.ipynb`
2. `pipeline_graphics.ipynb`
3. `qubo_annealing_pipeline.ipynb`

For a fresh machine, install the dependencies from a terminal before opening the notebooks:

```bash
py -3.13 -m pip install pandas networkx matplotlib nbformat nbclient ipykernel jupyter dwave-samplers
py -3.13 -m ipykernel install --user --name python313 --display-name "Python 3.13.14"
```

The notebooks are configured to use the `Python 3.13.14` Jupyter kernel. Paths are resolved from the project root, so the notebooks should also work if Jupyter starts from the repository root or from `results/`. The QUBO notebook builds QUBOs and then runs D-Wave simulated annealing with the maintained `dwave-samplers` package by default. This is the required local annealing backend for the project. A real QPU run uses the same QUBO artifacts, but depends on D-Wave hardware access and credentials.

## Main artifacts produced by `mwis_pipeline.ipynb`

| File | Meaning |
|---|---|
| `run_config.json` | Parameters used for the run. |
| `candidate_slots.csv` | Candidate slots loaded from Layer E. |
| `conflict_graph_nodes.csv` | One row per MWIS graph node. |
| `conflict_graph_edges.csv` | One row per conflict/coupler. |
| `components_summary.csv` | Natural connected-component statistics. |
| `decomposition_summary.csv` | Exact separator decomposition status by component. |
| `terminal_subgraphs.json` | QPU-compatible terminal subgraphs for QUBO construction. |
| `mwis_solution.csv` | Selected candidate contacts in the final MWIS solution. |
| `heuristic_fallback_details.csv` | Details for any incomplete components solved heuristically. |
| `final_summary.csv` | Main summary table with objective, bounds, and subgraph counts. |

## What was removed from the clean MWIS notebook

The new `mwis_pipeline.ipynb` intentionally removes or avoids:

- legacy Phase 1/2 compatibility discussion;
- the disabled Phase 4 top-k heuristic;
- small artificial test cases;
- long plotting functions;
- QUBO construction;
- annealing execution;
- repeated validation text that made the notebook hard to follow.

## Exactness and fallback policy

The MWIS notebook first tries to solve the graph exactly through:

1. natural connected-component decomposition;
2. exact separator decomposition;
3. exact terminal MWIS solving.

If a component cannot be decomposed within the configured limits, only that unresolved part is solved with a deterministic weighted-greedy heuristic.

When this happens, the final result is reported as:

```text
feasible objective <= true optimum <= conservative upper bound
```

If no fallback is used, the feasible objective and upper bound are equal, so the result is certified exact.

## Notes

- The legacy `ground_station_pipeline.ipynb` is kept in this folder for reference.
- The new graphics and QUBO notebooks should not rebuild the MWIS graph or decomposition.
- To create multiple experiment runs, change `RUN_NAME` in all three notebooks, or copy a completed `runs/mwis_pipeline/latest_mwis_run` folder to a descriptive experiment name.
