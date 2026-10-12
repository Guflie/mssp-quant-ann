# Reordered budget checks: MWIS Experiment 1 baseline

This run uses the finalized supplied active-count 200/50 CSV without regenerating Data Wrangling outputs. Scenario-dependent resource selection is accepted for this campaign. The old input and experiment outputs remain preserved.

Input: `data/experiment_inputs/exp_snapshot_2026-10-12/slots_s200_g50.csv`. SHA-256: `8c7f74c2e3b02e7c8158429b0fa9f21df7bbdc3445de913fdf563f8c3e560c8e`.

Python 3.13.14; 30 logical variables / 200 couplers; separator and exact-solver limits unchanged, with resolved/fitting leaves accepted before exhausted recursion budgets; 500 classical simulated-annealing reads with base seed 42; MILP 5 seconds/component and 30 seconds total solver time; triangle-LP budgets unchanged.

| Metric | Value |
|---|---:|
| Active satellites / gateways | 200 / 50 |
| Candidates / conflict edges | 1228 / 7647 |
| Natural components | 20 |
| Terminal QUBOs | 30 |
| Independent maxima: variables / couplers | 28 / 199 |
| Incomplete components | 1 |
| Selected contacts / conflicts | 180 / 0 |
| Classical baseline objective | 12323.959 |
| Before MILP | 12323.959 |
| Final hybrid objective | 12581.450 |
| Hybrid upper bound | 12581.450 |
| Certified gap (%) | 0.000000 |
| Status | hybrid_exact |
| MILP components proven / attempted | 1 / 1 |
| Notebook runtime (s) | 683.947 |

## Timing breakdown

| Stage | Seconds |
|---|---:|
| Decomposition | 586.600 |
| Classical evaluation | 86.554 |
| Simulated annealing | 2.085 |

Notebook wall time excludes interpreter startup, independent validation and report generation. This is one execution. Certificates apply to the complete hybrid pipeline, including classical fallback. The pre-MILP schedule includes sampled terminals, exact fixed choices and greedy incomplete leaves.

Producer outputs: `results/runs/mwis/pipeline_results/new_pipeline_results_reordered/exp1_s200_g50`.
Annealing outputs: `results/runs/mwis/qubo_annealing/new_pipeline_results_reordered/exp1_s200_g50`.

Independent validation rebuilt all conflicts from the frozen CSV, checked schedules and objectives, terminal QUBOs, read counts and seeds, and exercised the comparison loader using the explicit new snapshot path. Configuration, source hashes, packages, executed notebooks, timings and validators are saved in this directory.

## Compared with the preserved baseline

| Metric | Before ordering change | Reordered |
|---|---:|---:|
| Terminal QUBOs | 27 | 30 |
| Maximum terminal variables | 13 | 28 |
| Maximum terminal couplers | 78 | 199 |
| Before-MILP objective | 12323.959 | 12323.959 |
| Conservative upper bound before MILP | 77004.590 | 75644.180 |
| Incomplete leaves across all branches | 4992 | 4987 |
| Final objective | 12581.450 | 12581.450 |
| Upper bound | 12581.450 | 12581.450 |
| Selected contacts | 180 | 180 |
| Incomplete components | 1 | 1 |
| Notebook wall seconds | 527.168 | 683.947 |

The input, conflict graph, node IDs, branch structure, separator choices and budgets were checked against the preserved baseline. Only the five expected budget-blocked leaves were reclassified: three nonempty terminals and two leaves resolved by exact preprocessing. The newly accepted 28-variable/199-coupler piece was verified in the annealing samples. Existing terminal QUBOs remain identical. Additional terminals change the index-based seeds assigned to later terminals; the base seed and seed policy are unchanged.

The original baseline is preserved under `new_pipeline_results/exp1_s200_g50`. Boundary checks, old/new source backups, source hashes and the structural comparison are stored under `new_pipeline_results_reordered`. This request executes only the baseline.

The baseline's feasible objective before MILP did not improve in this execution; the change increased the terminal coverage. The observed notebook runtime was 683.947 seconds versus 527.168 seconds previously. Most of this difference occurred during decomposition; classical evaluation remained approximately 86.6 seconds and simulated annealing increased from about 1.6 to 2.1 seconds. These single executions do not isolate the cause of the runtime difference or establish a general performance effect.
