# Reordered pipeline: MWIS Experiment 3 gateway scaling

The 200/25 and 200/100 inputs were executed afresh; the validated reordered 200/50 Experiment 1 baseline was reused. All three passed independent validation. The supplied CSVs are frozen and were not regenerated.

Python 3.13.14, packages, producer/annealing notebook sources, execution method, 30-variable / 200-coupler terminal limits, decomposition budgets, exact-solver limits, reward/penalty factors, 500 simulated-annealing reads, base seed 42 and MILP/triangle-LP budgets match across cases.

The accepted trimming policy selects active resources independently per scenario. Counts are exact, but resource identities are not fully nested: the 200/25 to 200/50 comparison replaces 37 satellite IDs, and the 200/50 to 200/100 comparison replaces 11. Only 152 satellite IDs are common to all three scenarios. These observations describe the supplied instances; composition changes can contribute to scaling differences.

| Active gateways | Contacts in input | Conflict edges | Selected contacts | Before MILP | Final objective | Upper bound | Gap (%) | Notebook runtime (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 915 | 3942 | 186 | 12482.484 | 12895.044 | 12895.044 | 0.000000 | 331.348 |
| 50 | 1228 | 7647 | 180 | 12323.959 | 12581.450 | 12581.450 | 0.000000 | 683.947 |
| 100 | 2033 | 23995 | 184 | 12827.587 | 13058.865 | 13058.865 | 0.000000 | 1337.326 |

## Timing and fallback

| Gateways | Decomposition (s) | Classical evaluation (s) | Simulated annealing (s) | MILP solver (s) | MILP proven / attempted |
|---:|---:|---:|---:|---:|---|
| 25 | 266.235 | 59.268 | 1.531 | 0.096 | 1/1 |
| 50 | 586.600 | 86.554 | 2.085 | 0.277 | 1/1 |
| 100 | 1126.680 | 202.013 | 1.705 | 1.704 | 1/1 |

These are single executions per case, including the reused baseline. Notebook wall times exclude interpreter startup, independent validation and comparison generation. Final certificates concern the full hybrid pipeline. Before-MILP objectives include sampled terminals, exact fixed choices and greedy incomplete leaves. This campaign used classical simulated annealing.

Per-run output and provenance folders use `new_pipeline_results_reordered/exp3_s200_g25`, `new_pipeline_results_reordered/exp1_s200_g50`, and `new_pipeline_results_reordered/exp3_s200_g100`. The CSV includes terminal counts, independent maximum sizes, bounds, incomplete components and fallback contributions. Existing results remain in their original folders.

## Reordering verification

The graph, existing terminal QUBOs and separator branches are unchanged from the same-input previous runs. Only fitting leaves previously rejected by recursion budgets are reclassified. All added terminal QUBOs are sampled.

| Gateways | Previous QUBOs | Reordered QUBOs | Newly sampled terminals | Reclassified leaves |
|---:|---:|---:|---:|---:|
| 25 | 25 | 26 | 1 | 5 |
| 50 | 27 | 30 | 3 | 5 |
| 100 | 28 | 30 | 2 | 5 |

## Final schedule provenance

Sampled terminals belonging to a component replaced by a proven MILP optimum do not contribute to the final schedule. The reconstruction category below includes retained annealing results and exact preprocessing choices.

| Gateways | Contacts from MILP components | Contacts from retained reconstruction | Retained annealed terminals |
|---:|---:|---:|---:|
| 25 | 174 | 12 | 5 |
| 50 | 157 | 23 | 8 |
| 100 | 168 | 16 | 6 |
