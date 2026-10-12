# Reordered pipeline: MWIS Experiment 2 satellite scaling

The 100/50 and 500/50 inputs were executed afresh; the validated reordered 200/50 Experiment 1 baseline was reused. All three passed independent validation. The supplied CSVs are frozen and were not regenerated.

Python 3.13.14, packages, producer/annealing notebook sources, execution method, 30-variable / 200-coupler terminal limits, decomposition budgets, exact-solver limits, reward/penalty factors, 500 simulated-annealing reads, base seed 42 and MILP/triangle-LP budgets match across cases.

The accepted trimming policy selects active resources independently per scenario. Counts are exact, but resource identities are not fully nested: the 500/50 case exchanges one gateway relative to the baseline and loses two baseline satellite IDs. These observations describe the supplied instances; composition changes can contribute to scaling differences.

| Active satellites | Contacts in input | Conflict edges | Selected contacts | Before MILP | Final objective | Upper bound | Gap (%) | Notebook runtime (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 656 | 3933 | 96 | 6702.815 | 6734.972 | 6734.972 | 0.000000 | 339.079 |
| 200 | 1228 | 7647 | 180 | 12323.959 | 12581.450 | 12581.450 | 0.000000 | 683.947 |
| 500 | 5224 | 53998 | 464 | 32417.936 | 33972.808 | 33972.808 | 0.000000 | 87.109 |

## Timing and fallback

| Satellites | Decomposition (s) | Classical evaluation (s) | Simulated annealing (s) | MILP solver (s) | MILP proven / attempted |
|---:|---:|---:|---:|---:|---|
| 100 | 77.325 | 223.924 | 34.330 | 0.152 | 3/3 |
| 200 | 586.600 | 86.554 | 2.085 | 0.277 | 1/1 |
| 500 | 73.199 | 1.833 | 0.394 | 2.105 | 1/1 |

These are single executions per case, including the reused baseline. Notebook wall times exclude interpreter startup, independent validation and comparison generation. Final certificates concern the full hybrid pipeline. Before-MILP objectives include sampled terminals, exact fixed choices and greedy incomplete leaves. This campaign used classical simulated annealing.

Per-run output and provenance folders use `new_pipeline_results_reordered/exp2_s100_g50`, `new_pipeline_results_reordered/exp1_s200_g50`, and `new_pipeline_results_reordered/exp2_s500_g50`. The CSV includes terminal counts, independent maximum sizes, bounds, incomplete components and fallback contributions. Existing results remain in their original folders.

## Reordering verification

All cases use the reordered source revision. The reused baseline has three additional sampled terminal QUBOs and two newly resolved empty leaves compared with its preserved earlier run. For the freshly executed scaling cases, independent structural validation compares the same-input old and reordered trees without invoking a solver.

| Satellites | Previous QUBOs | Reordered QUBOs | Newly sampled terminals | Reclassified leaves |
|---:|---:|---:|---:|---:|
| 100 | 145 | 194 | 49 | 51 |
| 200 | 27 | 30 | 3 | 5 |
| 500 | 4 | 4 | 0 | 0 |
