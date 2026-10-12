# Reordered pipeline: MWIS terminal-size sensitivity, 200 satellites / 50 gateways

The 20/100 and 50/350 profiles were executed afresh. The validated reordered 30/200 Experiment 1 baseline was reused as the reference. All three profiles passed independent validation.

The same frozen input (1,228 contacts and 7,647 conflict edges), Python 3.13.14, notebook sources, packages, execution method, decomposition budgets, 500 simulated-annealing reads, base seed 42, reward and penalty factors, and MILP/triangle-LP budgets were checked to match. Only the two terminal limits and output paths vary. Profiles change the variable and coupler limits jointly.

| Limits | Terminal QUBOs | Maximum terminal variables / maximum couplers | Incomplete components | Contacts | Before MILP | Final hybrid | Upper bound | Gap (%) | Runtime (s) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 20/100 | 29 | 13/78 | 1 | 180 | 12323.959 | 12581.450 | 12581.450 | 0.000 | 526.220 |
| 30/200 | 30 | 28/199 | 1 | 180 | 12323.959 | 12581.450 | 12581.450 | 0.000 | 683.947 |
| 50/350 | 31 | 30/228 | 1 | 180 | 12327.707 | 12581.450 | 12581.450 | 0.000 | 519.717 |

## Timing and fallback

| Limits | Decomposition (s) | Classical evaluation (s) | Simulated annealing (s) | MILP solver (s) | MILP components proven / attempted |
|---|---:|---:|---:|---:|---|
| 20/100 | 434.101 | 86.234 | 1.592 | 0.273 | 1/1 |
| 30/200 | 586.600 | 86.554 | 2.085 | 0.277 | 1/1 |
| 50/350 | 427.288 | 85.836 | 2.403 | 0.272 | 1/1 |

The pre-MILP schedule combines sampled terminal solutions, exact fixed choices, and greedy incomplete leaves. Final certificates apply to the whole hybrid pipeline. Equal final objectives can conceal sensitivity before MILP; the CSV also records producer bounds, incomplete nodes, terminal provenance, and available triangle-LP-only bounds.

These are single executions per profile, including the reused reference; runtime observations do not establish statistical differences. Notebook wall times include the classical baseline, and exclude interpreter startup, independent validation and comparison generation. D-Wave classical simulated annealing was used. The study does not measure physical QPU embedding or quantum performance.

Reused reference: `new_pipeline_results_reordered/exp1_s200_g50`. Existing experiments are preserved under their original names. New configurations, provenance, executed notebooks, timings and validators are under `results/runs/mwis/execution_logs/<run>/`. Campaign status and retention checks are under `results/runs/mwis/execution_logs/new_pipeline_results_reordered/exp4_terminal_sensitivity/`.

Files: `mwis_exp4_terminal_sensitivity_summary.csv` and `mwis_exp4_terminal_sensitivity.png`.

## Reordering verification

Same-input comparisons with the preserved previous runs check unchanged graphs, tree branches and existing QUBOs. Only fitting leaves previously rejected by recursion budgets are reclassified. Every added terminal is sampled.

| Profile | Previous QUBOs | Reordered QUBOs | Newly sampled terminals | Reclassified leaves |
|---|---:|---:|---:|---:|
| 20/100 | 27 | 29 | 2 | 4 |
| 30/200 | 27 | 30 | 3 | 5 |
| 50/350 | 28 | 31 | 3 | 5 |

## Final schedule provenance

The reconstruction category combines retained annealing and exact preprocessing. Sampled pieces in a component replaced by a proven MILP optimum do not contribute to the final schedule.

| Profile | Contacts from MILP components | Contacts from retained reconstruction | Retained annealed terminals |
|---|---:|---:|---:|
| 20/100 | 157 | 23 | 8 |
| 30/200 | 157 | 23 | 8 |
| 50/350 | 157 | 23 | 8 |

## Cross-profile structural verification

Independent comparisons of the completed trees confirmed that increasing 20/100 to 30/200 accepts the 28-variable, 199-coupler piece `component_001_dnode_10006`. Increasing 30/200 to 50/350 additionally accepts the 30-variable, 228-coupler piece `component_001_dnode_8226`. In each comparison, the other 10,024 tree nodes, separator branches, fixed choices and shared terminal QUBOs are unchanged.

The detailed audit is saved in [terminal_profile_structure_report.md](terminal_profile_structure_report.md), with machine-readable evidence in [terminal_profile_structure_validation.json](terminal_profile_structure_validation.json).
