# Dominance-enhanced MWIS baseline test

The copied producer applies exact same-satellite dominance before component splitting and at every uncached recursive node. The frozen input, contact conflicts, MWIS separator search, logical limits (30/200), decomposition budgets, terminal solvers, QUBO, 500 reads, seed schedule, and MILP budgets match the control. Python is 3.13.14. The shared annealing notebook has only an output-routing replacement in its execution copy.

Both runs use the exact same 1,228-contact frozen 200/50 input. All original candidate data are preserved. Dominance changes component boundaries while preserving the optimum. Consequently, the number of independent problems, terminal seeds and total reads differ even though the per-terminal sampling settings are held fixed.

| Measurement | Preserved control | Dominance enhanced |
|---|---:|---:|
| candidate contacts | 1,228 | 1,228 |
| contacts after global dominance | 1,228 | 460 |
| conflict edges after global dominance | 7,647 | 1,106 |
| components after global dominance | 20 | 90 |
| largest component contacts | 1,189 | 315 |
| largest component edges | 7,615 | 1,024 |
| decomposition tree nodes | 10,025 | 5,990 |
| incomplete tree leaves | 4,987 | 1,330 |
| incomplete components | 1 | 1 |
| terminal qubos sampled | 30 | 1,286 |
| terminal qubos retained in final schedule | 8 | 19 |
| max terminal variables | 28 | 30 |
| max terminal couplers | 199 | 119 |
| reads per terminal | 500 | 500 |
| total annealing reads | 15,000 | 643,000 |
| classical baseline objective | 12,323.959 | 12,576.412 |
| pre milp annealed objective | 12,323.959 | 12,576.412 |
| pre milp gap against certified optimum percent | 2.047 | 0.040 |
| final objective | 12,581.450 | 12,581.450 |
| final upper bound | 12,581.450 | 12,581.450 |
| final gap percent | 0.000 | 0.000 |
| final contacts | 180 | 180 |
| final conflicts | 0 | 0 |
| milp components proven optimal | 1 | 1 |
| milp solver seconds | 0.277 | 0.028 |
| decomposition seconds | 586.587 | 88.895 |
| classical evaluation seconds | 85.241 | 2,422.880 |
| notebook execution seconds | 683.947 | 2,808.389 |
| producer notebook seconds | 679.296 | 2,515.092 |
| annealing notebook seconds | 4.260 | 293.069 |
| final contacts from annealing | 12 | 34 |
| final contacts from exact preprocessing | 11 | 70 |
| final contacts from separator assignments | 0 | 0 |
| final contacts from greedy fallback | 0 | 0 |
| final contacts from milp | 157 | 76 |

The pre-MILP objective changed by 252.453. The notebook runtime was 4.11 times the control. The final objective is checked against the previously certified optimum of 12,581.450. Contact origins count contacts retained in the final schedule; they do not measure CPU effort or quantum advantage. Sampling uses D-Wave's classical simulated annealer.

Dominance is promising for improving pre-MILP quality and reducing dependence on component-level MILP: the pre-MILP gap falls from about 2.047% to 0.040%, and MILP supplies 76 rather than 157 final contacts. However, only 34 final contacts come from annealing; another 70 are fixed by exact classical preprocessing. Of the 1,286 sampled terminal QUBOs, only 19 are retained after MILP replaces the incomplete component. The additional annealing workload therefore includes many alternative branches that do not survive into the final schedule.

This variant is slower overall. Decomposition falls from 586.587 to 88.895 seconds, but classical terminal evaluation rises from 85.241 to 2,422.880 seconds. The unchanged solver policy exhaustively enumerates terminals with at most 22 vertices, resulting in 780 such terminals and 392,069,480 subsets. Sampling also increases from 15,000 to 643,000 reads with the per-terminal budget fixed at 500. Before rerunning the complete campaign, address exact terminal-verification cost in a separate controlled test, using a faster exact method and validating its objectives and bounds. Preserve this dominance-only run and the original campaign for comparison.

Validation passed: 300 small-instance exhaustive optimum checks; original conflict graph rebuilt independently from the frozen input; all global and recursive dominance witnesses checked; reduced graph and component membership checked; terminal limits, QUBO coefficients, read counts, seeds and energies checked; pre-MILP reconstruction evaluated independently; classical and final schedules checked against the full original conflicts; provenance and objective/bound consistency checked. Existing MWIS runs, source notebooks, global configuration and frozen input hashes are unchanged.

This is one baseline execution with no repetitions. Runtime values are descriptive and include exact classical terminal verification. A change in the number of annealed terminals is not itself evidence of better schedule quality: many alternative branches can be sampled and later replaced by component-level MILP. The current campaign remains the control; other experiments have not been rerun.

Outputs: `pipeline_results/baseline_s200_g50`, `qubo_annealing/baseline_s200_g50`, and `execution_logs/baseline_s200_g50`. The execution logs include the full per-run configuration, package/runtime provenance, executed notebooks, source and execution hashes, validation report and preservation manifest.
