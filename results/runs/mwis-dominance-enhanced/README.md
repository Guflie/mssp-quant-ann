# Dominance-enhanced MWIS baseline

This is an isolated experiment using the frozen 200-satellite/50-gateway baseline, 30/200 logical limits, 500 simulated annealing reads and the existing decomposition/MILP budgets. The only algorithmic addition is exact same-satellite dominance before component splitting and during recursive preprocessing.

The producer is `results/02.A_mwis_pipeline_dominance_enhanced.ipynb`. The executor uses the unchanged shared `3_qubo_annealing.ipynb`, applying only an output-directory replacement to its execution copy. Per-run configuration is passed with `SWE2_CONFIG`; the global configuration is unchanged. Original and reduced graphs plus removal witnesses are saved separately. The original campaign is preserved as the control.
