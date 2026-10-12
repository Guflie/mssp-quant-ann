# Dominance-enhanced MWIS: gateway scaling

Fresh 200/25 and 200/100 executions are compared with the preserved reordered MWIS controls. The completed dominance 200/50 baseline is reused. All runs use the same frozen per-scenario input, Python 3.13.14, package versions, 30/200 logical limits, 500 reads per terminal, seed 42 and decomposition/MILP budgets. Only the dominance producer differs; annealer output routing is changed in memory. No control, baseline, source notebook or global configuration is modified.

| Gateways | Method | Retained contacts | Largest component | QUBOs | Incomplete leaves | Pre-MILP gap (%) | Final objective | Annealed / MILP contacts | Notebook seconds |
|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
| 25 | Current MWIS | 915 | 897 | 26 | 4986 | 3.199369 | 12895.044 | 6 / 174 | 331.111 |
| 25 | Dominance MWIS | 588 | 503 | 36 | 2736 | 2.400372 | 12895.044 | 24 / 127 | 231.264 |
| 50 | Current MWIS | 1228 | 1189 | 30 | 4987 | 2.046592 | 12581.450 | 12 / 157 | 683.556 |
| 50 | Dominance MWIS | 460 | 315 | 1286 | 1330 | 0.040043 | 12581.450 | 34 / 76 | 2808.160 |
| 100 | Current MWIS | 2033 | 2000 | 30 | 4938 | 1.771042 | 13058.865 | 10 / 168 | 1337.093 |
| 100 | Dominance MWIS | 442 | 264 | 84 | 118 | 0.082052 | 13058.865 | 35 / 46 | 115.775 |

Pre-MILP gaps are retrospective distances to the control campaign's certified optimum. Final annealing provenance counts only contacts retained after branch selection and MILP replacement; QUBO/read counts include discarded alternative branches. These are single executions, not replicated timing estimates. Notebook runtime excludes independent validation and outer orchestration; detailed phase times and pre-MILP provenance are in the CSV. Classical exact evaluation of terminals is included and can dominate runtime. Sampling is classical simulated annealing.

The active-resource inputs are selected independently per scenario. Changing the gateway count also changes the selected satellite identities. Comparisons of the variant and control use an identical input; cross-scenario scaling also reflects these composition differences.

All existing experiment artifacts and figure/analysis outputs were checked unchanged by SHA-256. Each new case was independently checked for original conflicts, schedule feasibility, objective/bound consistency, QUBO coefficients, caps, reads and seeds, dominance witnesses, reconstruction and final provenance.


## Observed tradeoffs

The new 200/25 and 200/100 executions both improve pre-MILP quality and observed runtime relative to the current MWIS controls. The 200/50 row reuses the prior dominance baseline, whose runtime tradeoff remains less favorable. All three configurations retain identical certified final objectives and contact counts to their controls, with zero conflicts.

For 200/25, global dominance removes 327 of 915 contacts (35.7%). The largest component shrinks from 897 to 503 contacts, and natural components increase from 12 to 49. The pre-MILP objective rises from 12,482.484 to 12,585.515, reducing the retrospective gap from 3.199% to 2.400%. Final annealing contacts increase from six to 24, while MILP contacts decrease from 174 to 127; 35 contacts are fixed by exact preprocessing. The large component remains incomplete, with 2,736 incomplete leaves rather than 4,986. Sampling workload increases moderately from 26 to 36 QUBOs (13,000 to 18,000 reads). Decomposition drops from 266.229 to 177.287 seconds, classical evaluation from 57.959 to 47.396 seconds, and total notebook time from 331.111 to 231.264 seconds.

For 200/100, dominance removes 1,591 of 2,033 contacts (78.3%). The largest component shrinks from 2,000 to 264 contacts and natural components increase from 13 to 119. The pre-MILP objective rises from 12,827.587 to 13,048.150, leaving a gap of 0.082% instead of 1.771%. Final annealing contacts increase from 10 to 35, MILP contacts decrease from 168 to 46, and 103 contacts are fixed by exact preprocessing. Incomplete leaves fall from 4,938 to 118, though one natural component still requires MILP certification. The run samples 84 QUBOs instead of 30 (42,000 rather than 15,000 reads). Decomposition takes 9.359 seconds instead of 1,126.675, classical evaluation takes 87.351 instead of 200.631, and total notebook time falls from 1,337.093 to 115.775 seconds.

The reused 200/50 baseline reaches a smaller pre-MILP gap (2.047% to 0.040%) and retains more annealing contacts (12 to 34), but generates 1,286 QUBOs instead of 30. Its classical evaluation consumes approximately 2,423 seconds; notebook runtime increases from 683.556 to 2,808.160 seconds. This is the same saved baseline used in Experiments 1 and 2, not a fresh repetition or an additional statistical sample.

These results reinforce that dominance is mathematically safe and can substantially improve the division of work in the implemented MWIS pipeline. Its workload effect depends on the resulting graph and separator tree. Raw input size alone does not explain runtime: 200/100 starts with more candidates than 200/50, but after dominance its tree exports far fewer alternative terminal problems, and it runs much faster. Experiment 2's 500/50 result remains an important counterexample to a claim of uniformly better performance.

All three dominance configurations have sampled pre-MILP objectives equal to their classical tree reference objectives. The final provenance counts identify contacts retained from sampled terminals; they do not establish an objective improvement over classical terminal evaluation, and more sampled QUBOs do not guarantee more retained annealing contacts. Sampling here is classical simulated annealing, and the timing figures are single observed executions rather than replicated performance estimates. The candidate-contact model's certified optimum is distinct from unrestricted operational satellite scheduling.

No source notebook, global configuration, frozen input, original campaign, previous dominance result or official figure/analysis was modified. Independent validation checked the original conflict graph, weighted objectives, schedule feasibility, dominance witnesses, QUBO coefficients/caps, reads/seeds, pre-MILP reconstruction and final provenance. Hash checks confirmed that all 1,942 preserved files were unchanged. The reporting parser accepts integer-valued CSV counts such as `0.0`; this change concerns reporting only.
