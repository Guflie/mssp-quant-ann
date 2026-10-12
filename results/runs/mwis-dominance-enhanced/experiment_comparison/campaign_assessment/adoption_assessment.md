# Dominance-enhanced MWIS: complete campaign assessment

Seven distinct configurations compare the preserved reordered MWIS campaign with the isolated dominance variant. The 200/50, 30/200 baseline is counted once; its reuse across experiment groups is not an independent repetition. Within each pair, inputs, package versions, logical limits, 500 reads per terminal, seed and other configuration values match. The producer adds same-satellite dominance; the shared annealer changes only its output routing in memory.

Values show current MWIS → dominance-enhanced MWIS.

| Scenario | Pre-MILP gap (%) | Final annealed contacts | Final MILP contacts | Sampled QUBOs | Notebook time |
|---|---:|---:|---:|---:|---|
| 100/50; 30/200 | 0.477 → 0.000 | 15 → 26 | 72 → 0 | 194 → 11 | 5m39s → 0m11s |
| 200/50; 30/200 | 2.047 → 0.040 | 12 → 34 | 157 → 76 | 30 → 1286 | 11m24s → 46m48s |
| 500/50; 30/200 | 4.577 → 4.510 | 3 → 3 | 459 → 449 | 4 → 6 | 1m27s → 2h10m17s |
| 200/25; 30/200 | 3.199 → 2.400 | 6 → 24 | 174 → 127 | 26 → 36 | 5m31s → 3m51s |
| 200/100; 30/200 | 1.771 → 0.082 | 10 → 35 | 168 → 46 | 30 → 84 | 22m17s → 1m56s |
| 200/50; 20/100 | 2.047 → 0.044 | 12 → 34 | 157 → 76 | 29 → 1022 | 8m46s → 16m07s |
| 200/50; 50/350 | 2.017 → 0.005 | 12 → 34 | 157 → 76 | 31 → 1098 | 8m39s → 20m34s |

Dominance improves pre-MILP quality in 7/7 configurations, retains more final annealed contacts in 6/7, and is faster in 3/7 observed runs. Final objectives agree in 7/7 pairs. All 14/14 records are certified optimal; all 14 have zero conflicts.

The reduction is supported by the existing exhaustive tests and per-run independent dominance-witness checks. Its strongest practical benefits are reduced fallback dependence and better pre-MILP schedules, rather than a higher final objective. Its runtime effect depends on the graph and bounded decomposition tree. The 500/50 case is the clearest counterexample: only a small pre-MILP improvement, unchanged retained annealing contribution and approximately two hours of separator decomposition.

The 200/50 baseline and terminal-profile timings also include exact classical evaluation of all alternative terminals. The unchanged implementation enumerates subsets through 22 vertices and uses a weighted-clique solver above that threshold, through 50 vertices. The traces show some smaller enumerated terminals taking substantially longer than larger clique-solved terminals. This is a verification implementation bottleneck; observed timing differences should not be attributed solely to dominance, QUBO size or sampling. No solver choice or algorithm was changed mid-campaign.

The sampled pre-MILP objective matches the classical tree reference in 6/7 dominance configurations. For n50_c350 it is 0.205 lower. The subsequent MILP resolves the final gap. No improvement over the exact classical terminal-reference objective was observed. This distinguishes a benefit of the reduction/decomposition pipeline from a benefit attributable to sampling alone.

My recommendation is to retain both complete campaigns and keep dominance as a separately evaluated improvement for now. The mathematical reduction is useful and safe for this model. A blanket replacement with the current implementation would conceal substantial runtime regressions. Before adopting it as the default, test a faster exact terminal-evaluation strategy and a way to limit expensive separator exploration on large, weakly separated components. Those changes require validation and fresh timing measurements; they were not implemented in this campaign.

Counts of sampled QUBOs include discarded alternative branches. Final provenance records contacts from sampled terminals that survive branch selection and MILP replacement. A MILP-replaced component is attributed to MILP as a whole, even when some contact identities also appeared in the pre-MILP incumbent. These are complementary measurements, not evidence that sampling alone caused an objective improvement. Sampling was classical simulated annealing; no quantum hardware or physical embedding was tested.

If the dominance campaign is adopted for the report, regenerate the CSP/MWIS figures and revise the analysis explicitly around the new implementation. The final annealed/MILP contact counts now both match CSP in 7/7 matched configurations. This changes the earlier provenance comparison and supports a more careful distinction between preprocessing and formulation effects. CSP has a different separator method, encoding, 1,000 reads per terminal and incomplete hardware provenance, so runtime comparisons remain descriptive.

All comparisons use single saved executions. Pre-MILP gaps are retrospective distances to certified final optima; the looser bounds available before fallback are different metrics. Cross-scenario active-resource identities are not fully nested. Only terminal-size sensitivity uses the identical input across all profiles. Certificates concern the finite candidate-contact model, not unrestricted operational MMSP.

Detailed source hashes and paired records are saved beside this assessment. No experiment, official figure or original analysis was modified.
