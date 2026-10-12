# Dominance-enhanced MWIS: terminal-size sensitivity

Fresh 20/100 and 50/350 profiles executions are compared with the preserved reordered MWIS controls. The completed dominance 200/50 baseline is reused. All runs use the same frozen per-scenario input, Python 3.13.14, package versions, 500 reads per terminal, seed 42 and decomposition/MILP budgets. Only the paired logical variable/coupler limits vary among the profiles. Only the dominance producer differs; annealer output routing is changed in memory. No control, baseline, source notebook or global configuration is modified.

| Logical variables / couplers | Method | Retained contacts | Largest component | QUBOs | Incomplete leaves | Pre-MILP gap (%) | Final objective | Annealed / MILP contacts | Notebook seconds |
|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
| 20/100 | Current MWIS | 1228 | 1189 | 29 | 4988 | 2.046592 | 12581.450 | 12 / 157 | 525.989 |
| 20/100 | Dominance MWIS | 460 | 315 | 1022 | 1648 | 0.043993 | 12581.450 | 34 / 76 | 967.441 |
| 30/200 | Current MWIS | 1228 | 1189 | 30 | 4987 | 2.046592 | 12581.450 | 12 / 157 | 683.556 |
| 30/200 | Dominance MWIS | 460 | 315 | 1286 | 1330 | 0.040043 | 12581.450 | 34 / 76 | 2808.160 |
| 50/350 | Current MWIS | 1228 | 1189 | 31 | 4986 | 2.016803 | 12581.450 | 12 / 157 | 519.481 |
| 50/350 | Dominance MWIS | 460 | 315 | 1098 | 879 | 0.005460 | 12581.450 | 34 / 76 | 1234.133 |

Pre-MILP gaps are retrospective distances to the control campaign's certified optimum. Final annealing provenance counts only contacts retained after branch selection and MILP replacement; QUBO/read counts include discarded alternative branches. These are single executions, not replicated timing estimates. Notebook runtime excludes independent validation and outer orchestration; detailed phase times and pre-MILP provenance are in the CSV. Classical exact evaluation of terminals is included and can dominate runtime. Sampling is classical simulated annealing.

All three profiles use the identical frozen 200-satellite/50-gateway candidate CSV, including resource identities and weights. Both logical limits change together; this tests each combined profile rather than isolating a single limit. The 30/200 baseline is reused and is not an independent repetition.

All existing experiment artifacts and figure/analysis outputs were checked unchanged by SHA-256. Each new case was independently checked for original conflicts, schedule feasibility, objective/bound consistency, QUBO coefficients, caps, reads and seeds, dominance witnesses, reconstruction and final provenance.


## Observed tradeoffs

All three dominance profiles return 180 conflict-free contacts with certified objective and upper bound 12,581.450. The 30/200 row is the existing baseline, reused rather than rerun. Each profile uses the identical input file and the same 500 reads per terminal, decomposition budgets, classical solver settings and fallback limits. The global reduction is identical: 1,228 candidates become 460 contacts in 90 natural components, with a largest component of 315 contacts.

The 20/100 profile samples 1,022 QUBOs (511,000 reads), with maximum dimensions 20 variables and 70 couplers. It leaves 1,648 incomplete leaves. Its sampled pre-MILP objective is 12,575.915, a retrospective gap of 0.044%, compared with the control's 2.047%. Notebook time is 967.441 seconds (about 16 minutes 7 seconds), compared with 525.989 seconds for the control. Its classical-evaluation cell costs 732.404 seconds and its annealing/reconstruction notebook costs 147.754 seconds.

The reused 30/200 baseline samples 1,286 QUBOs (643,000 reads), leaves 1,330 incomplete leaves and achieves a pre-MILP objective of 12,576.412, a gap of 0.040%. Its 2,808.160-second notebook time is dominated by 2,423.417 seconds of classical evaluation. The small quality difference from 20/100 does not justify assuming that this profile is the best runtime/quality tradeoff.

The 50/350 profile samples 1,098 QUBOs (549,000 reads), with maximum dimensions 50 variables and 211 couplers. Incomplete leaves decrease to 879. Its sampled pre-MILP objective is 12,580.763, leaving a gap of 0.005460% (an absolute difference of 0.687 from the final optimum), compared with the control's 2.017%. Notebook time is 1,234.133 seconds (about 20 minutes 34 seconds), compared with 519.481 seconds for the control. Classical evaluation costs 660.837 seconds, while the annealing/reconstruction notebook costs 501.093 seconds. The larger profile improves intermediate quality, but its sampling workload costs more time than 20/100 despite having fewer total reads than 30/200.

The 50/350 sampled reconstruction is 0.205 below its exact classical tree reference (12,580.968). This is a small but real sampling loss; the MILP fallback subsequently resolves the final gap. The other two dominance profiles match their classical tree reference objectives. Final optimality belongs to the complete hybrid pipeline, not to an assumption that every sampled terminal reached its optimum.

All profiles retain the same final provenance: 34 annealed contacts, 70 contacts fixed by exact preprocessing and 76 contacts from MILP, versus 12 annealed, 11 fixed and 157 MILP contacts in each control profile. Each dominance profile still has one separator-incomplete natural component, so relaxing the limits does not eliminate whole-component MILP replacement. Their pre-MILP schedules contain 81, 80 and 82 annealed contacts for 20/100, 30/200 and 50/350 respectively; many are subsequently attributed to the replacing MILP solution. Final retained contributions and total sampled workloads therefore answer different questions.

Neither QUBO counts nor runtime vary monotonically with the terminal limits. The profiles change the bounded decomposition trees and the set of exported alternative branches. They also change which exact backend evaluates a terminal: enumeration through 22 vertices versus a weighted-clique solver above that threshold through 50 vertices. The traces show smaller enumerated terminals taking longer than some larger clique-solved terminals. Both limits change together, so the experiment does not isolate the effect of either logical variable or coupler capacity. No physical embedding was performed.

For the full seven-configuration adoption comparison, see ../campaign_assessment/adoption_assessment.md and its paired CSV/JSON records. These are single observed executions rather than replicated timing estimates. All 2,039 preserved source, input, earlier result and official figure/analysis files passed SHA-256 preservation checks. No experiment was rerun to generate these comparisons.
