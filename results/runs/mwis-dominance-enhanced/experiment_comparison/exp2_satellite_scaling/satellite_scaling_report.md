# Dominance-enhanced MWIS: satellite scaling

Fresh 100/50 and 500/50 executions are compared with the preserved reordered MWIS controls. The completed dominance 200/50 baseline is reused. All runs use the same frozen per-scenario input, Python 3.13.14, package versions, 30/200 logical limits, 500 reads per terminal, seed 42 and decomposition/MILP budgets. Only the dominance producer differs; annealer output routing is changed in memory. No control, baseline, source notebook or global configuration is modified.

| Satellites | Method | Retained contacts | Largest component | QUBOs | Incomplete leaves | Pre-MILP gap (%) | Final objective | Annealed / MILP contacts | Notebook seconds |
|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
| 100 | Current MWIS | 656 | 454 | 194 | 927 | 0.477463 | 6734.972 | 15 / 72 | 338.853 |
| 100 | Dominance MWIS | 169 | 20 | 11 | 0 | 0.000000 | 6734.972 | 26 / 0 | 10.919 |
| 200 | Current MWIS | 1228 | 1189 | 30 | 4987 | 2.046592 | 12581.450 | 12 / 157 | 683.556 |
| 200 | Dominance MWIS | 460 | 315 | 1286 | 1330 | 0.040043 | 12581.450 | 34 / 76 | 2808.160 |
| 500 | Current MWIS | 5224 | 5215 | 4 | 16 | 4.576813 | 33972.808 | 3 / 459 | 86.892 |
| 500 | Dominance MWIS | 4553 | 4534 | 6 | 4991 | 4.509854 | 33972.808 | 3 / 449 | 7817.367 |

Pre-MILP gaps are retrospective distances to the control campaign's certified optimum. Final annealing provenance counts only contacts retained after branch selection and MILP replacement; QUBO/read counts include discarded alternative branches. These are single executions, not replicated timing estimates. Notebook runtime excludes independent validation and outer orchestration; detailed phase times and pre-MILP provenance are in the CSV. Classical exact evaluation of terminals is included and can dominate runtime. Sampling is classical simulated annealing.

The supplied active-resource subsets are not fully nested; in particular the 500/50 selection exchanges a gateway and drops two baseline satellite identities. The comparisons isolate the dominance addition within each identical input; cross-scenario scaling also reflects this composition difference.

All existing experiment artifacts and figure/analysis outputs were checked unchanged by SHA-256. Each new case was independently checked for original conflicts, schedule feasibility, objective/bound consistency, QUBO coefficients, caps, reads and seeds, dominance witnesses, reconstruction and final provenance.


## Observed tradeoffs

At 100/50, dominance removes 487 of 656 contacts (74.2%), reduces the largest component from 454 to 20 contacts and eliminates all incomplete leaves. The optimum is obtained before MILP, with 26 contacts from sampling and 70 from exact preprocessing. No MILP is needed. Sampling workload decreases from 194 to 11 QUBOs, while the final retained annealing contribution increases from 15 to 26 contacts. The smaller workload is beneficial here because the reduced components fit directly, without a large tree of alternative separator states. Notebook time decreases from 338.853 to 10.919 seconds.

The reused 200/50 baseline illustrates a different tradeoff. Its pre-MILP objective improves from 12,323.959 to 12,576.412, reducing the retrospective gap from 2.047% to 0.040%. Final annealed contacts increase from 12 to 34 and MILP contacts decrease from 157 to 76. However, 1,286 QUBOs are sampled instead of 30, and notebook time rises from 683.556 to 2,808.160 seconds. Classical evaluation, at about 2,423 seconds, is the main runtime cost in this variant's baseline.

At 500/50, dominance removes only 671 of 5,224 contacts (12.8%). The largest component decreases from 5,215 to 4,534 contacts and remains incomplete. The decomposition expands to 10,020 recorded nodes, with 9,980 separator states and 6,437 cache hits; 4,991 incomplete leaves remain, compared with 16 in the control. The 10,000-node setting limits further expansion, rather than imposing a strict cap on all recorded terminal/incomplete nodes: pending branches may still be recorded after expansion stops. This is consistent with the reordered terminal/budget checks.

The expanded 500/50 search costs 7,352.596 seconds in the decomposition cell and 446.137 seconds in classical evaluation. It generates only six non-empty QUBOs, compared with four in the control; both versions retain one sampled terminal and three annealed contacts in the final schedule. Before MILP, the objective increases by just 22.748, from 32,417.936 to 32,440.684. The gap improves by approximately 0.067 percentage points, from 4.577% to 4.510%. The final dominance schedule still uses 449 MILP contacts (96.8% of its 464 contacts), versus 459 in the control; its other contacts are three annealed and 12 fixed by exact preprocessing. MILP proves optimality in 2.365 seconds. The increase in total notebook time from 86.892 to 7,817.367 seconds is therefore chiefly a decomposition cost, not a sampling or MILP cost.

All three configurations reach the same certified final objectives as the controls, with the same final contact counts and zero conflicts. The mathematical reduction passed the original exhaustive tests and independent witness checks on both new inputs. These observations support the safety of dominance for this model, but do not establish uniformly better performance for the bounded pipeline. Sampling is classical simulated annealing. In all three dominance cases, the sampled pre-MILP objective equals the classical tree reference objective; more sampled terminals should not be interpreted as an objective improvement caused solely by sampling.

I would retain the dominance variant as a separate experiment rather than replace the official campaign at this point. It has a strong benefit at 100/50 and a substantial quality/provenance benefit at 200/50, but the 500/50 result exposes an unfavorable interaction with the current separator search. Further work should target that decomposition workload and the baseline's classical evaluation cost before rerunning the remaining campaign. Runtime ratios describe single observed executions; they are not replicated performance estimates.

The initial reporting-only error was a zero count exported as `0.0`. The corrected report was generated by `recover_satellite_scaling_report.py`, which converts this numeric CSV count through `float` before `int` in an execution copy of the driver. The recorded driver, source notebooks, inputs, control results, reused dominance baseline and official figures/analysis remain unchanged. No notebook or solver was rerun for reporting recovery.
