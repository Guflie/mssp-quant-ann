# MWIS Pipeline Changes and Experiment Plan Summary

This document summarizes the recent MWIS-side changes and the proposed experiment plan before running the final tests.

## 1. What Changed

### MWIS pipeline cleanup and reconstruction

The MWIS pipeline now produces enough information to reconstruct a full schedule after annealing.

Main additions:

- `reconstruction_metadata.json` is written by the MWIS pipeline.
- Terminal subgraphs now keep their forced selected nodes from preprocessing.
- The reconstruction notebook can combine terminal QUBO results back into one full MWIS schedule.
- Heuristic fallback is now explicit in the output, even when it is not used.

Current result for the latest run:

- Selected contacts: `113`
- Total feasible objective: `8052.602`
- Infeasible terminal annealing samples: `0`
- Heuristic fallback nodes: `0`

This means the current run was solved through natural components, exact separator decomposition, terminal QUBOs, simulated annealing, and reconstruction. No greedy fallback was needed.

### QUBO penalty correction

The QUBO penalty now scales with the effective reward:

```text
penalty = penalty_factor * (max effective reward + 1)
```

This avoids a bad case where increasing the reward factor could make invalid conflicting selections attractive.

### Baseline parameter update

Only the baseline MWIS experiment values were changed:

- Elevation mask changed from `10 deg` to `25 deg`.
- Blackouts per station changed from `2` to `0`.

Reason:

- `25 deg` is a more defensible main Starlink-like visibility assumption.
- Blackouts are synthetic in our data, so the clean baseline should not include artificial disruptions.
- Blackouts can still be used later as an operational stress test.

## 2. Baseline Experiment

Recommended baseline:

| Parameter | Value |
|---|---:|
| Satellites | `100` |
| Gateways | `50` |
| Horizon | `4 hours` |
| Visibility step | `60 seconds` |
| Elevation mask | `25 deg` |
| Minimum contact duration | `5 minutes` |
| Candidate start step | `2 minutes` |
| Antenna reconfiguration time | `180 seconds` |
| Blackouts per station | `0` |
| QPU terminal node limit | `30` |
| QPU terminal coupler limit | `200` |
| Annealing reads | `500` |

Purpose:

This gives us a clean reference case. It uses real orbital geometry, real gateway positions, operational reconfiguration time, and no artificial blackout disruptions.

## 3. Proposed Experiments

These experiments should still be discussed with the team before implementation.

### Experiment 1: Baseline

Run the baseline configuration above.

Show:

- Number of candidate slots.
- Number of conflict edges.
- Number of natural components.
- Number of terminal QUBOs.
- Number of direct, separator, and heuristic subgraphs.
- Final selected contacts and objective.
- Whether the reconstructed schedule has any conflicts.

### Experiment 2: Satellite scaling

Suggested values:

```text
satellites = 100, 200, 500
gateways = 50
```

Purpose:

Show how constellation size affects graph size, decomposition, QUBO size, and final schedule quality.

### Experiment 3: Gateway scaling

Suggested values:

```text
satellites = 500
gateways = 25, 50, 100
```

Purpose:

Show how ground-resource availability affects conflicts and achievable objective.

### Experiment 4: QPU-size sensitivity

Suggested values:

```text
node/coupler limits = 20/100, 30/200, 50/350
```

Purpose:

Show how hardware limits affect direct QUBO solving, separator decomposition, and fallback usage.


## 4. Standard Output Files

Each final experiment run should produce the same files:

```text
run_config_log.json
terminal_subgraphs.json
reconstruction_metadata.json
final_summary.csv
terminal_qubo_summary.csv
annealing_results.csv
reconstructed_summary.csv
reconstructed_schedule.csv
component_provenance_summary.csv
heuristic_fallback_details.csv
```

This gives us consistent files for tables and plots in the report.

## 5. Recommended CSP Checks Before Final Experiments

No CSP files were changed by the MWIS work. Before comparing CSP and MWIS, the CSP team should check a few things:

1. Use the same input scenario as MWIS.

   CSP and MWIS should use the same satellite count, gateway count, horizon, elevation mask, contact duration, step size, reconfiguration time, and blackout setting.

2. Confirm the CSP formulation matches the comparison goal.

   MWIS currently allows multiple scheduled contacts for the same satellite if they come from different visibility windows and do not conflict. If CSP allows only one contact per satellite, then the two methods are solving different problems.

3. Export comparable output files.

   CSP should produce a final schedule, objective value, conflict/feasibility check, runtime, number of variables, and number of couplers or constraints.

4. Add reconstruction if needed.

   If CSP also decomposes or anneals smaller QUBOs, it should reconstruct the final schedule in the same way MWIS does.

5. Use the same metrics in the report.

   Recommended shared metrics:

   - selected contacts
   - total objective
   - candidate slots or variables
   - conflict edges, constraints, or couplers
   - QUBO variables
   - QUBO couplers
   - number of subproblems
   - feasibility violations
   - runtime
   - exact, annealed, or heuristic provenance

## 6. Main Message for the Report

The MWIS pipeline transforms satellite contact scheduling into a conflict graph. Independent graph components are solved separately. Large components are split with exact separator decomposition when possible. Terminal subgraphs are converted to QUBOs and solved with D-Wave simulated annealing. The final reconstruction combines the terminal results into one feasible schedule and reports whether any heuristic fallback was needed.

The current implementation is exact in the decomposition logic, but the annealing result itself is a solver result rather than a mathematical proof of optimality. The outputs clearly report feasibility, objective value, provenance, and fallback use.
