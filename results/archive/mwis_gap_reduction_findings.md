# MWIS gap-reduction findings

This note preserves the conclusions from the isolated MWIS gap-reduction tests. The exploratory scripts and their generated outputs were removed to keep the project tidy; the production behavior and frozen experiment results are documented separately in the MWIS notebooks and run archive.

## What the tests established

- **Triangle-strengthened LP:** Adding valid triangle inequalities tightened the classical LP upper bound without changing the MWIS model or the feasible schedule. On the 500-satellite/50-gateway case, it reduced the reported global gap from 39,562.984 to 15,580.631 (a 60.6% reduction), but did not prove the integer optimum. Tests on the other saved experiment profiles also tightened bounds, with gap reductions varying by case. This was useful as a bound, but the remaining gap could still be substantial.
- **Time-limited triangle-strengthened MILP:** Solving the unresolved natural components with binary variables sometimes closed the remaining gap and certified their optima. With a 5-second solver limit per component and a 30-second total budget, the tests proved all unresolved components in the saved 200/50, 500/25, 500/50, 500/100, and QPU-size profiles. The 500/100 unresolved components took 2.233 seconds of total solver time; the slowest took 2.162 seconds. Proofs were accepted only when the solver reported optimality, and reconstructed schedules were checked against the full conflict graph.
- **Annealing comparison:** In the 500/25 case, annealing scored 20,938.970 on the separator-incomplete component; the MILP proved its optimum was 21,747.841 in 0.162 seconds. The complete annealing schedule was short of the certified full objective by 808.871. Replacing only that unresolved component with its MILP solution produced a conflict-free schedule with objective 37,166.186.

## Why this approach was selected

The LP is a relatively light way to improve an upper bound, but a tightened LP bound is not an exact solution. The MILP can certify an unresolved component when it finishes with an optimal status, while a strict time budget prevents it from becoming an unbounded step. Tests showed that the unresolved components in these saved runs were tractable within the proposed limits, so the project adopted MILP as a narrowly scoped fallback rather than applying it to every component.

## Current pipeline behavior

1. Natural conflict-graph components are processed using the existing exact-separator and annealing workflow.
2. Only a natural component left unresolved because separator decomposition is incomplete is sent to the bounded classical MILP.
3. A MILP-proven optimum replaces that component's prior contribution. If the MILP times out or does not prove optimality, a validated better incumbent may improve the feasible schedule; a bounded triangle-LP solve may tighten its upper bound. If neither improves the result, the previous conservative bound remains.
4. The full reconstructed schedule is checked for conflicts. Results identify annealed and classical contributions separately; a MILP result is not described as an annealing success.

The configured starting limits are 5 solver seconds per component and 30 seconds total per run for MILP, with the same per-component and total caps for triangle-LP fallback solves. These are solver-time limits; overall notebook runtime also includes graph preparation and other pipeline work. All tests described here were classical computations. No physical QPU execution or quantum advantage was established.
