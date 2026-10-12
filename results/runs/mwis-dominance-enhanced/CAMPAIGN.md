# Completed dominance-enhanced MWIS campaign

This folder preserves the separately evaluated dominance variant. The original
reordered MWIS campaign remains in `results/runs/mwis/*/new_pipeline_results_reordered`.
Adding these files does not change the default MWIS notebook or adopt dominance
as the default pipeline.

## Implementation and experiment settings

The producer is `results/02.A_mwis_pipeline_dominance_enhanced.ipynb`. It applies
exact same-satellite dominance before connected-component extraction and during
recursive preprocessing. A removed alternative has a surviving alternative of
at least its weight and no additional external conflicts; same-satellite
alternatives form a clique. Removal witnesses and the original and reduced
graphs are saved for independent validation.

The canonical `results/3_qubo_annealing.ipynb` is shared with the original
pipeline. Executors change only its output route in memory. Per-run settings
come from `execution_logs/<run>/config.json` through `SWE2_CONFIG`; the global
configuration is not changed. Sampling uses classical simulated annealing,
500 reads per terminal and seed 42. Logical terminal limits are not physical
embedding measurements. MILP limits remain 5 seconds per component and 30 seconds
in total. Python 3.13.14 and package versions are recorded with each execution.

All inputs are the frozen candidate CSVs in
`data/experiment_inputs/exp_snapshot_2026-10-12`. Resource identities are not fully
nested across scaling scenarios. Terminal sensitivity uses the identical 200/50
CSV throughout.

## Completed runs

| Run | Active satellites / gateways | Logical variables / couplers |
|---|---:|---:|
| `baseline_s200_g50` | 200 / 50 | 30 / 200 |
| `exp2_s100_g50` | 100 / 50 | 30 / 200 |
| `exp2_s500_g50` | 500 / 50 | 30 / 200 |
| `exp3_s200_g25` | 200 / 25 | 30 / 200 |
| `exp3_s200_g100` | 200 / 100 | 30 / 200 |
| `exp4_s200_g50_n20_c100` | 200 / 50 | 20 / 100 |
| `exp4_s200_g50_n50_c350` | 200 / 50 | 50 / 350 |

The baseline is reused across experiment groups and is not an independent
replicate. Every run has producer outputs, annealing/reconstruction outputs,
executed notebooks, settings, timings and an independent validation report.
`dominance_test_report.json` records exhaustive small-instance safety tests.
Generated IPython runtime folders and Python caches are excluded from Git.

The original `README.md` and baseline report describe the first test. The full
assessment is [adoption_assessment.md](experiment_comparison/campaign_assessment/adoption_assessment.md).
All seven final schedules are feasible and certified optimal for the finite
candidate model. Dominance improves pre-MILP quality in all seven comparisons,
but observed runtime improves in only three. This is a completed experiment,
not a claim of universal speedup or quantum advantage.

## Figures and analysis regeneration

From the repository root, with the recorded project environment:

```powershell
python -B -X utf8 results/5_csp_mwis_comparison_plots.py --mwis-variant dominance --output-dir results/figures/csp_mwis_reordered_dominance
```

This reads saved dominance and CSP results and writes plots and analysis tables;
it does not run an experiment. The figures folder contains 26 PNG/PDF pairs,
an HTML gallery, metric tables, captions and validation/source manifests. Without
`--mwis-variant dominance`, the plotting script continues to use the original
reordered MWIS campaign. CSP source bundles must be extracted at the paths
recorded in the figure source manifest.

`build_campaign_assessment.py` rebuilds the original-versus-dominance assessment
from the saved runs. Historical campaign runners and setup scripts are retained
for provenance; do not rerun setup or experiment preparation over these completed
folders. In particular, `setup_variant.py` writes baseline configuration and
setup files. The satellite report's numeric-parser recovery is documented in
`recover_satellite_scaling_report.py` and its saved campaign metadata; it did not
rerun any experiment.

Manifests record hashes of historical working-tree bytes. Some preservation
manifests reference local files outside this committed campaign, and Git newline
conversion on another operating system can affect raw hashes. They are historical
audit records; keep them unchanged. The restored producer's original SHA-256 is
`edd571cffa305acff64b47208570a8b04026258552cd8cf0d65ffdd6b1577a6d`.
