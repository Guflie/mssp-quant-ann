# Input snapshot for Experiments 1–3

Captured by running Layers A and B with the Python 3.13.14 Jupyter kernel on 2026-09-27.

## Files and contents

- `starlink_celestrak_latest.tle`: cached Starlink TLE source used by Layer A.
- `starlink_satellite_subset.csv`: 1,000 unique satellite records; TLE epochs range from `2026-06-16T17:24:16Z` to `2026-06-18T08:00:02Z`.
- `starlink_gateways_raw.json`: 303 source rows fetched by Layer B.
- `starlink_gateways_clean.csv`: 302 validated gateway records.

Layer A attempted to refresh CelesTrak, but received HTTP 403 and used the existing local TLE cache. The TLE file was therefore not refreshed during this capture. Layer B fetched its gateway source successfully.

## Fixed propagation horizon

Use `2026-06-18T00:00:00Z` as the start of the four-hour horizon for every configuration. This places propagation close to the TLE epoch range and replaces the previous May 31 start. Keep all other visibility and scheduling settings fixed across the experiments.

For reproducibility, use these snapshot copies as the Layer C inputs throughout the experiment campaign. Do not rerun Layers A or B between configurations.

## Final scenario inputs

The canonical weighted candidate-slot CSVs are in `scenario_inputs/`. All five use a **four-minute minimum contact duration** and a **two-minute candidate-start step**. These are the final inputs for Experiments 1–3; use only these files for the experiment runs.

`scenario_inputs_manifest.json` records the shared settings, candidate counts, file hashes, and MWIS graph preflight metrics for each configuration. Verify the SHA-256 value before a run if the files are copied or moved.

The five configurations are:

- `slots_s100_g50.csv` — Experiment 1 baseline and Experiment 2 reference.
- `slots_s200_g50.csv` — Experiment 2 satellite scaling.
- `slots_s500_g50.csv` — shared 500-satellite case for Experiments 2 and 3.
- `slots_s500_g25.csv` — Experiment 3 gateway scaling, 25 gateways.
- `slots_s500_g100.csv` — Experiment 3 gateway scaling, 100 gateways.

The default MWIS input in `results/_config.json` points to the 100/50 baseline. For another configuration, change only `candidate_slot_csv` and give the run a distinct `run_name_mwis`; keep the shared experiment settings unchanged.
