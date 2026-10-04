# Archived MWIS Experiments — Previous Model

This folder preserves the frozen results from the MWIS model used before same-satellite overlapping contacts were prohibited. It is kept for historical reference and is not the current result set for the paper. The revised model's final frozen results will be kept separately under `results/frozen/mwis_experiments`.

## Layout

- `exp1_baseline/` — baseline input, MWIS output, and QUBO/annealing/reconstruction output.
- `exp2_satellite_scaling/` — 200/50 and 500/50 satellite scenarios, each with its input and both run outputs.
- `exp3_gateway_scaling/` — 500/25 and 500/100 gateway scenarios, each with its input and both run outputs.
- `exp4_qpu_size_sensitivity/` — shared 500/50 input, three QPU-size profiles, their run outputs, and Experiment 4 comparison table/figure.
- `comparison_experiments_1_3/` — the table and figures comparing Experiments 1–3.
- `notebooks/` — frozen MWIS, annealing/reconstruction, and comparison notebook sources. See `notebooks/README.md` for how these relate to the active notebooks.
- `manifest.json` — run settings, result summaries, environment and validation notes, and SHA-256 checksums for archived files.

This is the single retained complete snapshot for the MWIS model before satellite-overlap conflicts were added. Older duplicate snapshots and exploratory trials were removed during project cleanup.

## Reading the results

All eight run pairs passed the saved consistency checks: reconstructed schedules are conflict-free, objective values agree with selected weights, and annealing output covers the terminal QUBOs used by reconstruction. The 500/100 gateway case has a 1.1126% conservative gap. In Experiment 4, the 20-node/100-coupler profile has a 2.7018% conservative gap; the 30/200 and 50/350 profiles report zero gap. The two nonzero-gap cases include heuristic fallback and are not certified optimal.

The annealing results use D-Wave's classical `SimulatedAnnealingSampler`; they are not results from physical quantum hardware.
