"""Broken domain walls in the QPU reads of this run, by terminal size and by satellite domain length.

A satellite's domain wall (one bit per slot, slot selected when bit - next_bit == 1) is broken when
the bit string rises somewhere (bit 0 followed by next_bit 1), i.e. it is not a single wall.
Counts are weighted by num_occurrences, so each read counts once per satellite.
Writes domain_wall_breaks_per_satellite.csv next to this script.
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUN = HERE.parent                                            # results/runs/csp/qubo_annealing/s400_g50_qpu
PRODUCER = RUN.parents[1] / "pipeline_results" / "csp_test1" / "s400_g50"

qubos = {t["terminal_id"]: t for t in json.loads((PRODUCER / "csp_terminal_qubos.json").read_text())}
subs = {t["terminal_id"]: t for t in json.loads((PRODUCER / "csp_terminal_subproblems.json").read_text())}
with gzip.open(RUN / "annealing_samplesets.json.gz", "rt", encoding="utf-8") as handle:
    samplesets = json.load(handle)

rows = []
for terminal_id, sampleset in samplesets.items():
    qubo = qubos[terminal_id]
    column = {int(v): i for i, v in enumerate(sampleset["variables"])}
    samples = np.array(sampleset["record"]["sample"])
    occurrences = np.array(sampleset["record"]["num_occurrences"])
    for satellite, domain in subs[terminal_id]["domains"].items():
        bits = [qubo["index_map"][str(s)] for s in domain if str(s) in qubo["index_map"]]
        if not bits:
            continue
        rises = np.zeros(len(samples), dtype=bool)
        walls = np.zeros(len(samples), dtype=int)
        for bit, next_bit in bits:
            b = samples[:, column[bit]]
            n = samples[:, column[next_bit]] if next_bit is not None else 0
            rises |= (b == 0) & (n == 1)
            walls += (b - n == 1)
        rows.append({"terminal_id": terminal_id, "terminal_qubits": qubo["qubits"], "satellite": satellite,
                     "slots": len(bits), "reads": int(occurrences.sum()), "broken": int(occurrences[rises].sum()),
                     "two_plus_slots": int(occurrences[walls >= 2].sum())})

table = pd.DataFrame(rows)
table.to_csv(HERE / "domain_wall_breaks_per_satellite.csv", index=False)
for column, bins, labels in [("terminal_qubits", [0, 30, 100, 150, 250], ["<=30", "31-100", "101-150", "151-250"]),
                             ("slots", [0, 5, 15, 30, 60], ["<=5", "6-15", "16-30", "31-60"])]:
    groups = table.groupby(pd.cut(table[column], bins, labels=labels), observed=True)[["reads", "broken", "two_plus_slots"]].sum()
    groups["broken_%"] = (100 * groups.broken / groups.reads).round(2)
    print(groups.to_string(), "\n")
print(f"total: {table.reads.sum()} satellite encodings, {table.broken.sum()} broken "
      f"({100 * table.broken.sum() / table.reads.sum():.2f} %)")
