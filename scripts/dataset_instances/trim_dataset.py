"""Trim a Layer E candidate-slot CSV to exactly N satellites and M ground stations.

Satellites and stations are taken in the Layer C order (satellites by NORAD id,
stations by station id), keeping only those with at least one candidate slot,
so smaller trims of the same input are nested in larger ones. Slots and weights
are copied unchanged.

Run from the project root, for example:
    python scripts/dataset_instances/trim_dataset.py data/dataset/starlink/starlink_scheduling_slots.csv --satellites 200 --stations 50
    python scripts/dataset_instances/trim_dataset.py input.csv --satellites 100 --stations 25 --output data/my_inputs/slots_s100_g25.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def trim(slots: pd.DataFrame, satellites: int, stations: int) -> pd.DataFrame:
    """Keep the first `stations` stations and `satellites` satellites that have slots.
    A station left without slots once the satellites are cut is replaced by the next one."""
    excluded: set[str] = set()
    while True:
        kept_stations = sorted(set(slots["station_id"]) - excluded)[:stations]
        if len(kept_stations) < stations:
            raise ValueError(f"only {len(kept_stations)} stations with slots are available, {stations} requested")
        on_stations = slots[slots["station_id"].isin(kept_stations)]
        kept_satellites = sorted(set(on_stations["norad_id"]))[:satellites]
        if len(kept_satellites) < satellites:
            raise ValueError(f"only {len(kept_satellites)} satellites have slots on the first {stations} stations, "
                             f"{satellites} requested: generate the input from a larger pool")
        trimmed = on_stations[on_stations["norad_id"].isin(kept_satellites)]
        empty = set(kept_stations) - set(trimmed["station_id"])
        if not empty:
            return trimmed.reset_index(drop=True)
        excluded |= empty


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Layer E candidate-slot CSV")
    parser.add_argument("--satellites", type=int, required=True, help="satellites in the output")
    parser.add_argument("--stations", type=int, required=True, help="ground stations in the output")
    parser.add_argument("--output", type=Path, default=None,
                        help="output CSV (default: <input>_s<satellites>_g<stations>.csv next to the input)")
    args = parser.parse_args()
    if args.satellites <= 0 or args.stations <= 0:
        parser.error("--satellites and --stations must be positive")

    slots = pd.read_csv(args.input)
    trimmed = trim(slots, args.satellites, args.stations)
    output = args.output or args.input.with_name(f"{args.input.stem}_s{args.satellites}_g{args.stations}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    trimmed.to_csv(output, index=False)

    print(f"input:  {len(slots)} slots, {slots['norad_id'].nunique()} satellites, {slots['station_id'].nunique()} stations")
    print(f"output: {len(trimmed)} slots, {trimmed['norad_id'].nunique()} satellites, "
          f"{trimmed['station_id'].nunique()} stations -> {output}")


if __name__ == "__main__":
    main()
