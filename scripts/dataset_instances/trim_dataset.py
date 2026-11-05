"""Trim a Layer E candidate-slot CSV to exactly N satellites and M ground stations.

Without --seed, satellites and stations are taken in the Layer C order (satellites
by NORAD id, stations by station id). With --seed, both are taken in a random
order fixed by the seed (one shuffle for the stations, an independent one for the
satellites), giving a random sample of the constellation and of the stations
instead of the lowest ids. Either way only satellites and stations with at least
one candidate slot are kept, and for the same order (no seed, or the same seed)
smaller trims of the same input are nested in larger ones. Slots and weights are
copied unchanged.

Run from the project root, for example:
    python scripts/trim_dataset.py data/dataset/starlink/starlink_scheduling_slots_weighted.csv --satellites 200 --stations 50
    python scripts/trim_dataset.py input.csv --satellites 100 --stations 25 --output data/my_inputs/slots_s100_g25.csv
    python scripts/trim_dataset.py input.csv --satellites 200 --stations 50 --seed 1
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def selection_order(values, seed: int | None, stream: str) -> list:
    """Order in which candidates are taken: sorted by id, or a seeded shuffle of that list.
    Each stream ("stations", "satellites") has its own generator, so the satellite order
    does not depend on how many stations the input has."""
    order = sorted(set(values))
    if seed is not None:
        random.Random(f"{seed}-{stream}").shuffle(order)
    return order


def trim(slots: pd.DataFrame, satellites: int, stations: int, seed: int | None = None) -> pd.DataFrame:
    """Keep the first `stations` stations and `satellites` satellites that have slots, in id
    order or, with a seed, in a random order fixed by it. A station left without slots once
    the satellites are cut is replaced by the next one."""
    station_order = selection_order(slots["station_id"], seed, "stations")
    satellite_order = selection_order(slots["norad_id"], seed, "satellites")
    excluded: set[str] = set()
    while True:
        kept_stations = [station for station in station_order if station not in excluded][:stations]
        if len(kept_stations) < stations:
            raise ValueError(f"only {len(kept_stations)} stations with slots are available, {stations} requested")
        on_stations = slots[slots["station_id"].isin(kept_stations)]
        present = set(on_stations["norad_id"])
        kept_satellites = [satellite for satellite in satellite_order if satellite in present][:satellites]
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
    parser.add_argument("--seed", type=int, default=None,
                        help="sample satellites and stations in a random order fixed by this seed "
                             "(default: lowest NORAD ids and station ids)")
    parser.add_argument("--output", type=Path, default=None,
                        help="output CSV (default: <input>_s<satellites>_g<stations>[_seed<seed>].csv next to the input)")
    args = parser.parse_args()
    if args.satellites <= 0 or args.stations <= 0:
        parser.error("--satellites and --stations must be positive")

    slots = pd.read_csv(args.input)
    trimmed = trim(slots, args.satellites, args.stations, args.seed)
    seed_suffix = "" if args.seed is None else f"_seed{args.seed}"
    output = args.output or args.input.with_name(
        f"{args.input.stem}_s{args.satellites}_g{args.stations}{seed_suffix}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    trimmed.to_csv(output, index=False)

    print(f"input:  {len(slots)} slots, {slots['norad_id'].nunique()} satellites, {slots['station_id'].nunique()} stations")
    print(f"output: {len(trimmed)} slots, {trimmed['norad_id'].nunique()} satellites, "
          f"{trimmed['station_id'].nunique()} stations "
          f"({'id order' if args.seed is None else f'random order, seed {args.seed}'}) -> {output}")


if __name__ == "__main__":
    main()
