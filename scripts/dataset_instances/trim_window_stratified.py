"""Trim a Layer C visibility-window CSV to N satellites and M ground stations, stratifying the stations by region.

The stratified twin of trim_window.py. trim_window.py takes the stations in one (seeded) random
order, so a sample can over- or under-represent whole parts of the world by chance. Here every
station of the input is assigned to a geographic region (REGIONS below: the US split into Census
regions plus Alaska, the rest of the world by continent), and the stations are taken in an order
that keeps the regions in proportion:
    1. --min-per-region rounds (default 1) take one station from every region, largest region
       first, so each region is represented (when M is at least the number of regions);
    2. then the next station always comes from the region furthest below its share of the input
       (share * stations taken so far - stations taken from it).
Within a region the stations are taken in station-id order, or with --seed in a random order
fixed by the seed and the region. Because the selection is a prefix of one order, for the same
input, seed and options smaller trims are nested in larger ones (g25 inside g50 inside g100).
The shares are those of the stations with at least one usable window in the input.

Satellites are then selected as in trim_window.py (same id order or seeded shuffle, among the
satellites seen by the kept stations), and a kept station left without windows on those
satellites is replaced by the next station of the stratified order (normally one of its region).
Windows shorter than the slot length are dropped first, as in trim_window.py.

Regions come from --gateways (country, and for US stations the state at the end of the station
name, "City, ST"; a US station without a state takes the region of the nearest one with a state).
--no-us-split keeps the US (without Alaska and the Pacific territories) as one region. An unknown
country stops the script: add it to REGIONS.

Output: the trimmed window CSV (default <input>_s<N>_g<M>_strat[_seed<seed>].csv next to the
input). The allocation per region is printed.

Run from the project root, for example:
    python scripts/dataset_instances/trim_window_stratified.py data/dataset/starlink/windows/starlink_visibility_windows_elevdeg60.csv --satellites 500 --stations 50 --seed 1234
then
    python scripts/data_ingestion/slots.py --windows <trimmed windows CSV> --out-dir data/my_inputs
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trim_dataset import selection_order
from trim_window import (DEFAULT_CONFIG_PATH, SELECTION_COLUMNS, horizon_summary, read_in_chunks, slot_length,
                         update_stats)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATEWAYS = PROJECT_ROOT / "data" / "dataset" / "starlink" / "intermediate" / "starlink_gateways_clean.csv"

# US states and territories (suffix of the station name) -> region: the four Census regions, plus
# Alaska and the Pacific / Caribbean territories, which are far from the contiguous US.
US_STATE_REGION = {
    **dict.fromkeys(["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"], "US Northeast"),
    **dict.fromkeys(["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"], "US Midwest"),
    **dict.fromkeys(["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV", "AL", "KY", "MS", "TN", "AR", "LA",
                     "OK", "TX"], "US South"),
    **dict.fromkeys(["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "CA", "OR", "WA"], "US West"),
    "AK": "Alaska",
    **dict.fromkeys(["HI", "GU", "AS", "MP"], "Pacific islands"),
    **dict.fromkeys(["PR", "VI"], "Central America & Caribbean"),
}
US_CONTIGUOUS = {"US Northeast", "US Midwest", "US South", "US West"}
# Country (as in the gateway table) -> region, for every country but the US.
REGIONS = {
    "Canada": "Canada",
    **dict.fromkeys(["Mexico", "Dominican Republic", "Curaçao", "Puerto Rico", "Guatemala", "Costa Rica",
                     "Panama", "Jamaica", "Haiti", "Bahamas"], "Central America & Caribbean"),
    **dict.fromkeys(["Brazil", "Chile", "Argentina", "Colombia", "Peru", "Ecuador", "Uruguay", "Paraguay",
                     "Bolivia", "Venezuela"], "South America"),
    **dict.fromkeys(["United Kingdom", "Ireland", "France", "Germany", "Italy", "Spain", "Portugal", "Poland",
                     "Lithuania", "Latvia", "Estonia", "Norway", "Sweden", "Finland", "Denmark", "Netherlands",
                     "Belgium", "Austria", "Switzerland", "Czech Republic", "Greece", "Romania", "Iceland"],
                    "Europe"),
    **dict.fromkeys(["Turkey", "Oman", "Nigeria", "Kenya", "South Africa", "Ghana", "Mozambique", "Rwanda",
                     "Qatar", "United Arab Emirates", "Saudi Arabia", "Israel", "Jordan"],
                    "Middle East & Africa"),
    **dict.fromkeys(["Japan", "Philippines", "Indonesia", "Malaysia", "Singapore", "South Korea", "Taiwan",
                     "Mongolia", "India", "Vietnam"], "Asia"),
    **dict.fromkeys(["Australia", "New Zealand"], "Oceania"),
    **dict.fromkeys(["Fiji", "Tonga", "Samoa", "Vanuatu"], "Pacific islands"),
}


# ---------------------------------------------------------------- regions

def station_regions(gateways: pd.DataFrame, us_split: bool = True) -> pd.Series:
    """Region of every station of the gateway table (index station_id)."""
    region = gateways["country"].map(REGIONS)
    us = gateways["country"].isin(["US", "USA", "United States"])
    state = gateways["station_name"].str.extract(r",\s*([A-Z]{2})\s*$")[0]
    region[us] = state[us].map(US_STATE_REGION)
    # US stations without a (known) state: region of the nearest US station with one
    missing = us & region.isna()
    known = us & region.notna()
    for index in gateways.index[missing]:
        dist = np.hypot(gateways.loc[known, "latitude"] - gateways.at[index, "latitude"],
                        (gateways.loc[known, "longitude"] - gateways.at[index, "longitude"])
                        * np.cos(np.radians(gateways.at[index, "latitude"])))
        region[index] = region[dist.idxmin()]
    unknown = sorted(gateways.loc[region.isna(), "country"].astype(str).unique())
    if unknown:
        raise ValueError(f"no region for the countries {unknown}: add them to REGIONS in {Path(__file__).name}")
    if not us_split:
        region[region.isin(US_CONTIGUOUS)] = "US contiguous"
    return pd.Series(region.to_numpy(), index=gateways["station_id"], name="region")


def stratified_order(region_of: pd.Series, seed: int | None, min_per_region: int = 1,
                     skip: set | None = None) -> list:
    """Stations (index of `region_of`) in an order whose every prefix keeps the regions in proportion
    to their share of `region_of`: first `min_per_region` rounds over the regions (largest first),
    then always the region furthest below its share. Stations in `skip` are left out (the shares
    still count them). Within a region: id order, or a shuffle fixed by the seed and the region."""
    skip = skip or set()
    groups = {region: sorted(ids) for region, ids in region_of.groupby(region_of).groups.items()}
    total = len(region_of)
    share = {region: len(ids) / total for region, ids in groups.items()}
    queues = {region: [s for s in selection_order(ids, seed, f"stations-{region}") if s not in skip]
              for region, ids in groups.items()}
    regions = sorted(groups, key=lambda region: (-share[region], region))  # largest first: tie-break order
    taken = dict.fromkeys(regions, 0)
    order = []

    def take(region):
        order.append(queues[region][taken[region]])
        taken[region] += 1

    for _ in range(min_per_region):
        for region in regions:
            if taken[region] < len(queues[region]):
                take(region)
    remaining = sum(len(queue) for queue in queues.values())
    while len(order) < remaining:
        n = len(order) + 1
        open_regions = [region for region in regions if taken[region] < len(queues[region])]
        # max() keeps the first of equal keys, i.e. the larger region
        take(max(open_regions, key=lambda region: (round(share[region] * n - taken[region], 9), share[region])))
    return order


# ---------------------------------------------------------------- selection

def trim_stratified(rows: pd.DataFrame, satellites: int, stations: int, region_of: pd.Series,
                    seed: int | None, min_per_region: int) -> tuple[pd.DataFrame, list]:
    """Keep the first `stations` stations of the stratified order and the first `satellites` satellites
    (id order or seeded shuffle) seen by them; a kept station left without rows is replaced by the next
    one of the order. Returns the kept rows and the kept stations in order."""
    satellite_order = selection_order(rows["norad_id"], seed, "satellites")
    available = region_of.loc[sorted(set(rows["station_id"]))]
    excluded: set[str] = set()
    while True:
        kept_stations = stratified_order(available, seed, min_per_region, skip=excluded)[:stations]
        if len(kept_stations) < stations:
            raise ValueError(f"only {len(kept_stations)} stations with windows are available, {stations} requested")
        on_stations = rows[rows["station_id"].isin(kept_stations)]
        present = set(on_stations["norad_id"])
        kept_satellites = [satellite for satellite in satellite_order if satellite in present][:satellites]
        if len(kept_satellites) < satellites:
            raise ValueError(f"only {len(kept_satellites)} satellites have windows on the selected {stations} "
                             f"stations, {satellites} requested")
        trimmed = on_stations[on_stations["norad_id"].isin(kept_satellites)]
        empty = set(kept_stations) - set(trimmed["station_id"])
        if not empty:
            return trimmed, kept_stations
        excluded |= empty


def allocation_table(available: pd.Series, kept: list) -> pd.DataFrame:
    """Per region: stations available in the input, their share, the proportional target and the kept count."""
    table = available.value_counts().rename("available").to_frame()
    table["share_%"] = 100 * table["available"] / table["available"].sum()
    table["proportional"] = table["available"] / table["available"].sum() * len(kept)
    table["kept"] = available.loc[kept].value_counts().reindex(table.index, fill_value=0)
    return table


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Layer C visibility-window CSV")
    parser.add_argument("--satellites", type=int, required=True, help="satellites in the output")
    parser.add_argument("--stations", type=int, required=True, help="ground stations in the output")
    parser.add_argument("--seed", type=int, default=None,
                        help="random order of the stations within each region and of the satellites, fixed by "
                             "this seed (default: id order)")
    parser.add_argument("--min-per-region", type=int, default=1,
                        help="stations guaranteed to every region before the proportional allocation "
                             "(default: %(default)s; 0 for a purely proportional sample)")
    parser.add_argument("--no-us-split", action="store_true",
                        help="keep the contiguous US as one region instead of its four Census regions")
    parser.add_argument("--gateways", type=Path, default=DEFAULT_GATEWAYS,
                        help="gateway table with station_id, station_name, country, latitude, longitude "
                             "(default: %(default)s)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="data-ingestion config giving the slot length (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=None,
                        help="output CSV (default: <input>_s<satellites>_g<stations>_strat[_seed<seed>].csv "
                             "next to the input)")
    args = parser.parse_args()
    if args.satellites <= 0 or args.stations <= 0:
        parser.error("--satellites and --stations must be positive")
    if args.min_per_region < 0:
        parser.error("--min-per-region must be >= 0")
    for path, flag in ((args.input, "input"), (args.gateways, "--gateways"), (args.config, "--config")):
        if not path.is_file():
            parser.error(f"{flag} file not found: {path}")
    sys.stdout.reconfigure(encoding="utf-8")  # station and country names with accents

    min_seconds = slot_length(args.config)
    gateways = pd.read_csv(args.gateways, encoding="utf-8")
    region_of = station_regions(gateways, us_split=not args.no_us_split)

    # selection on the light columns only
    light = pd.concat(read_in_chunks(args.input, "1/2 reading ids", usecols=SELECTION_COLUMNS),
                      ignore_index=True)
    usable = light[light["duration_seconds"] >= min_seconds]
    unknown = set(usable["station_id"]) - set(region_of.index)
    if unknown:
        parser.error(f"{len(unknown)} stations of the input are not in {args.gateways}, e.g. {sorted(unknown)[:5]}")
    print(f"selecting {args.satellites} satellites and {args.stations} stations (stratified by region) ...",
          file=sys.stderr, flush=True)
    started = time.monotonic()
    selected, kept_stations = trim_stratified(usable, args.satellites, args.stations, region_of, args.seed,
                                              args.min_per_region)
    print(f"selection done in {time.monotonic() - started:.0f} s", file=sys.stderr, flush=True)
    satellites, stations = set(selected["norad_id"]), set(selected["station_id"])
    available = region_of.loc[sorted(set(usable["station_id"]))]
    input_counts = (len(light), light["norad_id"].nunique(), light["station_id"].nunique())
    dropped_short = len(light) - len(usable)
    del light, usable, selected

    # full rows of the selection, filtered chunk by chunk
    seed_suffix = "" if args.seed is None else f"_seed{args.seed}"
    output = args.output or args.input.with_name(
        f"{args.input.stem}_s{args.satellites}_g{args.stations}_strat{seed_suffix}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    input_stats: dict = {}
    output_stats: dict = {}
    kept = 0
    for index, chunk in enumerate(read_in_chunks(args.input, "2/2 writing   ")):
        update_stats(input_stats, chunk)
        part = chunk[chunk["norad_id"].isin(satellites) & chunk["station_id"].isin(stations)
                     & (chunk["duration_seconds"] >= min_seconds)]
        update_stats(output_stats, part)
        part.to_csv(output, index=False, mode="w" if index == 0 else "a", header=index == 0)
        kept += len(part)

    order = "id order" if args.seed is None else f"random order, seed {args.seed}"
    print(f"input:  {input_counts[0]} windows, {input_counts[1]} satellites, {input_counts[2]} stations "
          f"({dropped_short} windows shorter than the {min_seconds} s slot ignored)")
    print(f"output: {kept} windows, {len(satellites)} satellites, {len(stations)} stations "
          f"(stratified by region, at least {args.min_per_region} per region, {order}) -> {output}")
    print(allocation_table(available, kept_stations).to_string(float_format=lambda x: f"{x:.1f}"))
    print(f"input  {horizon_summary(input_stats)}")
    print(f"output {horizon_summary(output_stats)}")
    if output_stats != input_stats:
        print("WARNING: longest window or horizon differ from the input: Layer E weights on this output "
              "will differ slightly from a full-table run")


if __name__ == "__main__":
    main()
