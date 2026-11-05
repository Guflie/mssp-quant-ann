"""Dataset variant with aggregated US ground stations plus every station outside the US.

Same US aggregation as dataset_us_clusters.py (whose functions it uses): the US stations are
clustered by complete linkage within --diameter-km, every cluster of at least --min-stations
stations becomes one station with one antenna per member, seeing the sky of its representative
(the member closest to the centroid). The US stations outside those clusters are dropped, and
every non-US station is kept unchanged, with its single antenna and its own slots.

US and non-US are told apart by the country column of the Layer B gateway table (the rule of
gateways.infer_country, as in dataset_no_USA.py); every station of the slot CSV must be in it.

Slots of the non-US stations are copied as text, unchanged; slot copies of the aggregated stations
get the id <slot_id>_A<k> and the antenna ANT_<cluster id>_<k> (see dataset_us_clusters.py).
A gateway table of the kept stations (<output stem>_gateways.csv: the aggregated ones placed at
their representative, then the non-US ones with 1 antenna) is written next to the slots, for
results/_graphics/station_map.py --gateways.

The input is read in chunks, so the full CSV never has to fit in memory. The output goes to
data/test/intermediate/ by default: it is the source the test instances in data/test/ are cut
from, with scripts/dataset_instances/trim_dataset.py.

Run from anywhere, for example:
    python scripts/dataset_instances/dataset_us_clusters_intl.py
    python scripts/dataset_instances/dataset_us_clusters_intl.py --diameter-km 100 --min-stations 3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_us_clusters import (  # noqa: E402
    CHUNK_ROWS,
    CLUSTERED_COUNTRY,
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    antenna_copies,
    cluster_stations,
    parse_args,
    stations_with_slots,
)

OUTPUT_SUFFIX = "_us_clusters_intl"


def write_slots(input_path: Path, output_path: Path, clusters: pd.DataFrame, non_us: set[str],
                chunk_rows: int = CHUNK_ROWS) -> dict[str, int]:
    """Write the slots of the `non_us` stations unchanged and the antenna copies of the cluster
    representatives' slots to `output_path`. Returns the slots read and written per kind."""
    report = {"slots_in": 0, "non_us_slots": 0, "cluster_slots": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first = True
    # every column as text, so the kept values are written back exactly as they were read
    for chunk in pd.read_csv(input_path, dtype=str, keep_default_na=False, chunksize=chunk_rows):
        report["slots_in"] += len(chunk)
        parts = [chunk[chunk["station_id"].isin(non_us)], *antenna_copies(chunk, clusters)]
        report["non_us_slots"] += len(parts[0])
        report["cluster_slots"] += sum(len(part) for part in parts[1:])
        out = pd.concat(parts)
        if len(out):
            out.to_csv(output_path, mode="w" if first else "a", header=first, index=False)
            first = False
        print(f"  read {report['slots_in']:,} slots, wrote "
              f"{report['non_us_slots'] + report['cluster_slots']:,}", flush=True)
    if first:
        raise ValueError("no slot of a kept station was found in the input")
    return report


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv, description="Aggregate clusters of nearby US ground stations into multi-antenna "
                                        "stations, drop the other US stations and keep every non-US station.",
                      output_suffix=OUTPUT_SUFFIX)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    input_path = args.input or DEFAULT_DATA_DIR / config["slots"]["output_csv_name"]
    gateways_path = args.gateways or DEFAULT_DATA_DIR / "intermediate" / config["gateways"]["output_csv_name"]
    output_path = args.output or DEFAULT_OUT_DIR / f"{input_path.stem}{OUTPUT_SUFFIX}.csv"
    gateways_output = output_path.with_name(f"{output_path.stem}_gateways.csv")
    if output_path.resolve() == input_path.resolve():
        raise SystemExit("The output path must differ from the input path.")

    print(f"Reading the stations with slots in {input_path}")
    with_slots = stations_with_slots(input_path)
    gateways = pd.read_csv(gateways_path, dtype={"station_id": str})
    unknown = sorted(with_slots - set(gateways["station_id"]))
    if unknown:
        raise SystemExit(f"{len(unknown)} stations of the slot CSV are not in {gateways_path}, e.g. {unknown[:5]}")
    gateways = gateways[gateways["station_id"].isin(with_slots)]
    is_us = gateways["country"].eq(CLUSTERED_COUNTRY)
    clusters = cluster_stations(gateways[is_us], args.diameter_km, args.min_stations)
    non_us = gateways[~is_us]
    dropped_us = int(is_us.sum()) - int(clusters["antennas"].sum())
    print(f"{int(is_us.sum())} US stations with slots -> {len(clusters)} clusters of at least {args.min_stations} "
          f"stations within {args.diameter_km:g} km ({int(clusters['antennas'].sum())} stations merged, "
          f"{dropped_us} dropped); {len(non_us)} non-US stations kept")

    print("Writing the non-US slots and the cluster antenna copies")
    report = write_slots(input_path, output_path, clusters, set(non_us["station_id"]))

    station_table = pd.concat([
        clusters.assign(country=CLUSTERED_COUNTRY, status="aggregated", source=gateways_path.name),
        non_us.assign(antennas=1, representative_station_id=non_us["station_id"],
                      representative_station_name=non_us["station_name"],
                      member_station_ids=non_us["station_id"], member_station_names=non_us["station_name"]),
    ], ignore_index=True)
    station_table.to_csv(gateways_output, index=False)

    total = report["non_us_slots"] + report["cluster_slots"]
    print(f"Saved {total:,} slots ({report['non_us_slots']:,} on {len(non_us)} non-US stations, "
          f"{report['cluster_slots']:,} on {len(clusters)} aggregated US stations) to {output_path}")
    print(f"Saved the {len(station_table)} stations to {gateways_output}")
    print(f"Non-US stations per country: {non_us['country'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
