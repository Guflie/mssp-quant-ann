"""Dataset variant with aggregated US ground stations: every cluster of nearby US stations becomes
one multi-antenna station, and every other station is dropped.

Steps
    1. read the station ids of the full Layer E slot CSV (only stations with slots take part)
    2. cluster the US stations of the Layer B gateway table (country "US", as in gateways.py) by
       complete linkage on the great-circle distance: within a cluster every pair of stations is
       at most --diameter-km apart
    3. keep the clusters of at least --min-stations stations; each becomes one aggregated station
       with one antenna per member station
    4. the aggregated station sees the sky of its representative, the member closest to the
       cluster centroid: every slot of the representative is copied onto each antenna
    5. write the aggregated slots and a gateway table of the aggregated stations

Slot copies get the id <slot_id>_A<k> and the antenna ANT_<cluster id>_<k>; times, weights and
window ids are copied unchanged as text (the copies of a slot are the same visibility window seen
by different antennas). The CSP and MWIS conflicts are per antenna and per satellite, so the
antennas of one station work in parallel and a satellite still gets at most one contact.

The gateway table of the aggregated stations (<output stem>_gateways.csv) places each station at
its representative, so it can be passed to results/_graphics/station_map.py with --gateways. It
also lists the antennas, the member stations and the cluster centroid.

The input is read in chunks, so the full CSV never has to fit in memory. The output goes to
data/test/intermediate/ by default: it is the source the test instances in data/test/ are cut
from, with scripts/dataset_instances/trim_dataset.py.

Run from anywhere, for example:
    python scripts/dataset_instances/dataset_us_clusters.py
    python scripts/dataset_instances/dataset_us_clusters.py --diameter-km 100 --min-stations 3
    python scripts/dataset_instances/dataset_us_clusters.py my/slots.csv --output my/slots_us_clusters.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INGESTION_DIR = PROJECT_ROOT / "scripts" / "data_ingestion"
DEFAULT_CONFIG_PATH = DATA_INGESTION_DIR / "config.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "dataset" / "starlink"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "test" / "intermediate"
OUTPUT_SUFFIX = "_us_clusters"
CLUSTERED_COUNTRY = "US"
EARTH_RADIUS_KM = 6371.0
CHUNK_ROWS = 500_000


def unit_vectors(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Points on the unit sphere, one row per (latitude, longitude) in degrees."""
    lat, lon = np.radians(latitude), np.radians(longitude)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def distance_matrix_km(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Great-circle distances between all pairs of points, in km."""
    points = unit_vectors(latitude, longitude)
    return EARTH_RADIUS_KM * np.arccos(np.clip(points @ points.T, -1.0, 1.0))


def stations_with_slots(slots_path: Path, chunk_rows: int = CHUNK_ROWS) -> set[str]:
    """Station ids that have at least one slot in the slot CSV."""
    stations: set[str] = set()
    for chunk in pd.read_csv(slots_path, usecols=["station_id"], dtype=str, chunksize=chunk_rows):
        stations.update(chunk["station_id"].unique())
    return stations


def cluster_stations(gateways: pd.DataFrame, diameter_km: float, min_stations: int) -> pd.DataFrame:
    """Complete-linkage clusters of `gateways` (station_id, station_name, latitude, longitude)
    with every pairwise distance at most `diameter_km`, keeping those with at least
    `min_stations` members. One row per kept cluster, largest first (ties by representative id):
    cluster id, representative (member closest to the centroid), member ids and names,
    centroid and diameter."""
    gateways = gateways.sort_values("station_id").reset_index(drop=True)
    distances = distance_matrix_km(gateways["latitude"].to_numpy(), gateways["longitude"].to_numpy())
    labels = fcluster(linkage(squareform(distances, checks=False), method="complete"), diameter_km, "distance")

    clusters = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if len(members) < min_stations:
            continue
        points = unit_vectors(gateways["latitude"].to_numpy()[members], gateways["longitude"].to_numpy()[members])
        centroid = points.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        # closest member to the centroid (members are in station_id order, so ties go to the lowest id)
        representative = members[np.argmax(points @ centroid)]
        clusters.append({
            "representative_station_id": gateways.at[representative, "station_id"],
            "representative_station_name": gateways.at[representative, "station_name"],
            "latitude": gateways.at[representative, "latitude"],
            "longitude": gateways.at[representative, "longitude"],
            "antennas": len(members),
            "member_station_ids": ";".join(gateways["station_id"].to_numpy()[members]),
            "member_station_names": ";".join(gateways["station_name"].to_numpy()[members]),
            "centroid_latitude": round(float(np.degrees(np.arcsin(centroid[2]))), 4),
            "centroid_longitude": round(float(np.degrees(np.arctan2(centroid[1], centroid[0]))), 4),
            "diameter_km": round(float(distances[np.ix_(members, members)].max()), 1),
        })
    if not clusters:
        raise ValueError(f"no cluster of at least {min_stations} stations within {diameter_km} km")

    table = pd.DataFrame(clusters).sort_values(["antennas", "representative_station_id"], ascending=[False, True])
    table.insert(0, "station_id", [f"USCLUSTER_{index:02d}" for index in range(1, len(table) + 1)])
    table.insert(1, "station_name", table["representative_station_name"] + " cluster")
    return table.reset_index(drop=True)


def antenna_copies(chunk: pd.DataFrame, clusters: pd.DataFrame) -> list[pd.DataFrame]:
    """The slots of `chunk` (read as text) on a cluster representative, copied onto each antenna
    of its cluster: one frame per (cluster, antenna)."""
    by_representative = clusters.set_index("representative_station_id")
    kept = chunk[chunk["station_id"].isin(by_representative.index)]
    copies = []
    for representative, rows in kept.groupby("station_id", sort=False):
        cluster = by_representative.loc[representative]
        for antenna in range(1, int(cluster["antennas"]) + 1):
            copy = rows.copy()
            copy["slot_id"] = rows["slot_id"] + f"_A{antenna}"
            copy["antenna_id"] = f"ANT_{cluster['station_id']}_{antenna}"
            copy["station_id"] = cluster["station_id"]
            copy["station_name"] = cluster["station_name"]
            copies.append(copy)
    return copies


def aggregate_slots(input_path: Path, output_path: Path, clusters: pd.DataFrame,
                    chunk_rows: int = CHUNK_ROWS) -> dict[str, int]:
    """Copy every slot of each cluster representative onto the cluster's antennas and write them
    to `output_path`. Returns the slots read, the representative slots and the slots written."""
    slots_in = representative_slots = slots_out = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first = True
    # every column as text, so the copied values are written back exactly as they were read
    for chunk in pd.read_csv(input_path, dtype=str, keep_default_na=False, chunksize=chunk_rows):
        slots_in += len(chunk)
        representative_slots += int(chunk["station_id"].isin(clusters["representative_station_id"]).sum())
        copies = antenna_copies(chunk, clusters)
        if copies:
            out = pd.concat(copies)
            out.to_csv(output_path, mode="w" if first else "a", header=first, index=False)
            first = False
            slots_out += len(out)
        print(f"  read {slots_in:,} slots, wrote {slots_out:,}", flush=True)
    if first:
        raise ValueError("no slot of any cluster representative was found in the input")
    return {"slots_in": slots_in, "representative_slots": representative_slots, "slots_out": slots_out}


def parse_args(argv: list[str] | None = None, description: str = "Aggregate clusters of nearby US ground "
               "stations into multi-antenna stations and drop every other station.",
               output_suffix: str = OUTPUT_SUFFIX) -> argparse.Namespace:
    """Command line shared with dataset_us_clusters_intl.py, which passes its own description and suffix."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("input", type=Path, nargs="?", default=None,
                        help="Layer E slot CSV (default: data/dataset/starlink/<slots output_csv_name from the "
                             "data-ingestion config>)")
    parser.add_argument("--gateways", type=Path, default=None,
                        help="Layer B gateway CSV with coordinates and countries "
                             "(default: data/dataset/starlink/intermediate/<gateways output_csv_name>)")
    parser.add_argument("--diameter-km", type=float, default=150.0,
                        help="largest distance between two stations of a cluster (default: %(default)s)")
    parser.add_argument("--min-stations", type=int, default=4,
                        help="smallest cluster kept as an aggregated station (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=None,
                        help=f"output slot CSV (default: {DEFAULT_OUT_DIR.relative_to(PROJECT_ROOT).as_posix()}/"
                             f"<input name>{output_suffix}.csv); the gateway table is written next to it")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="data-ingestion config JSON, used only for the default paths (default: %(default)s)")
    args = parser.parse_args(argv)
    if args.diameter_km <= 0 or args.min_stations < 1:
        parser.error("--diameter-km must be positive and --min-stations at least 1")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
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
    us = gateways[gateways["country"].eq(CLUSTERED_COUNTRY) & gateways["station_id"].isin(with_slots)]
    clusters = cluster_stations(us, args.diameter_km, args.min_stations)
    print(f"{len(us)} US stations with slots -> {len(clusters)} clusters of at least {args.min_stations} stations "
          f"within {args.diameter_km:g} km, merging {clusters['antennas'].sum()} stations")

    print(f"Copying the representatives' slots onto the cluster antennas")
    report = aggregate_slots(input_path, output_path, clusters)
    clusters.assign(country=CLUSTERED_COUNTRY, status="aggregated", source=gateways_path.name).to_csv(
        gateways_output, index=False)

    print(f"Saved {report['slots_out']:,} slots ({report['representative_slots']:,} representative slots "
          f"copied onto {clusters['antennas'].sum()} antennas) to {output_path}")
    print(f"Saved the {len(clusters)} aggregated stations to {gateways_output}")
    for row in clusters.itertuples(index=False):
        print(f"  {row.station_id}  {row.station_name:<28} {row.antennas:>2} antennas, "
              f"diameter {row.diameter_km:>5.1f} km: {row.member_station_names.replace(';', ', ')}")


if __name__ == "__main__":
    main()
