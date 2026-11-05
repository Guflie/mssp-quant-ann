"""Dataset variant without US ground stations: drop every slot on a station in the United States.

Reads the full Layer E slot CSV and keeps the slots whose station is outside the US. The country
comes from the station name in each row (`City, State/Country`), with the same rule as Layer B
(gateways.infer_country): US states and territories (Alaska, Hawaii, Puerto Rico, Guam, ...) all
count as US, and names without a suffix are resolved by its override table.

The remaining rows are copied as text, unchanged: slot, window and station ids and weights are
the same as in the full dataset, so the variant differs from it only by the dropped stations.
The input is read in chunks, so the full CSV never has to fit in memory. The output goes to
data/exp_datasets/ by default, next to the other experiment datasets.

Run from anywhere, for example:
    python scripts/dataset_instances/dataset_no_USA.py
    python scripts/dataset_instances/dataset_no_USA.py my/slots.csv --output my/slots_no_usa.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INGESTION_DIR = PROJECT_ROOT / "scripts" / "data_ingestion"
sys.path.insert(0, str(DATA_INGESTION_DIR))
from gateways import infer_country  # noqa: E402

DEFAULT_CONFIG_PATH = DATA_INGESTION_DIR / "config.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "dataset" / "starlink"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "exp_datasets"
EXCLUDED_COUNTRY = "US"
OUTPUT_SUFFIX = "_no_usa"
CHUNK_ROWS = 500_000


def filter_slots(input_path: Path, output_path: Path, chunk_rows: int = CHUNK_ROWS) -> dict:
    """Copy the slots of non-US stations from `input_path` to `output_path` and return a report.

    The report counts slots read and kept, the station ids kept and dropped, and the kept slots
    per country.
    """
    countries: dict[str, str] = {}   # station_name -> country, inferred once per station
    kept_stations: set[str] = set()
    dropped_stations: set[str] = set()
    slots_per_country: Counter = Counter()
    slots_in = slots_out = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # every column as text, so the kept rows are written back exactly as they were read
    chunks = pd.read_csv(input_path, dtype=str, keep_default_na=False, chunksize=chunk_rows)
    for index, chunk in enumerate(chunks):
        for name in chunk["station_name"].unique():
            if name not in countries:
                countries[name] = infer_country(name)
        chunk_countries = chunk["station_name"].map(countries)
        is_kept = chunk_countries.ne(EXCLUDED_COUNTRY)
        chunk[is_kept].to_csv(output_path, mode="w" if index == 0 else "a", header=index == 0, index=False)

        slots_in += len(chunk)
        slots_out += int(is_kept.sum())
        kept_stations.update(chunk.loc[is_kept, "station_id"])
        dropped_stations.update(chunk.loc[~is_kept, "station_id"])
        slots_per_country.update(chunk_countries[is_kept].value_counts().to_dict())
        print(f"  read {slots_in:,} slots, kept {slots_out:,}", flush=True)

    return {"slots_in": slots_in, "slots_out": slots_out, "kept_stations": kept_stations,
            "dropped_stations": dropped_stations, "slots_per_country": slots_per_country}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drop every slot on a US ground station from a Layer E slot CSV.")
    parser.add_argument("input", type=Path, nargs="?", default=None,
                        help="Layer E slot CSV (default: data/dataset/starlink/<slots output_csv_name from the "
                             "data-ingestion config>)")
    parser.add_argument("--output", type=Path, default=None,
                        help=f"output CSV (default: {DEFAULT_OUT_DIR.relative_to(PROJECT_ROOT).as_posix()}/"
                             f"<input name>{OUTPUT_SUFFIX}.csv)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="data-ingestion config JSON, used only for the default input (default: %(default)s)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = args.input
    if input_path is None:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        input_path = DEFAULT_DATA_DIR / config["slots"]["output_csv_name"]
    output_path = args.output or DEFAULT_OUT_DIR / f"{input_path.stem}{OUTPUT_SUFFIX}.csv"
    if output_path.resolve() == input_path.resolve():
        raise SystemExit("The output path must differ from the input path.")

    print(f"Filtering {input_path}")
    report = filter_slots(input_path, output_path)
    print(f"Saved {report['slots_out']:,} of {report['slots_in']:,} slots to {output_path}")
    print(f"Stations kept {len(report['kept_stations'])}, dropped as US {len(report['dropped_stations'])}")
    print("Kept slots per country:")
    for country, count in report["slots_per_country"].most_common():
        print(f"  {country:<20} {count:,}")
    if report["slots_per_country"].get("unknown"):
        print("WARNING: some stations have an unknown country, add them to "
              "STATION_COUNTRY_OVERRIDES in scripts/data_ingestion/gateways.py")


if __name__ == "__main__":
    main()
