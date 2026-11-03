"""Layer B of the data-ingestion pipeline: Starlink gateway ground stations.

Builds the ground-station table consumed by Layer C (visibility windows). The stations are
Starlink gateways: fixed earth stations that connect satellites to terrestrial internet.
The source is the public Hugging Face dataset `28hery/starlink-ground-stations`. It is
unofficial: its card describes it as derived from Starlink Insider community data, FCC IBFS
filings, and geocoding for stations without coordinates.

Steps
    1. load the raw rows from the local JSON cache, or download them page by page from the
       Hugging Face dataset-server rows API
    2. clean them: valid coordinates, normalized names and status, a country label, and one
       row per (name, latitude, longitude), keeping `operational` over `planned`
    3. assign station ids GATEWAY_0001, ... in (name, latitude, longitude) order
    4. validate the table and write it as CSV

The cache is reused by default and replaced only with --refresh, so reruns are reproducible.
Station ids depend on the sorted station list: a refreshed source with new stations shifts them.

Output columns: station_id, station_name, latitude, longitude, country, status, source.

Run from anywhere, for example:
    python src/data_ingestion/gateways.py
    python src/data_ingestion/gateways.py --out-dir data/my_run
    python src/data_ingestion/gateways.py --refresh          # download fresh rows
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
CONFIG_SECTION = "gateways"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "dataset" / "starlink" / "intermediate"

USER_AGENT = "mmsp-quant-ann/0.1 (research data ingestion)"
SOURCE_COLUMNS = ("name", "lat", "lon", "status")
OUTPUT_COLUMNS = ("station_id", "station_name", "latitude", "longitude", "country", "status", "source")
STATUS_PRIORITY = {"operational": 0, "planned": 1}  # lower wins among duplicates, others rank last

US_STATE_AND_TERRITORY_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR", "GU", "VI",
}
COUNTRY_ALIASES = {"UK": "United Kingdom", "NZ": "New Zealand", "DR": "Dominican Republic"}

log = logging.getLogger("data_ingestion.gateways")


def fetch_huggingface_rows(api_url: str, dataset_id: str, dataset_config: str, split: str,
                           rows_per_request: int, timeout: float) -> list[dict]:
    """Download every row of a Hugging Face dataset split through the dataset-server rows API.

    The API serves at most 100 rows per request, so the split is read page by page until the
    `num_rows_total` it reports has been collected.
    """
    rows: list[dict] = []
    total = None
    while total is None or len(rows) < total:
        query = urlencode({"dataset": dataset_id, "config": dataset_config, "split": split,
                           "offset": len(rows), "length": rows_per_request})
        request = Request(f"{api_url}?{query}", headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        total = payload["num_rows_total"]
        batch = [item["row"] for item in payload["rows"]]
        if not batch and len(rows) < total:
            raise RuntimeError(f"Rows API returned no rows at offset {len(rows)} of {total}.")
        rows.extend(batch)
    return rows


def load_gateway_rows(cache_path: Path, source: dict, refresh: bool = False) -> list[dict]:
    """Return the raw gateway rows, from `cache_path` when it exists and `refresh` is False.

    Otherwise the rows are downloaded (`source` holds the fetch_huggingface_rows arguments) and
    cached as JSON. A failed download raises: an explicit refresh never falls back to stale data.
    """
    if cache_path.exists() and not refresh:
        log.info("Using cached gateway rows %s", cache_path)
        raw = cache_path.read_bytes()
    else:
        log.info("Downloading gateway rows of %s from %s", source["dataset_id"], source["api_url"])
        try:
            rows = fetch_huggingface_rows(**source)
        except OSError as exc:
            raise RuntimeError(f"Could not download gateway rows from {source['api_url']}: {exc}") from exc
        raw = json.dumps(rows, indent=2).encode("utf-8")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(raw)
        log.info("Saved %d raw rows to %s", len(rows), cache_path)
    log.info("Gateway cache sha256: %s", hashlib.sha256(raw).hexdigest())
    return json.loads(raw.decode("utf-8"))


def normalize_string(value: object) -> str:
    """Strip the value and collapse every run of whitespace into a single space."""
    return " ".join(str(value).strip().split())


def infer_country(station_name: str) -> str:
    """Infer a country label from a `City, State/Country` station name.

    US state and territory codes give `US`, a few abbreviations are expanded (COUNTRY_ALIASES),
    any other suffix is returned as is, and names without a suffix give `unknown`.
    """
    if "," not in station_name:
        return "unknown"
    suffix = station_name.rsplit(",", 1)[1].strip()
    if suffix in US_STATE_AND_TERRITORY_CODES:
        return "US"
    return COUNTRY_ALIASES.get(suffix, suffix or "unknown")


def clean_gateway_table(raw_gateways: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """Turn raw rows (`name`, `lat`, `lon`, `status`) into the project's gateway schema.

    Rows with out-of-range or non-numeric coordinates, or an empty name or status, are dropped.
    Rows sharing name and coordinates are merged into one, preferring `operational` over `planned`.
    """
    missing_columns = [column for column in SOURCE_COLUMNS if column not in raw_gateways.columns]
    if missing_columns:
        raise ValueError(f"Missing required source columns: {missing_columns}")

    gateways = raw_gateways.loc[:, list(SOURCE_COLUMNS)].copy()
    gateways["station_name"] = gateways["name"].map(normalize_string)
    gateways["latitude"] = pd.to_numeric(gateways["lat"], errors="coerce")
    gateways["longitude"] = pd.to_numeric(gateways["lon"], errors="coerce")
    gateways["status"] = gateways["status"].map(normalize_string).str.lower()
    gateways["country"] = gateways["station_name"].map(infer_country)
    gateways["source"] = source_label

    valid = (
        gateways["latitude"].between(-90, 90)
        & gateways["longitude"].between(-180, 180)
        & gateways["station_name"].ne("")
        & gateways["status"].ne("")
    )
    dropped = int((~valid).sum())
    if dropped:
        log.info("Dropped %d rows with invalid coordinates or empty name/status.", dropped)
    gateways = gateways.loc[valid].copy()

    key = ["station_name", "latitude", "longitude"]
    gateways["status_priority"] = gateways["status"].map(STATUS_PRIORITY).fillna(len(STATUS_PRIORITY))
    gateways = (
        gateways.sort_values([*key, "status_priority"])
        .drop_duplicates(subset=key, keep="first")
        .sort_values(key)
        .reset_index(drop=True)
    )
    gateways.insert(0, "station_id", [f"GATEWAY_{index:04d}" for index in range(1, len(gateways) + 1)])
    return gateways[list(OUTPUT_COLUMNS)]


def validate_gateway_table(gateways: pd.DataFrame) -> dict[str, int]:
    """Check the table is usable downstream and return a report; raise if any check fails."""
    missing_columns = [column for column in OUTPUT_COLUMNS if column not in gateways.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    report = {
        "rows": len(gateways),
        "duplicate_station_ids": int(gateways["station_id"].duplicated().sum()),
        "missing_coordinates": int(gateways[["latitude", "longitude"]].isna().any(axis=1).sum()),
        "unknown_country": int(gateways["country"].eq("unknown").sum()),
    }
    failed_checks = {name: report[name] for name in ("duplicate_station_ids", "missing_coordinates") if report[name]}
    if report["rows"] == 0:
        failed_checks["rows"] = 0
    if failed_checks:
        raise ValueError(f"Gateway table validation failed: {failed_checks}")
    if report["unknown_country"]:
        log.warning("%d gateways have no recognizable country suffix.", report["unknown_country"])
    return report


def build_gateway_table(config: dict, cache_path: Path, refresh: bool = False) -> pd.DataFrame:
    """Run Layer B end to end from the "gateways" config section and return the validated table."""
    source = {
        "api_url": config["rows_api_url"],
        "dataset_id": config["dataset_id"],
        "dataset_config": config["dataset_config"],
        "split": config["dataset_split"],
        "rows_per_request": config["rows_per_request"],
        "timeout": config["download_timeout_seconds"],
    }
    raw_gateways = pd.DataFrame.from_records(load_gateway_rows(cache_path, source, refresh))
    source_label = f"Hugging Face {config['dataset_id']} ({config['dataset_config']})"
    gateways = clean_gateway_table(raw_gateways, source_label)
    report = validate_gateway_table(gateways)
    log.info("Raw rows %d, clean gateways %d. Validation: %s", len(raw_gateways), len(gateways), report)
    return gateways


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer B: build the Starlink gateway table from Hugging Face.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help=f"data-ingestion config JSON, reads its \"{CONFIG_SECTION}\" section (default: %(default)s)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="directory for the output CSV (default: %(default)s)")
    parser.add_argument("--raw-cache", type=Path, default=None,
                        help="raw rows JSON cache (default: <out-dir>/<raw_cache_name from the config>)")
    parser.add_argument("--refresh", action="store_true",
                        help="download fresh rows and overwrite the cache (may change the dataset)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))[CONFIG_SECTION]
    cache_path = args.raw_cache if args.raw_cache is not None else args.out_dir / config["raw_cache_name"]

    gateways = build_gateway_table(config, cache_path, args.refresh)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / config["output_csv_name"]
    gateways.to_csv(output_path, index=False)
    log.info("Saved %d gateways to %s", len(gateways), output_path)


if __name__ == "__main__":
    main()
