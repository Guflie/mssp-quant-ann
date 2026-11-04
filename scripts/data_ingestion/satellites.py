"""Layer A of the data-ingestion pipeline: Starlink satellite orbital data.

Builds the satellite table consumed by Layer C (visibility windows). CelesTrak publishes
General Perturbations (GP) data for the Starlink group in Two-Line Element (TLE) format;
skyfield parses each record with the SGP4 model and exposes its NORAD id and epoch.

Steps
    1. load the raw TLE text from the local cache, or download it from CelesTrak
    2. parse the 3-line records (name, line 1, line 2) into a table
    3. keep a deterministic subset: the first N satellites by NORAD catalog id
    4. validate the subset and write it as CSV

CelesTrak always serves the *latest* elements, so a download produces a different dataset
every time. The cache is therefore reused by default and replaced only with --refresh.

Output columns: satellite_name, norad_id, source, tle_line_1, tle_line_2, epoch (UTC ISO).

Run from anywhere, for example:
    python scripts/data_ingestion/satellites.py
    python scripts/data_ingestion/satellites.py --subset-size 1000 --out-dir data/my_run
    python scripts/data_ingestion/satellites.py --refresh          # download fresh elements
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd
from skyfield.api import EarthSatellite, load

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
CONFIG_SECTION = "satellites"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "dataset" / "starlink" / "intermediate"

SOURCE_LABEL = "CelesTrak Starlink GP/TLE"
USER_AGENT = "mmsp-quant-ann/0.1 (research data ingestion)"
REQUIRED_COLUMNS = ("satellite_name", "norad_id", "source", "tle_line_1", "tle_line_2", "epoch")

log = logging.getLogger("data_ingestion.satellites")


def download_tle_text(url: str, timeout: float) -> bytes:
    """Download the raw TLE file from `url` and return its bytes unchanged."""
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/plain,*/*"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def load_tle_text(url: str, cache_path: Path, timeout: float, refresh: bool = False) -> str:
    """Return the Starlink TLE text, from `cache_path` when it exists and `refresh` is False.

    Otherwise the file is downloaded from `url` (waiting at most `timeout` seconds) and stored
    byte-for-byte in `cache_path`.
    A failed download raises: an explicit refresh never silently falls back to stale data.
    """
    if cache_path.exists() and not refresh:
        log.info("Using cached TLE file %s", cache_path)
        raw = cache_path.read_bytes()
    else:
        log.info("Downloading TLE data from %s", url)
        try:
            raw = download_tle_text(url, timeout)
        except HTTPError as exc:
            # CelesTrak explains refusals in the body, e.g. a 403 when the data has not changed
            # since the last successful download (it is updated every 2 hours)
            detail = " ".join(exc.read().decode("utf-8", "replace").split())
            raise RuntimeError(f"CelesTrak refused the TLE download ({exc.code} {exc.reason}): {detail}") from exc
        except OSError as exc:
            raise RuntimeError(f"Could not download TLE data from {url}: {exc}") from exc
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(raw)
        log.info("Saved TLE cache to %s", cache_path)
    log.info("TLE file sha256: %s", hashlib.sha256(raw).hexdigest())
    return raw.decode("utf-8")


def parse_tles(tle_text: str, source: str = SOURCE_LABEL) -> pd.DataFrame:
    """Parse 3-line TLE text (name, line 1, line 2) into one row per satellite."""
    lines = [line.strip() for line in tle_text.splitlines() if line.strip()]
    if len(lines) % 3 != 0:
        raise ValueError(f"Expected TLE text in 3-line records, found {len(lines)} non-empty lines.")

    timescale = load.timescale()
    records = []
    for index in range(0, len(lines), 3):
        name, line_1, line_2 = lines[index:index + 3]
        if not (line_1.startswith("1 ") and line_2.startswith("2 ")):
            raise ValueError(f"Malformed TLE record for {name!r} at non-empty line {index + 1}.")
        satellite = EarthSatellite(line_1, line_2, name, timescale)
        records.append({
            "satellite_name": satellite.name,
            "norad_id": satellite.model.satnum,
            "source": source,
            "tle_line_1": line_1,
            "tle_line_2": line_2,
            "epoch": satellite.epoch.utc_iso(),
        })
    return pd.DataFrame.from_records(records, columns=list(REQUIRED_COLUMNS))


def select_satellite_subset(satellites: pd.DataFrame, subset_size: int) -> pd.DataFrame:
    """Keep the first `subset_size` satellites by NORAD catalog id (ties broken by name)."""
    if subset_size <= 0:
        raise ValueError("subset_size must be a positive integer.")
    return (
        satellites.sort_values(["norad_id", "satellite_name"])
        .head(subset_size)
        .reset_index(drop=True)
    )


def validate_satellite_table(satellites: pd.DataFrame) -> dict[str, int]:
    """Check the table is usable downstream and return a report; raise if any check fails."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in satellites.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    report = {
        "rows": len(satellites),
        "missing_norad_ids": int(satellites["norad_id"].isna().sum()),
        "empty_tle_line_1": int(satellites["tle_line_1"].fillna("").str.strip().eq("").sum()),
        "empty_tle_line_2": int(satellites["tle_line_2"].fillna("").str.strip().eq("").sum()),
        "duplicate_norad_ids": int(satellites["norad_id"].duplicated().sum()),
    }
    failed_checks = {name: value for name, value in report.items() if name != "rows" and value != 0}
    if failed_checks:
        raise ValueError(f"Satellite table validation failed: {failed_checks}")
    return report


def build_satellite_table(url: str, cache_path: Path, subset_size: int, timeout: float,
                          refresh: bool = False) -> pd.DataFrame:
    """Run Layer A end to end and return the validated satellite subset."""
    satellites = parse_tles(load_tle_text(url, cache_path, timeout, refresh))
    subset = select_satellite_subset(satellites, subset_size)
    report = validate_satellite_table(subset)
    log.info("Parsed %d satellites, kept %d (requested %d). Validation: %s",
             len(satellites), len(subset), subset_size, report)
    if len(subset) < subset_size:
        log.warning("Only %d satellites available, fewer than the %d requested.", len(subset), subset_size)
    return subset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer A: build the Starlink satellite table from CelesTrak TLEs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help=f"data-ingestion config JSON, reads its \"{CONFIG_SECTION}\" section (default: %(default)s)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="directory for the output CSV (default: %(default)s)")
    parser.add_argument("--tle-cache", type=Path, default=None,
                        help="TLE cache file (default: <out-dir>/<tle_cache_name from the config>)")
    parser.add_argument("--subset-size", type=int, default=None,
                        help="number of satellites to keep (default: subset_size from the config)")
    parser.add_argument("--refresh", action="store_true",
                        help="download fresh TLEs and overwrite the cache (changes the dataset)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))[CONFIG_SECTION]
    subset_size = args.subset_size if args.subset_size is not None else config["subset_size"]
    cache_path = args.tle_cache if args.tle_cache is not None else args.out_dir / config["tle_cache_name"]

    subset = build_satellite_table(config["tle_url"], cache_path, subset_size,
                                   config["download_timeout_seconds"], args.refresh)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / config["output_csv_name"]
    subset.to_csv(output_path, index=False)
    log.info("Saved %d satellites to %s", len(subset), output_path)


if __name__ == "__main__":
    main()
