"""Layer C of the data-ingestion pipeline: satellite-to-gateway visibility windows.

Combines the satellites of Layer A with the gateways of Layer B. A satellite is visible from a
gateway while its elevation above the local horizon is at least the configured elevation mask;
a visibility window is a maximal interval of such visibility.

Steps
    1. load the Layer A and Layer B tables and keep the first N satellites (by NORAD id) and
       the first M gateways (by station id)
    2. resolve the horizon start, check it is close to the TLE epochs, and build a regular time
       grid over the horizon
    3. for every satellite-gateway pair, propagate the orbit (SGP4) on the grid, compute the
       elevation, and group consecutive samples at or above the mask into windows
    4. drop the windows that cannot host a minimum-length contact, assign window ids, and
       write the CSV

Windows are sampled, so their bounds are exact only to the grid step: a window starts at its
first visible sample and ends at its first non-visible sample (at the last sample when it is
still open at the end of the horizon).

Minimum contact
    A window shorter than `min_contact_duration_seconds` cannot host any scheduled contact, so it
    is dropped here rather than carried through Layers D and E. The same setting is the length
    of the candidate slots built from the windows in Layer E.

Horizon start
    SGP4 positions are accurate only within a few days of the TLE epoch, so the horizon must lie
    close to the epochs of the Layer A elements. `horizon_start_utc` is either a fixed ISO time
    (used to reproduce a frozen dataset) or "auto": the latest epoch of the selected satellites,
    rounded up to the next full hour, plus `horizon_start_offset_hours`. Either way, the run
    stops if most satellites would be propagated more than `max_epoch_distance_days` away from
    their epoch, and warns about the few that would.

Output columns: window_id, satellite_name, norad_id, station_id, station_name, start_time,
end_time, duration_seconds, max_elevation_deg, source_epoch.

Run from anywhere, for example:
    python scripts/data_ingestion/visibility.py
    python scripts/data_ingestion/visibility.py --satellites my/sats.csv --gateways my/gw.csv --out-dir data/my_run
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from skyfield.api import EarthSatellite, load, wgs84

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
CONFIG_SECTION = "visibility"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "dataset" / "starlink" / "intermediate"

OUTPUT_COLUMNS = ("window_id", "satellite_name", "norad_id", "station_id", "station_name", "start_time",
                  "end_time", "duration_seconds", "max_elevation_deg", "source_epoch")
PROGRESS_EVERY_SATELLITES = 100

log = logging.getLogger("data_ingestion.visibility")


def load_layer_inputs(satellites_path: Path, gateways_path: Path, satellite_subset_size: int,
                      gateway_subset_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Layer A and Layer B tables and keep the first satellites and gateways.

    Satellites are ordered by (norad_id, satellite_name), gateways by (station_id, station_name).
    """
    for path, layer in ((satellites_path, "Layer A satellite"), (gateways_path, "Layer B gateway")):
        if not path.exists():
            raise FileNotFoundError(f"Missing {layer} CSV: {path}")
    if satellite_subset_size <= 0 or gateway_subset_size <= 0:
        raise ValueError("Subset sizes must be positive integers.")

    satellites = (
        pd.read_csv(satellites_path)
        .sort_values(["norad_id", "satellite_name"])
        .head(satellite_subset_size)
        .reset_index(drop=True)
    )
    gateways = (
        pd.read_csv(gateways_path)
        .sort_values(["station_id", "station_name"])
        .head(gateway_subset_size)
        .reset_index(drop=True)
    )
    for table, requested, name in ((satellites, satellite_subset_size, "satellites"),
                                   (gateways, gateway_subset_size, "gateways")):
        if len(table) < requested:
            log.warning("Only %d %s available, fewer than the %d requested.", len(table), name, requested)
    return satellites, gateways


def resolve_horizon_start(setting: str, epochs: pd.Series, offset_hours: float) -> str:
    """Return the horizon start as an ISO UTC time.

    `setting` is returned unchanged unless it is "auto", in which case the start is the latest of
    `epochs` rounded up to the next full hour, plus `offset_hours`.
    """
    if setting != "auto":
        return setting
    latest_epoch = pd.to_datetime(epochs, utc=True).max()
    start = latest_epoch.ceil("h") + pd.Timedelta(hours=offset_hours)
    log.info("Automatic horizon start %s: latest TLE epoch %s + %s h offset (rounded up to the hour)",
             start.isoformat(), latest_epoch.isoformat(), offset_hours)
    return start.isoformat()


def check_epoch_distance(epochs: pd.Series, start: pd.Timestamp, end: pd.Timestamp, max_days: float) -> float:
    """Check how far from its TLE epoch each satellite is propagated over [start, end].

    Returns the largest distance in days. Raises if more than half of the satellites exceed
    `max_days` (the horizon does not match the elements), warns if only some do (stale elements).
    """
    epochs = pd.to_datetime(epochs, utc=True)
    distance_days = np.maximum((end - epochs).abs(), (epochs - start).abs()) / pd.Timedelta(days=1)
    too_far = int((distance_days > max_days).sum())
    if too_far > len(epochs) / 2:
        raise ValueError(
            f"Horizon {start.isoformat()} - {end.isoformat()} is more than {max_days} days from the TLE epoch "
            f"of {too_far}/{len(epochs)} satellites (epochs {epochs.min().isoformat()} - {epochs.max().isoformat()}). "
            f"Move horizon_start_utc next to the epochs or set it to \"auto\".")
    if too_far:
        log.warning("%d satellites are propagated more than %s days from their TLE epoch.", too_far, max_days)
    return float(distance_days.max())


def build_time_grid(start_utc: str, horizon_hours: float, step_seconds: int, timescale) -> tuple[pd.DatetimeIndex, object]:
    """Return the sample timestamps from `start_utc` to `start_utc + horizon_hours` (both included),
    as pandas timestamps and as the matching Skyfield Time array."""
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive.")
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive.")

    start = pd.Timestamp(start_utc).tz_convert("UTC")
    end = start + pd.Timedelta(hours=horizon_hours)
    timestamps = pd.date_range(start=start, end=end, freq=f"{step_seconds}s", inclusive="both")
    return timestamps, timescale.from_datetimes(timestamps.to_pydatetime())


def extract_windows(visible: np.ndarray, elevations_deg: np.ndarray, timestamps: pd.DatetimeIndex,
                    min_duration_seconds: int) -> list[dict]:
    """Group consecutive visible samples into windows of at least `min_duration_seconds`.

    A window starts at its first visible sample and ends at the first non-visible sample after
    it, or at the last sample when the satellite is still visible at the end of the horizon.
    """
    if len(visible) != len(timestamps):
        raise ValueError("visible and timestamps must have the same length.")

    # pad with False so every run of True has both a rising (+1) and a falling (-1) edge,
    # also when it touches the first or the last sample
    edges = np.diff(np.concatenate(([False], visible, [False])).astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)  # index of the first non-visible sample (len() if none)

    windows = []
    for start_index, end_index in zip(starts, ends):
        start_time = timestamps[start_index]
        end_time = timestamps[end_index] if end_index < len(timestamps) else timestamps[-1]
        duration_seconds = int((end_time - start_time).total_seconds())
        if duration_seconds < min_duration_seconds:
            continue
        windows.append({
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration_seconds,
            "max_elevation_deg": round(float(np.nanmax(elevations_deg[start_index:end_index])), 3),
        })
    return windows


def generate_visibility_windows(satellites: pd.DataFrame, gateways: pd.DataFrame, timestamps: pd.DatetimeIndex,
                                times, timescale, elevation_mask_deg: float,
                                min_duration_seconds: int) -> pd.DataFrame:
    """Compute the visibility windows of every satellite-gateway pair, sorted by
    (start_time, station_id, norad_id) and numbered VIS_000001, ..."""
    observers = [
        (gateway.station_id, gateway.station_name, wgs84.latlon(gateway.latitude, gateway.longitude))
        for gateway in gateways.itertuples(index=False)
    ]

    records = []
    started = time.perf_counter()
    for count, satellite in enumerate(satellites.itertuples(index=False), start=1):
        orbit = EarthSatellite(satellite.tle_line_1, satellite.tle_line_2, satellite.satellite_name, timescale)
        for station_id, station_name, observer in observers:
            altitude, _, _ = (orbit - observer).at(times).altaz()
            elevations_deg = altitude.degrees
            for window in extract_windows(elevations_deg >= elevation_mask_deg, elevations_deg,
                                          timestamps, min_duration_seconds):
                records.append({
                    "satellite_name": satellite.satellite_name,
                    "norad_id": int(satellite.norad_id),
                    "station_id": station_id,
                    "station_name": station_name,
                    **window,
                    "source_epoch": satellite.epoch,
                })
        if count % PROGRESS_EVERY_SATELLITES == 0 or count == len(satellites):
            log.info("Propagated %d/%d satellites (%.0f s, %d windows so far)",
                     count, len(satellites), time.perf_counter() - started, len(records))

    windows = pd.DataFrame.from_records(records, columns=[column for column in OUTPUT_COLUMNS if column != "window_id"])
    windows = windows.sort_values(["start_time", "station_id", "norad_id"]).reset_index(drop=True)
    windows.insert(0, "window_id", [f"VIS_{index:06d}" for index in range(1, len(windows) + 1)])
    return windows


def build_visibility_table(config: dict, satellites_path: Path, gateways_path: Path) -> pd.DataFrame:
    """Run Layer C end to end from the "visibility" config section and return the windows."""
    satellites, gateways = load_layer_inputs(satellites_path, gateways_path,
                                             config["satellite_subset_size"], config["gateway_subset_size"])
    horizon_start = resolve_horizon_start(config["horizon_start_utc"], satellites["epoch"],
                                          config["horizon_start_offset_hours"])
    timescale = load.timescale()
    timestamps, times = build_time_grid(horizon_start, config["horizon_hours"],
                                        config["time_step_seconds"], timescale)
    max_distance = check_epoch_distance(satellites["epoch"], timestamps[0], timestamps[-1],
                                        config["max_epoch_distance_days"])
    log.info("%d satellites x %d gateways, %d samples from %s to %s, elevation mask %s deg, "
             "max propagation from epoch %.2f days",
             len(satellites), len(gateways), len(timestamps), timestamps[0].isoformat(),
             timestamps[-1].isoformat(), config["elevation_mask_deg"], max_distance)

    windows = generate_visibility_windows(satellites, gateways, timestamps, times, timescale,
                                          config["elevation_mask_deg"], config["min_contact_duration_seconds"])
    log.info("Generated %d windows of at least %d s covering %d satellites and %d gateways",
             len(windows), config["min_contact_duration_seconds"], windows["norad_id"].nunique(),
             windows["station_id"].nunique())
    return windows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer C: compute satellite-to-gateway visibility windows.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help=f"data-ingestion config JSON, reads its \"{CONFIG_SECTION}\" section (default: %(default)s)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="directory for the output CSV, and default location of the inputs (default: %(default)s)")
    parser.add_argument("--satellites", type=Path, default=None,
                        help="Layer A satellite CSV (default: <out-dir>/<satellites.output_csv_name from the config>)")
    parser.add_argument("--gateways", type=Path, default=None,
                        help="Layer B gateway CSV (default: <out-dir>/<gateways.output_csv_name from the config>)")
    parser.add_argument("--horizon-start", default=None,
                        help="ISO UTC horizon start or \"auto\" (default: horizon_start_utc from the config)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    full_config = json.loads(args.config.read_text(encoding="utf-8"))
    config = full_config[CONFIG_SECTION]
    if args.horizon_start is not None:
        config["horizon_start_utc"] = args.horizon_start
    satellites_path = args.satellites or args.out_dir / full_config["satellites"]["output_csv_name"]
    gateways_path = args.gateways or args.out_dir / full_config["gateways"]["output_csv_name"]

    windows = build_visibility_table(config, satellites_path, gateways_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / config["output_csv_name"]
    windows.to_csv(output_path, index=False)
    log.info("Saved %d windows to %s", len(windows), output_path)


if __name__ == "__main__":
    main()
