"""Layer E of the data-ingestion pipeline: weighted candidate slots.

Bridges the visibility windows of Layer C to the optimization pipelines. Each window is
discretized into fixed-length candidate slots, the units the MWIS and CSP models choose from.

Operational model (synthetic: gateway maintenance and reconfiguration logs are not public)
    every gateway is a single antenna that serves one contact at a time and
    needs `reconfiguration_time_seconds` to re-point between two contacts. Gateways are
    available for the whole horizon (no maintenance blackouts).

Steps
    1. load the Layer C windows
    2. cut every window into slots of `min_contact_duration_seconds` (from the "visibility"
       section), starting every `step_size_seconds`, that fit entirely inside the window
    3. give every window a priority, urgency and weight, inherited by all its slots:
         priority  1-5, a hash of (seed, NORAD id), +1 when max elevation >= 60 deg and
                   -1 when < 20 deg
         urgency   1-5, decreasing linearly with the window start over the horizon
         weight    100 * (w_p * priority_norm + w_d * duration_norm + w_u * urgency_norm
                          + w_e * elevation_norm), every norm in [0, 1]
    4. write the weighted slots, a single table meant to be sliced into experiment instances
       (e.g. with scripts/trim_dataset.py)

The scenario horizon is the earliest start and latest end of the windows, rounded outwards to
the minute; slot times are also given in hours from its start (`*_hours` columns).

Run from anywhere, for example:
    python src/data_ingestion/slots.py
    python src/data_ingestion/slots.py --windows my/windows.csv --out-dir data/my_run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
CONFIG_SECTION = "slots"
DEFAULT_IN_DIR = PROJECT_ROOT / "data" / "dataset" / "starlink" / "intermediate"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "dataset" / "starlink"

WINDOW_COLUMNS = ("window_id", "satellite_name", "norad_id", "station_id", "station_name", "start_time",
                  "end_time", "duration_seconds", "max_elevation_deg", "source_epoch")
METADATA_COLUMNS = ("priority", "urgency", "priority_norm", "duration_norm", "urgency_norm", "elevation_norm",
                    "weight")
FULL_SCENARIO_TAG = "starlink_realistic_discretized_full"

log = logging.getLogger("data_ingestion.slots")


# ---------------------------------------------------------------- inputs

def load_visibility_windows(path: Path) -> pd.DataFrame:
    """Load the Layer C windows with parsed UTC times, sorted by (start_time, station_id, norad_id)."""
    if not path.exists():
        raise FileNotFoundError(f"Missing Layer C visibility CSV: {path}")
    windows = pd.read_csv(path)
    missing_columns = [column for column in WINDOW_COLUMNS if column not in windows.columns]
    if missing_columns:
        raise ValueError(f"Missing required Layer C columns: {missing_columns}")
    if windows.empty:
        raise ValueError(f"No visibility windows in {path}.")
    windows["start_time"] = pd.to_datetime(windows["start_time"], utc=True)
    windows["end_time"] = pd.to_datetime(windows["end_time"], utc=True)
    return windows.sort_values(["start_time", "station_id", "norad_id"]).reset_index(drop=True)


def time_bounds(windows: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Earliest window start and latest window end, rounded outwards to the minute."""
    return windows["start_time"].min().floor("min"), windows["end_time"].max().ceil("min")


def deterministic_integer(min_value: int, max_value: int, *parts: object) -> int:
    """Return a stable pseudo-random integer in [min_value, max_value] derived from the SHA-256 of `parts`.

    Every combination of `parts` gets its own independent draw. A single seeded generator would
    not do: its values depend on how many draws came before, so adding or removing one satellite
    would change the priorities of every satellite after it.
    """
    if min_value > max_value:
        raise ValueError("min_value cannot be greater than max_value.")
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return min_value + (int(digest[:12], 16) % (max_value - min_value + 1))


# ---------------------------------------------------------------- E1: candidate slots

def generate_candidate_slots(windows: pd.DataFrame, scenario_start: pd.Timestamp, contact_seconds: int,
                             step_seconds: int, reconfiguration_seconds: float) -> pd.DataFrame:
    """Cut every window into slots of `contact_seconds`, one every `step_seconds`, that fit inside it.

    Slots are sorted by (start_time, station_id, norad_id, index in parent) and numbered SLOT_000001, ...
    """
    contact = pd.Timedelta(seconds=contact_seconds)
    step = pd.Timedelta(seconds=step_seconds)
    records = []
    for window in windows.itertuples(index=False):
        window_duration_seconds = int((window.end_time - window.start_time).total_seconds())
        latest_start = window.end_time - contact
        current_start = window.start_time
        local_index = 1
        while current_start <= latest_start:
            current_end = current_start + contact
            records.append({
                "parent_window_id": window.window_id,
                "source_interval_id": window.window_id,
                "satellite_name": window.satellite_name,
                "satellite_id": str(window.norad_id),
                "norad_id": int(window.norad_id),
                "antenna_id": f"ANT_{window.station_id}",
                "station_id": window.station_id,
                "station_name": window.station_name,
                "start_time": current_start.isoformat(),
                "end_time": current_end.isoformat(),
                "start_time_hours": (current_start - scenario_start).total_seconds() / 3600,
                "end_time_hours": (current_end - scenario_start).total_seconds() / 3600,
                "duration_seconds": contact_seconds,
                "duration_hours": contact_seconds / 3600,
                "parent_visibility_start": window.start_time.isoformat(),
                "parent_visibility_end": window.end_time.isoformat(),
                "parent_visibility_duration_seconds": window_duration_seconds,
                "candidate_index_in_parent": local_index,
                "max_elevation_deg": window.max_elevation_deg,
                "source_epoch": window.source_epoch,
                "reconfiguration_time_seconds": reconfiguration_seconds,
                "reconfiguration_time_hours": reconfiguration_seconds / 3600,
            })
            local_index += 1
            current_start += step

    if not records:
        raise ValueError("No candidate slots were generated: no window is as long as the minimum contact.")
    slots = pd.DataFrame.from_records(records)
    slots = slots.sort_values(["start_time", "station_id", "norad_id", "candidate_index_in_parent"]).reset_index(drop=True)
    slots.insert(0, "slot_id", [f"SLOT_{index:06d}" for index in range(1, len(slots) + 1)])
    return slots


def check_slots_inside_parents(slots: pd.DataFrame, windows: pd.DataFrame) -> None:
    """Raise if any slot starts before or ends after its parent window."""
    bounds = windows.set_index("window_id")[["start_time", "end_time"]]
    parent_start = slots["parent_window_id"].map(bounds["start_time"])
    parent_end = slots["parent_window_id"].map(bounds["end_time"])
    inside = (pd.to_datetime(slots["start_time"], utc=True) >= parent_start) \
        & (pd.to_datetime(slots["end_time"], utc=True) <= parent_end)
    if not inside.all():
        raise ValueError(f"{int((~inside).sum())} slots fall outside their parent window.")


# ---------------------------------------------------------------- E2: priority, urgency, weight

def compute_window_metadata(windows: pd.DataFrame, scenario_start: pd.Timestamp, scenario_end: pd.Timestamp,
                            config: dict) -> pd.DataFrame:
    """Priority, urgency, normalized metrics and weight of every window, keyed by parent_window_id."""
    parents = windows.copy()
    horizon_seconds = max(1.0, (scenario_end - scenario_start).total_seconds())

    base_priority = [deterministic_integer(1, 5, config["priority_random_seed"], norad_id)
                     for norad_id in parents["norad_id"]]
    adjustment = parents["max_elevation_deg"].map(lambda elevation: 1 if elevation >= 60 else -1 if elevation < 20 else 0)
    parents["priority"] = (pd.Series(base_priority, index=parents.index) + adjustment).clip(lower=1, upper=5)

    seconds_until_start = (parents["start_time"] - scenario_start).dt.total_seconds().clip(lower=0)
    urgency_norm = (1.0 - (seconds_until_start / horizon_seconds)).clip(lower=0, upper=1)
    parents["urgency"] = (1 + (urgency_norm * 4)).round().astype(int).clip(lower=1, upper=5)

    max_duration = max(float(parents["duration_seconds"].max()), 1.0)
    parents["priority_norm"] = (parents["priority"] - 1) / 4
    parents["duration_norm"] = (parents["duration_seconds"] / max_duration).clip(lower=0, upper=1)
    parents["urgency_norm"] = (parents["urgency"] - 1) / 4
    parents["elevation_norm"] = (parents["max_elevation_deg"] / 90).clip(lower=0, upper=1)
    parents["weight"] = (
        100 * (
            config["priority_weight"] * parents["priority_norm"]
            + config["duration_weight"] * parents["duration_norm"]
            + config["urgency_weight"] * parents["urgency_norm"]
            + config["elevation_weight"] * parents["elevation_norm"]
        )
    ).round(3)
    return parents[["window_id", *METADATA_COLUMNS]].rename(columns={"window_id": "parent_window_id"})


def add_slot_weights(slots: pd.DataFrame, metadata: pd.DataFrame, contact_seconds: int,
                     step_seconds: int) -> pd.DataFrame:
    """Attach the parent-window metadata and the scenario labels to every slot."""
    slots = slots.merge(metadata, on="parent_window_id", how="left")
    if slots["weight"].isna().any():
        raise ValueError("Some candidate slots could not inherit parent opportunity metadata.")
    slots["scenario_tag"] = FULL_SCENARIO_TAG
    slots["weight_scope"] = "parent_visibility_window"
    slots["min_contact_duration_seconds"] = contact_seconds
    slots["discretization_step_seconds"] = step_seconds
    return slots


# ---------------------------------------------------------------- pipeline

def validate_settings(config: dict, contact_seconds: int) -> None:
    step_seconds = config["step_size_seconds"]
    if step_seconds <= 0 or contact_seconds <= 0:
        raise ValueError("Contact duration and step size must be positive.")
    if step_seconds > contact_seconds:
        raise ValueError("step_size_seconds should be less than or equal to min_contact_duration_seconds.")
    if config["reconfiguration_time_seconds"] < 0:
        raise ValueError("reconfiguration_time_seconds cannot be negative.")
    weights = [config[name] for name in ("priority_weight", "duration_weight", "urgency_weight", "elevation_weight")]
    if min(weights) < 0:
        raise ValueError("Weight coefficients cannot be negative.")


def build_slot_table(config: dict, contact_seconds: int, windows_path: Path) -> pd.DataFrame:
    """Run Layer E end to end and return the weighted slots."""
    validate_settings(config, contact_seconds)
    step_seconds = config["step_size_seconds"]

    windows = load_visibility_windows(windows_path)
    scenario_start, scenario_end = time_bounds(windows)
    slots = generate_candidate_slots(windows, scenario_start, contact_seconds, step_seconds,
                                     config["reconfiguration_time_seconds"])
    check_slots_inside_parents(slots, windows)
    metadata = compute_window_metadata(windows, scenario_start, scenario_end, config)
    slots = add_slot_weights(slots, metadata, contact_seconds, step_seconds)

    parents_with_slots = slots["parent_window_id"].nunique()
    log.info("Scenario %s - %s: %d slots from %d of %d windows (%d too short), %d satellites, %d stations",
             scenario_start.isoformat(), scenario_end.isoformat(), len(slots), parents_with_slots, len(windows),
             len(windows) - parents_with_slots, slots["norad_id"].nunique(), slots["station_id"].nunique())
    return slots


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer E: build the weighted candidate slots.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help=f"data-ingestion config JSON, reads its \"{CONFIG_SECTION}\" section (default: %(default)s)")
    parser.add_argument("--windows", type=Path, default=None,
                        help=f"Layer C windows CSV (default: {DEFAULT_IN_DIR}/<visibility.output_csv_name from the config>)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="directory for the output CSV (default: %(default)s)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    full_config = json.loads(args.config.read_text(encoding="utf-8"))
    config = full_config[CONFIG_SECTION]
    contact_seconds = full_config["visibility"]["min_contact_duration_seconds"]
    windows_path = args.windows or DEFAULT_IN_DIR / full_config["visibility"]["output_csv_name"]

    slots = build_slot_table(config, contact_seconds, windows_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    slots_path = args.out_dir / config["output_csv_name"]
    slots.to_csv(slots_path, index=False)
    log.info("Saved %d weighted slots to %s", len(slots), slots_path)


if __name__ == "__main__":
    main()
