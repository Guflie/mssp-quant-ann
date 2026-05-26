"""Generate a reproducible CSV dataset of random raw visibility intervals.

The generated file is intended to be the shared input for the notebook
pipeline. Phase 2 turns feasible raw intervals into `CandidateSlot` objects.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


DEFAULT_OUTPUT_PATH = Path("raw_visibility_intervals.csv")
DEFAULT_RANDOM_SEED = 42
DEFAULT_INTERVAL_COUNT = 100

SATELLITE_IDS = [f"SAT-{index:03d}" for index in range(1, 31)]
ANTENNA_TO_STATION = {
    "ANT-1": "STATION-NORTH",
    "ANT-2": "STATION-NORTH",
    "ANT-3": "STATION-SOUTH",
    "ANT-4": "STATION-SOUTH",
    "ANT-5": "STATION-EAST",
    "ANT-6": "STATION-WEST",
}
UNKNOWN_ANTENNA_ID = "ANT-UNKNOWN"


def build_interval_row(
    interval_id: str,
    satellite_id: str,
    antenna_id: str,
    visibility_start: float,
    visibility_end: float,
    required_duration: float,
    priority: float,
    urgency: float,
) -> dict[str, str | float]:
    """Build one raw visibility interval row."""
    return {
        "interval_id": interval_id,
        "satellite_id": satellite_id,
        "antenna_id": antenna_id,
        "station_id": ANTENNA_TO_STATION.get(antenna_id, "UNKNOWN-STATION"),
        "visibility_start": visibility_start,
        "visibility_end": visibility_end,
        "required_duration": required_duration,
        "priority": priority,
        "urgency": urgency,
    }


def generate_raw_visibility_interval_rows(
    interval_count: int,
    random_seed: int,
) -> list[dict[str, str | float]]:
    """Generate random raw visibility interval rows.

    Rows are generated organically and may be accepted or rejected depending on
    the Phase 2 feasibility filters.
    """
    rng = random.Random(random_seed)
    rows = []
    antenna_ids = list(ANTENNA_TO_STATION) + [UNKNOWN_ANTENNA_ID]

    for index in range(1, interval_count + 1):
        antenna_id = rng.choice(antenna_ids)

        # Visibility windows intentionally reach outside ideal bounds so the
        # discretization filters have realistic work to do.
        visibility_start = round(rng.uniform(-0.75, 12.25), 2)
        visibility_duration = round(rng.uniform(0.10, 3.00), 2)
        visibility_end = round(visibility_start + visibility_duration, 2)

        # Required communication duration is separate from the visibility
        # window length. Sometimes it will not fit, which Phase 2 should reject.
        required_duration = round(rng.uniform(-0.10, 1.60), 2)

        rows.append(
            build_interval_row(
                interval_id=f"raw_{index:03d}",
                satellite_id=rng.choice(SATELLITE_IDS),
                antenna_id=antenna_id,
                visibility_start=visibility_start,
                visibility_end=visibility_end,
                required_duration=required_duration,
                priority=round(rng.uniform(1.0, 10.0), 2),
                urgency=round(rng.uniform(1.0, 10.0), 2),
            )
        )

    return rows


def write_raw_visibility_interval_csv(rows: list[dict[str, str | float]], output_path: Path) -> None:
    """Write generated raw visibility interval rows to a CSV file."""
    fieldnames = [
        "interval_id",
        "satellite_id",
        "antenna_id",
        "station_id",
        "visibility_start",
        "visibility_end",
        "required_duration",
        "priority",
        "urgency",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Parse command-line arguments and generate the CSV dataset."""
    parser = argparse.ArgumentParser(description="Generate random raw visibility intervals as CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--count", type=int, default=DEFAULT_INTERVAL_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("Interval count must be positive.")
    rows = generate_raw_visibility_interval_rows(interval_count=args.count, random_seed=args.seed)
    write_raw_visibility_interval_csv(rows=rows, output_path=args.output)
    print(f"Wrote {len(rows)} raw visibility intervals to {args.output}")


if __name__ == "__main__":
    main()
