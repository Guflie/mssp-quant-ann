"""Trim a Layer C visibility-window CSV to exactly N satellites and M ground stations.

The window-level twin of trim_dataset.py: it selects satellites and stations with the same
trim() (same id order without --seed, same seeded shuffle with it), but runs on the Layer C
windows instead of the Layer E slots. Running Layer E (slots.py --windows) on the output then
builds the slots of the instance only, which avoids generating the full slot table when it is
too large for memory.

A window shorter than the slot length gets no slot in Layer E, so those windows are dropped
before the selection: a satellite or station counts as available only if it would have at
least one slot. With this, the selection is the same as trimming the full Layer E slot CSV with
trim_dataset.py and the same arguments. The slot length is `slot_length_seconds` of the
"slots" section of --config, or `min_contact_duration_seconds` of the "visibility" section for
configs without it.

Layer E normalizes durations by the longest window and urgency by the scenario horizon (first
window start to last window end) of its input, so the script reports both for the input and
the output: when they match, the slots weights are those of a full-table Layer E run.

The windows file can be large, so only the id and duration columns are loaded for the
selection, and the full rows are then filtered in chunks. Rows keep their input order. Both
passes over the file print a progress bar on stderr (percent of the file read).

Run from the project root, for example:
    python scripts/dataset_instances/trim_window.py data/dataset/starlink/intermediate/starlink_visibility_windows.csv --satellites 500 --stations 50 --seed 1234
    python scripts/dataset_instances/trim_window.py windows.csv --satellites 200 --stations 50 --output data/my_inputs/windows_s200_g50.csv
then
    python scripts/data_ingestion/slots.py --windows data/my_inputs/windows_s200_g50.csv --out-dir data/my_inputs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from trim_dataset import trim

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "scripts" / "data_ingestion" / "config.json"
SELECTION_COLUMNS = ["norad_id", "station_id", "duration_seconds"]
CHUNK_ROWS = 250_000
BAR_WIDTH = 30


class Progress:
    """Progress bar on stderr from the bytes read of the input file. On a terminal the bar is
    redrawn in place; when stderr is redirected (log file) one line is written per 10%."""

    def __init__(self, label: str, handle, total_bytes: int):
        self.label, self.handle, self.total = label, handle, max(total_bytes, 1)
        self.start = time.monotonic()
        self.tty = sys.stderr.isatty()
        self.last_decile = -1

    def update(self, rows: int, done: bool = False) -> None:
        fraction = 1.0 if done else min(self.handle.tell() / self.total, 1.0)
        if not self.tty:
            decile = int(fraction * 10)
            if decile == self.last_decile:
                return
            self.last_decile = decile
        elapsed = time.monotonic() - self.start
        eta = f", ~{elapsed / fraction - elapsed:.0f} s left" if 0 < fraction < 1 else ""
        filled = int(fraction * BAR_WIDTH)
        line = (f"{self.label} [{'#' * filled}{'.' * (BAR_WIDTH - filled)}] {fraction:4.0%} "
                f"{rows:,} rows, {elapsed:.0f} s{eta}")
        end = "\n" if done or not self.tty else ""
        print(f"\r{line:<100}" if self.tty else line, end=end, file=sys.stderr, flush=True)


def read_in_chunks(path: Path, label: str, **read_csv_args):
    """Yield the CSV in chunks of CHUNK_ROWS rows while drawing a progress bar."""
    with open(path, "rb") as handle:
        progress = Progress(label, handle, path.stat().st_size)
        rows = 0
        for chunk in pd.read_csv(handle, chunksize=CHUNK_ROWS, **read_csv_args):
            rows += len(chunk)
            progress.update(rows)
            yield chunk
        progress.update(rows, done=True)


def slot_length(config_path: Path) -> int:
    """Slot length Layer E would use with this config (same fallback as slots.py)."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config["slots"].get("slot_length_seconds", config["visibility"]["min_contact_duration_seconds"])


def horizon_summary(stats: dict) -> str:
    return f"longest window {stats['max_duration']} s, horizon {stats['start']} - {stats['end']}"


def update_stats(stats: dict, chunk: pd.DataFrame) -> None:
    """Running longest window and horizon bounds (ISO UTC strings compare chronologically)."""
    if chunk.empty:
        return
    stats["max_duration"] = max(stats.get("max_duration", 0), int(chunk["duration_seconds"].max()))
    stats["start"] = min(stats.get("start", chunk["start_time"].min()), chunk["start_time"].min())
    stats["end"] = max(stats.get("end", chunk["end_time"].max()), chunk["end_time"].max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Layer C visibility-window CSV")
    parser.add_argument("--satellites", type=int, required=True, help="satellites in the output")
    parser.add_argument("--stations", type=int, required=True, help="ground stations in the output")
    parser.add_argument("--seed", type=int, default=None,
                        help="sample satellites and stations in a random order fixed by this seed "
                             "(default: lowest NORAD ids and station ids)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="data-ingestion config giving the slot length (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=None,
                        help="output CSV (default: <input>_s<satellites>_g<stations>[_seed<seed>].csv next to the input)")
    args = parser.parse_args()
    if args.satellites <= 0 or args.stations <= 0:
        parser.error("--satellites and --stations must be positive")

    min_seconds = slot_length(args.config)

    # selection on the light columns only
    light = pd.concat(read_in_chunks(args.input, "1/2 reading ids", usecols=SELECTION_COLUMNS),
                      ignore_index=True)
    usable = light[light["duration_seconds"] >= min_seconds]
    print(f"selecting {args.satellites} satellites and {args.stations} stations ...", file=sys.stderr, flush=True)
    started = time.monotonic()
    selected = trim(usable, args.satellites, args.stations, args.seed)
    print(f"selection done in {time.monotonic() - started:.0f} s", file=sys.stderr, flush=True)
    satellites, stations = set(selected["norad_id"]), set(selected["station_id"])
    input_counts = (len(light), light["norad_id"].nunique(), light["station_id"].nunique())
    dropped_short = len(light) - len(usable)
    del light, usable, selected

    # full rows of the selection, filtered chunk by chunk
    seed_suffix = "" if args.seed is None else f"_seed{args.seed}"
    output = args.output or args.input.with_name(
        f"{args.input.stem}_s{args.satellites}_g{args.stations}{seed_suffix}.csv")
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

    print(f"input:  {input_counts[0]} windows, {input_counts[1]} satellites, {input_counts[2]} stations "
          f"({dropped_short} windows shorter than the {min_seconds} s slot ignored)")
    print(f"output: {kept} windows, {len(satellites)} satellites, {len(stations)} stations "
          f"({'id order' if args.seed is None else f'random order, seed {args.seed}'}) -> {output}")
    print(f"input  {horizon_summary(input_stats)}")
    print(f"output {horizon_summary(output_stats)}")
    if output_stats != input_stats:
        print("WARNING: longest window or horizon differ from the input: Layer E weights on this output "
              "will differ slightly from a full-table run")


if __name__ == "__main__":
    main()
