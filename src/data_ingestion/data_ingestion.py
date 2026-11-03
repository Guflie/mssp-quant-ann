"""Run the data-ingestion pipeline: every layer script, in order.

    satellites.py   Layer A  Starlink TLEs -> satellite table
    gateways.py     Layer B  Hugging Face gateway rows -> gateway table
    visibility.py   Layer C  satellites x gateways -> visibility windows
    slots.py        Layer E  windows -> weighted candidate slots

Every layer reads its settings from the same config file and runs exactly as when started on
its own. Layers A-C write to the intermediate directory, Layer E reads the windows from there
and writes the slots to the output directory.

Run from anywhere, for example:
    python src/data_ingestion/data_ingestion.py                    # whole pipeline, cached downloads
    python src/data_ingestion/data_ingestion.py --refresh          # download fresh TLEs and gateways
    python src/data_ingestion/data_ingestion.py --from visibility  # keep Layers A-B, rebuild C and E
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import gateways
import satellites
import slots
import visibility

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
DEFAULT_INTERMEDIATE_DIR = satellites.DEFAULT_OUT_DIR
DEFAULT_OUT_DIR = slots.DEFAULT_OUT_DIR

STAGES = (("satellites", "Layer A", satellites), ("gateways", "Layer B", gateways),
          ("visibility", "Layer C", visibility), ("slots", "Layer E", slots))
STAGE_NAMES = [name for name, _, _ in STAGES]
DOWNLOAD_STAGES = ("satellites", "gateways")

log = logging.getLogger("data_ingestion")


def stage_arguments(name: str, args: argparse.Namespace, config: dict) -> list[str]:
    """Command-line arguments passed to the main() of the layer `name`."""
    arguments = ["--config", str(args.config)]
    if name == "slots":
        windows_path = args.intermediate_dir / config["visibility"]["output_csv_name"]
        return [*arguments, "--windows", str(windows_path), "--out-dir", str(args.out_dir)]
    arguments += ["--out-dir", str(args.intermediate_dir)]
    if args.refresh and name in DOWNLOAD_STAGES:
        arguments.append("--refresh")
    return arguments


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the data-ingestion pipeline (Layers A, B, C, E) in order.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="data-ingestion config JSON (default: %(default)s)")
    parser.add_argument("--from", dest="start", choices=STAGE_NAMES, default=STAGE_NAMES[0],
                        help="first layer to run, the earlier ones are skipped (default: %(default)s)")
    parser.add_argument("--refresh", action="store_true",
                        help="download fresh TLEs and gateway rows in Layers A and B (changes the dataset)")
    parser.add_argument("--intermediate-dir", type=Path, default=DEFAULT_INTERMEDIATE_DIR,
                        help="directory of the Layer A-C outputs (default: %(default)s)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="directory of the final slot CSV (default: %(default)s)")
    args = parser.parse_args(argv)
    if args.refresh and STAGE_NAMES.index(args.start) >= len(DOWNLOAD_STAGES):
        parser.error(f"--refresh has no effect when starting from {args.start}: Layers A and B are skipped.")
    return args


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))

    pipeline_started = time.perf_counter()
    for name, label, module in STAGES[STAGE_NAMES.index(args.start):]:
        log.info("===== %s: %s.py", label, name)
        started = time.perf_counter()
        module.main(stage_arguments(name, args, config))
        log.info("===== %s done in %.1f s", label, time.perf_counter() - started)
    log.info("Data ingestion finished in %.1f s", time.perf_counter() - pipeline_started)


if __name__ == "__main__":
    main()
