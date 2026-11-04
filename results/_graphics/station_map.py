"""World map of the ground stations used by a slot dataset, saved as an image.

The slot CSV has no coordinates, so the stations are placed with the Layer B gateway table,
matched on station_id. Stations with slots are coloured by the number of distinct satellites
they can serve, i.e. with at least one slot at the station (--color-by satellites, default), or
by their number of slots (--color-by slots, log scale). Gateways of the table with no slot in
the dataset (e.g. the US ones in the no-USA variant) are drawn as hollow grey circles.

The background shows the country borders of Natural Earth (1:110m, public domain) in a plain
longitude/latitude projection, drawn with matplotlib only. The GeoJSON is downloaded on first
use and cached in data/external/natural_earth/; --no-basemap draws the bare scatter instead.

The slot CSV is read in chunks and only its station_id and norad_id columns, so large datasets
are fine.

Run from anywhere, for example:
    python results/_graphics/station_map.py
    python results/_graphics/station_map.py data/exp_datasets/starlink_scheduling_slots_no_usa.csv
    python results/_graphics/station_map.py my/slots.csv --output my/figures/map.jpg
    python results/_graphics/station_map.py --color-by slots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")  # file output only, no window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "scripts" / "data_ingestion" / "config.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "dataset" / "starlink"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "stations_map"
CHUNK_ROWS = 1_000_000

BASEMAP_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
               "geojson/ne_110m_admin_0_countries.geojson")
BASEMAP_CACHE = PROJECT_ROOT / "data" / "external" / "natural_earth" / "ne_110m_admin_0_countries.geojson"
USER_AGENT = "mmsp-quant-ann/0.1 (research plotting)"
OCEAN_COLOR, LAND_COLOR, BORDER_COLOR = "#e8f0f7", "#f7f5f0", "#b0b0b0"
LATITUDE_LIMITS = (-62, 85)  # no station lies south of -60 or north of 80


COLOR_METRICS = {  # column of station_statistics -> colour bar label, log scale
    "satellites": ("distinct satellites observed per station", False),
    "slots": ("slots per station (log scale)", True),
}


def station_statistics(slots_path: Path, chunk_rows: int = CHUNK_ROWS) -> tuple[pd.DataFrame, int]:
    """Per station_id of the slot CSV: number of slots and of distinct satellites (norad_id) with
    at least one slot there. Also returns the number of distinct satellites in the whole CSV."""
    slot_counts = pd.Series(dtype="int64")
    pairs = []  # distinct (station_id, norad_id) of every chunk
    for chunk in pd.read_csv(slots_path, usecols=["station_id", "norad_id"], dtype=str, chunksize=chunk_rows):
        slot_counts = slot_counts.add(chunk["station_id"].value_counts(), fill_value=0)
        pairs.append(chunk.drop_duplicates())
    pairs = pd.concat(pairs).drop_duplicates()
    statistics = pd.DataFrame({
        "slots": slot_counts.astype("int64"),
        "satellites": pairs.groupby("station_id")["norad_id"].nunique(),
    })
    return statistics, pairs["norad_id"].nunique()


def load_country_outlines(cache_path: Path = BASEMAP_CACHE, url: str = BASEMAP_URL) -> list[list[tuple]]:
    """Outer rings of the Natural Earth country polygons as (longitude, latitude) lists.

    The GeoJSON is read from `cache_path`, downloaded from `url` first if missing. Polygon holes
    are skipped: at this scale they are enclaves that are countries of their own anyway.
    """
    if not cache_path.exists():
        print(f"Downloading the world basemap from {url}")
        request = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=60) as response:
            raw = response.read()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(raw)
    geojson = json.loads(cache_path.read_text(encoding="utf-8"))

    outlines = []
    for feature in geojson["features"]:
        geometry = feature["geometry"]
        polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        outlines.extend([tuple(point[:2]) for point in polygon[0]] for polygon in polygons)
    return outlines


def plot_station_map(gateways: pd.DataFrame, statistics: pd.DataFrame, color_by: str, title: str,
                     output_path: Path, dpi: int, basemap: list[list[tuple]] | None = None) -> None:
    """Scatter the gateways on a longitude/latitude map, coloured by the `color_by` column of
    `statistics`, and save the figure to `output_path`."""
    colorbar_label, log_scale = COLOR_METRICS[color_by]
    stations = gateways.set_index("station_id").join(statistics[color_by], how="left")
    used = stations[stations[color_by].notna()].sort_values(color_by)  # highest values drawn on top
    unused = stations[stations[color_by].isna()]
    values = used[color_by]
    norm = (LogNorm if log_scale else Normalize)(vmin=values.min(), vmax=values.max())

    fig, ax = plt.subplots(figsize=(12, 5.6))
    if basemap is not None:
        ax.set_facecolor(OCEAN_COLOR)
        ax.add_collection(PolyCollection(basemap, facecolors=LAND_COLOR, edgecolors=BORDER_COLOR,
                                         linewidths=0.4, zorder=1))
    else:
        ax.axhline(0, color="0.85", linewidth=0.8, zorder=0)
        ax.axvline(0, color="0.85", linewidth=0.8, zorder=0)
    if len(unused):
        ax.scatter(unused["longitude"], unused["latitude"], s=14, facecolors="none", edgecolors="0.45",
                   linewidths=0.6, zorder=2, label=f"not in dataset ({len(unused)})")
    points = ax.scatter(used["longitude"], used["latitude"], c=values, s=28, cmap="viridis", norm=norm,
                        edgecolors="black", linewidths=0.3, zorder=3, label=f"in dataset ({len(used)})")
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-60, 61, 30))
    ax.set(xlim=(-180, 180), ylim=LATITUDE_LIMITS, xlabel="Longitude [deg]", ylabel="Latitude [deg]", title=title)
    ax.set_aspect("equal")
    colorbar_axes = make_axes_locatable(ax).append_axes("right", size="2%", pad=0.15)  # as tall as the map
    fig.colorbar(points, cax=colorbar_axes, label=colorbar_label)
    ax.grid(True, linewidth=0.4, alpha=0.4, zorder=1.5)
    ax.legend(loc="lower left", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the ground stations of a slot dataset on a world map.")
    parser.add_argument("slots", type=Path, nargs="?", default=None,
                        help="slot CSV (default: data/dataset/starlink/<slots output_csv_name from the config>)")
    parser.add_argument("--gateways", type=Path, default=None,
                        help="Layer B gateway CSV with the station coordinates "
                             "(default: data/dataset/starlink/intermediate/<gateways output_csv_name>)")
    parser.add_argument("--output", type=Path, default=None,
                        help="image file, its extension sets the format: .png, .jpg, .pdf, ... "
                             f"(default: {DEFAULT_FIGURE_DIR.relative_to(PROJECT_ROOT)}/"
                             "<slots name>_stations_<color-by>.png)")
    parser.add_argument("--color-by", choices=sorted(COLOR_METRICS), default="satellites",
                        help="colour the stations by distinct satellites served or by slots (default: %(default)s)")
    parser.add_argument("--title", default=None, help="figure title (default: built from the slot file name)")
    parser.add_argument("--dpi", type=int, default=200, help="resolution of raster formats (default: %(default)s)")
    parser.add_argument("--no-basemap", action="store_true", help="draw the stations without the world map")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="data-ingestion config JSON, used only for the default paths (default: %(default)s)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    slots_path = args.slots or DEFAULT_DATA_DIR / config["slots"]["output_csv_name"]
    gateways_path = args.gateways or DEFAULT_DATA_DIR / "intermediate" / config["gateways"]["output_csv_name"]
    output_path = args.output or DEFAULT_FIGURE_DIR / f"{slots_path.stem}_stations_{args.color_by}.png"

    print(f"Counting slots and satellites per station in {slots_path}")
    statistics, total_satellites = station_statistics(slots_path)
    gateways = pd.read_csv(gateways_path)
    missing = sorted(set(statistics.index) - set(gateways["station_id"]))
    if missing:
        raise SystemExit(f"{len(missing)} stations of the slot CSV are not in {gateways_path}, "
                         f"e.g. {missing[:5]}: was the gateway table rebuilt from a different source?")

    title = args.title or (f"{slots_path.stem}: {len(statistics)} stations, {total_satellites:,} satellites, "
                           f"{int(statistics['slots'].sum()):,} slots")
    basemap = None if args.no_basemap else load_country_outlines()
    plot_station_map(gateways, statistics, args.color_by, title, output_path, args.dpi, basemap)
    print(f"{args.color_by} per station: min {statistics[args.color_by].min():,}, "
          f"median {statistics[args.color_by].median():,.0f}, max {statistics[args.color_by].max():,}")
    print(f"Saved the map of {len(statistics)} stations to {output_path}")


if __name__ == "__main__":
    main()
