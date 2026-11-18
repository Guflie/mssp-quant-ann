"""Map of the ground stations used by a slot dataset (contiguous USA or world), saved as an image.

The slot CSV has no coordinates, so the stations are placed with the Layer B gateway table,
matched on station_id. Stations with slots are coloured by the number of distinct satellites
they can serve, i.e. with at least one slot at the station (--color-by satellites, default), or
by their number of slots (--color-by slots, log scale). Gateways of the table with no slot in
the dataset (e.g. the US ones in the no-USA variant) are drawn as hollow grey circles.

The background shows the country borders of Natural Earth (1:110m, public domain) in a plain
longitude/latitude projection, drawn with matplotlib only. The GeoJSON is downloaded on first
use and cached in data/external/natural_earth/; --no-basemap draws the bare scatter instead.

--view chooses the area:
    world     (default) the whole world.
    usa       the contiguous USA only, where most gateways are crowded, with its colour bar;
              the colour scale spans the stations inside the area. Alaska, Hawaii and the
              territories fall outside it, a few stations of southern Canada and northern
              Mexico inside it.
    world+usa the world map with the USA area outlined, and the USA enlarged below it with the
              same colour scale.
The USA panel scales longitude by cos(latitude) at its centre, so the US keeps its usual shape.

The slot CSV is read in chunks and only its station_id and norad_id columns, so large datasets
are fine.

Run from anywhere, for example:
    python results/_graphics/station_map.py
    python results/_graphics/station_map.py data/exp_datasets/starlink_scheduling_slots_no_usa.csv
    python results/_graphics/station_map.py my/slots.csv --output my/figures/map.jpg
    python results/_graphics/station_map.py --color-by slots --view usa
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")  # file output only, no window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

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
ZOOM_COLOR = "crimson"  # outline of the USA area on the world map and frame of the USA panel

# map areas: name -> (label, longitude limits, latitude limits, (longitude, latitude) tick steps)
REGIONS = {
    "world": ("World", (-180, 180), LATITUDE_LIMITS, (60, 30)),
    # every gateway of the contiguous US lies within lon -124.4..-67.2, lat 25.8..48.8
    # (the box also holds a few stations of southern Canada and northern Mexico, counted in the panel)
    "usa": ("Contiguous USA", (-127, -64), (23, 51), (10, 10)),
}
# --view -> areas drawn, one panel each from top to bottom, and figure size in inches
VIEWS = {
    "usa": (["usa"], (10, 5.6)),
    "world": (["world"], (12, 5.6)),
    "world+usa": (["world", "usa"], (12, 11)),
}
MARKER_SIZES = {"world": 28, "usa": 60}


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


def draw_stations(ax: plt.Axes, used: pd.DataFrame, unused: pd.DataFrame, color_by: str, norm: Normalize,
                  basemap: list[list[tuple]] | None, marker_size: float) -> plt.Artist:
    """Draw the basemap and the stations on `ax`: used ones coloured by `color_by` with `norm`,
    unused ones as hollow grey circles. The legend counts only the stations drawn on `ax`
    (all of them for the world map, those inside the area for a zoom). Returns the coloured
    scatter, for the colour bar."""
    if basemap is not None:
        ax.set_facecolor(OCEAN_COLOR)
        ax.add_collection(PolyCollection(basemap, facecolors=LAND_COLOR, edgecolors=BORDER_COLOR,
                                         linewidths=0.4, zorder=1))
    else:
        ax.axhline(0, color="0.85", linewidth=0.8, zorder=0)
        ax.axvline(0, color="0.85", linewidth=0.8, zorder=0)
    if len(unused):
        ax.scatter(unused["longitude"], unused["latitude"], s=marker_size / 2, facecolors="none", edgecolors="0.45",
                   linewidths=0.6, zorder=2, label=f"not in dataset ({len(unused)})")
    points = ax.scatter(used["longitude"], used["latitude"], c=used[color_by], s=marker_size, cmap="viridis",
                        norm=norm, edgecolors="black", linewidths=0.3, zorder=3,
                        label=f"in dataset ({len(used)})" if len(used) else "_nolegend_")
    ax.grid(True, linewidth=0.4, alpha=0.4, zorder=1.5)
    ax.legend(loc="lower left", fontsize=9)
    return points


def inside(frame: pd.DataFrame, region: str) -> pd.DataFrame:
    """Rows of `frame` (with longitude and latitude columns) inside the area of `region`."""
    _, (lon_min, lon_max), (lat_min, lat_max), _ = REGIONS[region]
    return frame[frame["longitude"].between(lon_min, lon_max) & frame["latitude"].between(lat_min, lat_max)]


def plot_station_map(gateways: pd.DataFrame, statistics: pd.DataFrame, color_by: str, title: str,
                     output_path: Path, dpi: int, basemap: list[list[tuple]] | None = None,
                     view: str = "usa", area_subtitle: bool = True) -> None:
    """Scatter the gateways on a longitude/latitude map of the `view` areas (see VIEWS), coloured
    by the `color_by` column of `statistics`, and save the figure to `output_path`. All panels
    share the colour scale of the stations in the top panel; the colour bar sits beside it.
    `area_subtitle` adds to the title of a single regional panel how many dataset stations it
    shows."""
    colorbar_label, log_scale = COLOR_METRICS[color_by]
    stations = gateways.set_index("station_id").join(statistics[color_by], how="left")
    used = stations[stations[color_by].notna()].sort_values(color_by)  # highest values drawn on top
    unused = stations[stations[color_by].isna()]
    regions, figsize = VIEWS[view]

    # colour scale: the stations of the top panel (all of them unless it is a regional view);
    # with none there (e.g. the no-USA variant on the USA), no colour bar is drawn
    scale_values = inside(used, regions[0])[color_by]
    reference = scale_values if len(scale_values) else used[color_by]
    norm = (LogNorm if log_scale else Normalize)(vmin=reference.min(), vmax=reference.max())

    fig, axes = plt.subplots(len(regions), 1, figsize=figsize, squeeze=False,
                             gridspec_kw={"height_ratios": [5.6, 5.0][:len(regions)]})
    for position, (ax, region) in enumerate(zip(axes[:, 0], regions)):
        label, (lon_min, lon_max), (lat_min, lat_max), (lon_step, lat_step) = REGIONS[region]
        shown = inside(used, region)
        points = draw_stations(ax, shown, inside(unused, region), color_by, norm, basemap, MARKER_SIZES[region])
        ax.set_xticks(range(math.ceil(lon_min / lon_step) * lon_step, lon_max + 1, lon_step))
        ax.set_yticks(range(math.ceil(lat_min / lat_step) * lat_step, lat_max + 1, lat_step))
        ax.set(xlim=(lon_min, lon_max), ylim=(lat_min, lat_max), xlabel="Longitude [deg]", ylabel="Latitude [deg]")
        if region == "world":
            ax.set_aspect("equal")
        else:
            # one degree of longitude is cos(latitude) degrees of latitude long: keep the usual shape
            ax.set_aspect(1 / math.cos(math.radians((lat_min + lat_max) / 2)))

        if position == 0:
            subtitle = (f"\n{label} area: {len(shown)} of {len(used)} dataset stations"
                        if region != "world" and area_subtitle else "")
            ax.set_title(title + subtitle)
            if len(shown):
                # placed in the map's own coordinates, so it stays beside it and as tall as it
                # whatever room the fixed aspect ratio leaves around the map
                colorbar_axes = ax.inset_axes([1.015, 0, 0.02, 1])
                fig.colorbar(points, cax=colorbar_axes, label=colorbar_label)
        else:
            # a panel below the world map: outline its area on the world map and frame it alike
            axes[0, 0].add_patch(Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min, fill=False,
                                           edgecolor=ZOOM_COLOR, linewidth=1.2, zorder=4))
            for spine in ax.spines.values():
                spine.set(edgecolor=ZOOM_COLOR, linewidth=1.2)
            ax.set_title(f"{label} (outlined above, same colour scale): {len(shown)} dataset stations in this area")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the ground stations of a slot dataset on a map.")
    parser.add_argument("slots", type=Path, nargs="?", default=None,
                        help="slot CSV (default: data/dataset/starlink/<slots output_csv_name from the config>)")
    parser.add_argument("--gateways", type=Path, default=None,
                        help="Layer B gateway CSV with the station coordinates "
                             "(default: data/dataset/starlink/intermediate/<gateways output_csv_name>)")
    parser.add_argument("--output", type=Path, default=None,
                        help="image file, its extension sets the format: .png, .jpg, .pdf, ... "
                             f"(default: {DEFAULT_FIGURE_DIR.relative_to(PROJECT_ROOT)}/"
                             "<slots name>_stations_<color-by>[_usa | _world+usa].png)")
    parser.add_argument("--color-by", choices=sorted(COLOR_METRICS), default="satellites",
                        help="colour the stations by distinct satellites served or by slots (default: %(default)s)")
    parser.add_argument("--title", default=None, help="figure title (default: built from the slot file name)")
    parser.add_argument("--dpi", type=int, default=200, help="resolution of raster formats (default: %(default)s)")
    parser.add_argument("--no-basemap", action="store_true", help="draw the stations without the world map")
    parser.add_argument("--view", choices=list(VIEWS), default="world",
                        help="area to draw: the contiguous USA, the world, or the world with the USA "
                             "enlarged below (default: %(default)s)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="data-ingestion config JSON, used only for the default paths (default: %(default)s)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    slots_path = args.slots or DEFAULT_DATA_DIR / config["slots"]["output_csv_name"]
    gateways_path = args.gateways or DEFAULT_DATA_DIR / "intermediate" / config["gateways"]["output_csv_name"]
    view_suffix = "" if args.view == "world" else f"_{args.view}"   # world figures keep their old names
    output_path = args.output or DEFAULT_FIGURE_DIR / f"{slots_path.stem}_stations_{args.color_by}{view_suffix}.png"

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
    plot_station_map(gateways, statistics, args.color_by, title, output_path, args.dpi, basemap,
                     view=args.view, area_subtitle=args.title is None)
    print(f"{args.color_by} per station: min {statistics[args.color_by].min():,}, "
          f"median {statistics[args.color_by].median():,.0f}, max {statistics[args.color_by].max():,}")
    print(f"Saved the map of {len(statistics)} stations to {output_path}")


if __name__ == "__main__":
    main()
