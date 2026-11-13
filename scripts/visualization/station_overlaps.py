"""Ground-station view of simultaneous conflicts: two stations with overlapping slots on the same satellite.

A conflict here is a pair of slots of the same satellite at two different stations whose time
intervals overlap (start_a < end_b and start_b < end_a): both stations would contact the same
satellite at the same moment. Reconfiguration time is not added, since the two slots use different
antennas. Only stations whose visible regions overlap can be in such a conflict, so unlike the
same-satellite constraint over the whole horizon this is local.

Per station (black star) one world map with the ground track of every satellite the station has a
slot on, propagated with SGP4 (skyfield):
    grey     the ground track
    blue     while the station has a slot on the satellite (its own slots)
    orange   while the station and another one have overlapping slots on it: where the satellite is
             while both stations would contact it (drawn over blue)
    links    (dark red) from the station to every station it conflicts with over the whole horizon; the
             number is the count of overlapping slot pairs, the triangle and line grow with it
Each satellite is drawn once, for one orbital period (clipped to the horizon) centred on one of its
passes over the station (a pass = the slots of one visibility window, parent_window_id): by default
the pass with the most overlapping slot pairs (ties: highest max elevation), --pass highest / first
for the highest or earliest one. --full-horizon draws the whole track over the horizon instead,
which shows every overlap but is crowded. The title says how many of the station's overlapping
pairs fall on the drawn orbits.

Plus overview maps, network.png (world) and network_zoom.png (zoomed on the stations involved), with
every pair of conflicting stations of the slot CSV, and
station_pairs.csv / overlapping_pairs.csv with the counts and every overlapping slot pair.

Inputs:
    slots          path to the slot CSV (required)
    --satellites   TLE table (Layer A, starlink_satellite_subset.csv): the satellite tracks are
                   propagated from it with SGP4 (skyfield). It must be the table the slot CSV was built
                   from; the script warns when the TLE epochs do not match
    --gateways     station table with coordinates (Layer B, starlink_gateways_clean.csv); for the
                   us-clusters variants the table of aggregated stations in data/test/intermediate/ is
                   picked from the slot file name
--station takes station ids or parts of station names (default: every station of the slot CSV).
Output: visualization/plot_station_overlaps/<slot file stem>/ (or --out-dir).

The basemap and its colours come from results/_graphics/station_map.py.

Example:
    .venv/Scripts/python.exe visualization/station_overlaps.py data/test/starlink_scheduling_slots_s500_g50_deg60.csv --station Milwaukee
    .venv/Scripts/python.exe visualization/station_overlaps.py data/test/starlink_scheduling_slots_s500_g50_deg60.csv --full-horizon
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from skyfield.api import EarthSatellite, load, wgs84  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "results" / "_graphics"))
from station_map import BORDER_COLOR, LAND_COLOR, OCEAN_COLOR, load_country_outlines  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "dataset" / "starlink" / "intermediate"
DEFAULT_SATELLITES = DATA_DIR / "starlink_satellite_subset.csv"
DEFAULT_GATEWAYS = DATA_DIR / "starlink_gateways_clean.csv"
TEST_INTERMEDIATE = PROJECT_ROOT / "data" / "test" / "intermediate"
# Variant tag in the slot file name -> gateway table of its aggregated stations (most specific first).
VARIANT_GATEWAYS = {
    "us-clusters-intl": TEST_INTERMEDIATE / "starlink_scheduling_slots_us_clusters_intl_gateways.csv",
    "us-clusters": TEST_INTERMEDIATE / "starlink_scheduling_slots_us_clusters_gateways.csv",
}
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / f"plot_{Path(__file__).stem}"  # visualization/plot_<script name>
SLOT_COLUMNS = ["slot_id", "parent_window_id", "satellite_name", "norad_id", "station_id", "start_time", "end_time",
                "max_elevation_deg", "source_epoch"]
COVERAGE, OVERLAP, PARTNER = "tab:blue", "tab:orange", "#9e0142"
TRACK, OWN, SHARED = 0, 1, 2  # colour classes of the ground track, drawn in this order
STYLE = {TRACK: dict(color="0.45", linewidth=0.5, alpha=0.35),
         OWN: dict(color=COVERAGE, linewidth=2.0, alpha=0.9),
         SHARED: dict(color=OVERLAP, linewidth=2.6, alpha=0.95)}
PASS_LABEL = {"overlap": "most overlapping", "highest": "highest", "first": "first"}


# ---------------------------------------------------------------- inputs

def default_gateways(slots_path: Path) -> Path:
    """Gateway table matching the dataset variant in the slot file name."""
    tag = slots_path.stem.replace("_", "-")
    for variant, path in VARIANT_GATEWAYS.items():
        if variant in tag:
            return path
    return DEFAULT_GATEWAYS


def select_stations(gateways: pd.DataFrame, queries: list[str] | None, available: set[str]) -> pd.DataFrame:
    """Stations of the slot CSV matching the queries (station id, or case-insensitive part of the name)."""
    stations = gateways[gateways["station_id"].isin(available)]
    if not queries:
        return stations
    keep = pd.Series(False, index=stations.index)
    for query in queries:
        hit = (stations["station_id"] == query) | stations["station_name"].str.contains(query, case=False,
                                                                                         regex=False)
        if not hit.any():
            print(f"Warning: no station of the slot CSV matches {query!r}")
        keep |= hit
    return stations[keep]


def load_slots(slots_path: Path) -> pd.DataFrame:
    slots = pd.concat(pd.read_csv(slots_path, usecols=SLOT_COLUMNS, chunksize=2_000_000), ignore_index=True)
    slots["start"] = pd.to_datetime(slots["start_time"], utc=True)
    slots["end"] = pd.to_datetime(slots["end_time"], utc=True)
    return slots


# ---------------------------------------------------------------- conflicts

def overlapping_pairs(slots: pd.DataFrame) -> pd.DataFrame:
    """Every pair of slots of the same satellite at different stations whose intervals overlap,
    with station_a < station_b and the overlap [start, end]."""
    origin = slots["start"].min()
    s0 = (slots["start"] - origin).dt.total_seconds().to_numpy()
    s1 = (slots["end"] - origin).dt.total_seconds().to_numpy()
    station = slots["station_id"].to_numpy()
    rows = []
    for norad_id, idx in slots.groupby("norad_id").indices.items():
        if len(set(station[idx])) < 2:
            continue
        a0, a1, st = s0[idx], s1[idx], station[idx]
        hit = (a0[:, None] < a1[None, :]) & (a0[None, :] < a1[:, None]) & (st[:, None] != st[None, :])
        i, j = np.nonzero(np.triu(hit, 1))
        if len(i):
            rows.append(pd.DataFrame({"norad_id": norad_id, "slot_a": idx[i], "slot_b": idx[j],
                                      "t0": np.maximum(a0[i], a0[j]), "t1": np.minimum(a1[i], a1[j])}))
    if not rows:
        return pd.DataFrame(columns=["norad_id", "satellite_name", "station_a", "station_b", "slot_id_a",
                                     "slot_id_b", "start", "end"])
    pairs = pd.concat(rows, ignore_index=True)
    swap = station[pairs["slot_a"]] > station[pairs["slot_b"]]
    pairs.loc[swap, ["slot_a", "slot_b"]] = pairs.loc[swap, ["slot_b", "slot_a"]].to_numpy()
    return pd.DataFrame({
        "norad_id": pairs["norad_id"], "satellite_name": slots["satellite_name"].to_numpy()[pairs["slot_a"]],
        "station_a": station[pairs["slot_a"]], "station_b": station[pairs["slot_b"]],
        "slot_id_a": slots["slot_id"].to_numpy()[pairs["slot_a"]],
        "slot_id_b": slots["slot_id"].to_numpy()[pairs["slot_b"]],
        "start": origin + pd.to_timedelta(pairs["t0"], unit="s"),
        "end": origin + pd.to_timedelta(pairs["t1"], unit="s")})

# ---------------------------------------------------------------- tracks

def inside(times: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """True where a time falls inside any of the [start, end] intervals."""
    if len(starts) == 0:
        return np.zeros(len(times), dtype=bool)
    return ((times[:, None] >= starts[None, :]) & (times[:, None] <= ends[None, :])).any(axis=1)


def coloured_runs(lon: np.ndarray, lat: np.ndarray, times: np.ndarray, own: tuple, shared: tuple):
    """Polylines of one sampled track, split where the colour class changes (TRACK, OWN inside the
    station's slots, SHARED inside its overlaps, which win over OWN) or the track crosses the antimeridian."""
    middle = (times[:-1] + times[1:]) / 2
    category = np.full(len(middle), TRACK)
    category[inside(middle, *own)] = OWN
    category[inside(middle, *shared)] = SHARED
    category[np.abs(np.diff(lon)) > 180] = -1
    points = np.column_stack([lon, lat])
    change = np.flatnonzero(np.diff(category) != 0) + 1
    for first, stop in zip(np.r_[0, change], np.r_[change, len(category)]):
        if category[first] >= 0:
            yield int(category[first]), points[first:stop + 1]


def chosen_span(own: pd.DataFrame, shared: pd.DataFrame, half_period: float, horizon_seconds: float,
                choice: str) -> tuple[float, float]:
    """One orbital period (clipped to the horizon) centred on one pass of the satellite over the station.
    `own`: the station's slots on the satellite (s0, s1, parent_window_id, max_elevation_deg);
    `shared`: the station's overlapping slot pairs on it (s0, s1)."""
    passes = own.groupby("parent_window_id").agg(s0=("s0", "min"), s1=("s1", "max"),
                                                 elevation=("max_elevation_deg", "max"))
    if choice == "first":
        best = passes["s0"].idxmin()
    else:
        if choice == "overlap":
            passes["overlaps"] = [int(((shared["s0"] <= b) & (shared["s1"] >= a)).sum())
                                  for a, b in zip(passes["s0"], passes["s1"])]
        else:
            passes["overlaps"] = 0
        best = passes.sort_values(["overlaps", "elevation"], ascending=False).index[0]
    middle = (passes.at[best, "s0"] + passes.at[best, "s1"]) / 2
    return max(0.0, middle - half_period), min(horizon_seconds, middle + half_period)


def station_tracks(own: pd.DataFrame, shared: pd.DataFrame, tles: pd.DataFrame, timescale, origin: pd.Timestamp,
                   horizon_seconds: float, args) -> tuple[dict[int, list], dict[int, tuple[float, float]]]:
    """Coloured polylines of the ground track of every satellite the station has a slot on, grouped by
    colour class, and the drawn time span of each satellite (seconds from `origin`)."""
    runs = {TRACK: [], OWN: [], SHARED: []}
    drawn = {}
    for norad_id, sat_own in own.groupby("norad_id"):
        tle = tles.loc[norad_id]
        orbit = EarthSatellite(tle.tle_line_1, tle.tle_line_2, tle.satellite_name, timescale)
        sat_shared = shared[shared["norad_id"] == norad_id]
        if args.full_horizon:
            a, b = 0.0, horizon_seconds
        else:
            half_period = np.pi / orbit.model.no_kozai * 60  # no_kozai: mean motion [rad/min]
            a, b = chosen_span(sat_own, sat_shared, half_period, horizon_seconds, args.pass_choice)
        drawn[norad_id] = (a, b)
        own_iv = (sat_own["s0"].to_numpy(), sat_own["s1"].to_numpy())
        shared_iv = (sat_shared["s0"].to_numpy(), sat_shared["s1"].to_numpy())
        # regular samples plus every slot / overlap boundary, so short intervals are never skipped
        edges = np.concatenate([*own_iv, *shared_iv])
        times = np.union1d(np.append(np.arange(a, b, args.step_seconds), b), edges[(edges > a) & (edges < b)])
        sky = timescale.utc(origin.year, origin.month, origin.day, origin.hour, origin.minute, origin.second + times)
        point = wgs84.subpoint_of(orbit.at(sky))
        for category, line in coloured_runs(point.longitude.degrees, point.latitude.degrees, times, own_iv,
                                            shared_iv):
            runs[category].append(line)
    return runs, drawn


# ---------------------------------------------------------------- plots

def base_axes(basemap, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(OCEAN_COLOR)
    rings = [[(x + shift, y) for x, y in ring] for shift in (-360, 0, 360) for ring in basemap]
    ax.add_collection(PolyCollection(rings, facecolors=LAND_COLOR, edgecolors=BORDER_COLOR, linewidths=0.4,
                                     zorder=1))
    ax.grid(True, linewidth=0.4, alpha=0.4, zorder=1.5)
    ax.set(xlabel="Longitude [deg]", ylabel="Latitude [deg]")
    return fig, ax


def save(fig, ax, title: str, output_path: Path, dpi: int) -> None:
    ax.set_title(title, fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_station(station, runs: dict, partners: pd.DataFrame, others: pd.DataFrame, basemap, title: str,
                 output_path: Path, dpi: int) -> None:
    fig, ax = base_axes(basemap, (15, 7.8))
    for zorder, category in enumerate((TRACK, OWN, SHARED), start=2):
        if runs[category]:
            ax.add_collection(LineCollection(runs[category], zorder=zorder, **STYLE[category]))
    ax.scatter(others["longitude"], others["latitude"], marker="^", s=26, color="0.55", edgecolors="white",
               linewidths=0.4, zorder=5)
    if len(partners):
        scale = partners["pairs"].to_numpy() / partners["pairs"].max()
        for x, y, s, n in zip(partners["longitude"], partners["latitude"], scale, partners["pairs"]):
            ax.plot([station.longitude, x], [station.latitude, y], color=PARTNER, linewidth=0.8 + 3 * s,
                    alpha=0.6, zorder=5.5, solid_capstyle="round")
            ax.annotate(f"{n}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8, weight="bold",
                        color="black", zorder=8)
        ax.scatter(partners["longitude"], partners["latitude"], marker="^", s=40 + 160 * scale, color=PARTNER,
                   edgecolors="white", linewidths=0.7, zorder=6)
    ax.scatter([station.longitude], [station.latitude], marker="*", s=380, color="black", edgecolors="white",
               linewidths=0.9, zorder=7)
    ax.set(xlim=(-180, 180), ylim=(-90, 90))
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))
    ax.set_aspect("equal")
    handles = [Line2D([], [], **STYLE[TRACK], label="ground track"),
               Line2D([], [], **STYLE[OWN], label=f"{station.station_name}: own slots"),
               Line2D([], [], **STYLE[SHARED],
                      label="overlapping slot at another station on the same satellite"),
               Line2D([], [], marker="^", color=PARTNER, markeredgecolor="white", markersize=9, linestyle="",
                      label="conflicting station (number = overlapping slot pairs, whole horizon)"),
               Line2D([], [], marker="^", color="0.55", markeredgecolor="white", markersize=7, linestyle="",
                      label="other stations"),
               Line2D([], [], marker="*", color="black", markeredgecolor="white", markersize=15, linestyle="",
                      label=station.station_name)]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5, framealpha=0.92)
    save(fig, ax, title, output_path, dpi)


def plot_network(stations: pd.DataFrame, station_pairs: pd.DataFrame, basemap, title: str, output_path: Path,
                 dpi: int, zoom: bool = False) -> None:
    fig, ax = base_axes(basemap, (13, 9) if zoom else (15, 7.8))
    position = stations.set_index("station_id")[["longitude", "latitude"]]
    if len(station_pairs):
        ends = station_pairs.join(position, on="station_a").join(position.add_suffix("_b"), on="station_b")
        order = np.argsort(ends["pairs"].to_numpy())
        norm = plt.matplotlib.colors.LogNorm(1, max(2, ends["pairs"].max()))
        cmap = plt.get_cmap("YlOrRd")
        segments = [np.array([[r.longitude, r.latitude], [r.longitude_b, r.latitude_b]])
                    for r in ends.iloc[order].itertuples()]
        values = ends["pairs"].to_numpy()[order]
        ax.add_collection(LineCollection(segments, colors=cmap(norm(values)),
                                         linewidths=0.8 + 3 * norm(values), zorder=3))
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.75, pad=0.02)
        bar.set_label("overlapping slot pairs between the two stations")
    involved = set(station_pairs["station_a"]) | set(station_pairs["station_b"])
    free = stations[~stations["station_id"].isin(involved)]
    busy = stations[stations["station_id"].isin(involved)]
    ax.scatter(free["longitude"], free["latitude"], marker="^", s=30, color="0.55", edgecolors="white",
               linewidths=0.4, zorder=4, label=f"no simultaneous conflict ({len(free)})")
    ax.scatter(busy["longitude"], busy["latitude"], marker="^", s=45, color=OVERLAP, edgecolors="black",
               linewidths=0.6, zorder=5, label=f"in at least one ({len(busy)})")
    if zoom and len(busy):
        names = busy.set_index("station_id")["station_name"]
        totals = pd.concat([station_pairs.groupby("station_a")["pairs"].sum(),
                            station_pairs.groupby("station_b")["pairs"].sum()]).groupby(level=0).sum()
        for sid, total in totals.items():
            x, y = position.loc[sid]
            ax.annotate(f"{names[sid].split(',')[0]} ({total})", (x, y), xytext=(4, -10),
                        textcoords="offset points", fontsize=7, color="0.15", zorder=6)
        lon0, lon1 = busy["longitude"].min() - 4, busy["longitude"].max() + 4
        lat0, lat1 = busy["latitude"].min() - 3, busy["latitude"].max() + 3
        ax.set(xlim=(lon0, lon1), ylim=(lat0, lat1))
        ax.set_aspect(1 / max(0.2, np.cos(np.radians((lat0 + lat1) / 2))))
    else:
        ax.set(xlim=(-180, 180), ylim=(-90, 90))
        ax.set_xticks(range(-180, 181, 60))
        ax.set_yticks(range(-90, 91, 30))
        ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92)
    save(fig, ax, title, output_path, dpi)


# ---------------------------------------------------------------- main

def draw_dataset(slots_path: Path, gateways_path: Path, tles: pd.DataFrame, basemap, timescale,
                 out_dir: Path, args: argparse.Namespace) -> None:
    print(f"Reading slots from {slots_path}")
    slots = load_slots(slots_path)
    missing = set(slots["norad_id"]) - set(tles.index)
    if missing:
        print(f"Warning: {len(missing)} satellites have no TLE in {args.satellites}; they are skipped")
        slots = slots[~slots["norad_id"].isin(missing)].reset_index(drop=True)
    stale = slots["source_epoch"].ne(slots["norad_id"].map(tles["epoch"]))
    if stale.any():
        print(f"Warning: {slots.loc[stale, 'norad_id'].nunique()} satellites have a different TLE epoch in "
              f"{args.satellites} than in the slot CSV; their tracks may be off (wrong TLE table?)")
    pairs = overlapping_pairs(slots)
    gateways = pd.read_csv(gateways_path)
    all_stations = gateways[gateways["station_id"].isin(set(slots["station_id"]))]
    unknown = set(slots["station_id"]) - set(all_stations["station_id"])
    if unknown:
        print(f"Warning: {len(unknown)} stations of the slot CSV are not in {gateways_path}; they are not drawn")
    names = all_stations.set_index("station_id")["station_name"]
    station_pairs = (pairs.groupby(["station_a", "station_b"]).agg(pairs=("norad_id", "size"),
                                                                     satellites=("norad_id", "nunique"))
                     .reset_index().sort_values("pairs", ascending=False))
    station_pairs.insert(1, "name_a", station_pairs["station_a"].map(names))
    station_pairs.insert(3, "name_b", station_pairs["station_b"].map(names))
    out_dir.mkdir(parents=True, exist_ok=True)
    station_pairs.to_csv(out_dir / "station_pairs.csv", index=False)
    pairs.to_csv(out_dir / "overlapping_pairs.csv", index=False)
    in_pairs = set(pairs["slot_id_a"]) | set(pairs["slot_id_b"])
    print(f"{len(slots)} slots, {len(all_stations)} stations: {len(pairs)} overlapping slot pairs between "
          f"{len(station_pairs)} station pairs; {len(in_pairs)} slots ({100 * len(in_pairs) / len(slots):.1f}%) "
          f"are in at least one")
    network_title = (f"{slots_path.stem}: stations that would contact the same satellite at the same time\n"
                     f"{len(station_pairs)} station pairs, {len(pairs)} overlapping slot pairs, {len(in_pairs)} of "
                     f"{len(slots)} slots ({100 * len(in_pairs) / len(slots):.1f}%) involved")
    plot_network(all_stations, station_pairs, basemap, network_title, out_dir / "network.png", args.dpi)
    plot_network(all_stations, station_pairs, basemap, network_title + "\nzoom on the conflicting stations "
                 "(label: station, overlapping slot pairs)", out_dir / "network_zoom.png", args.dpi, zoom=True)

    # times in seconds from the start of the horizon, for the track sampling
    origin = slots["start"].min().floor("h")
    horizon_seconds = float(np.ceil((slots["end"].max() - origin).total_seconds() / 3600) * 3600)
    for frame in (slots, pairs):
        frame["s0"] = (pd.to_datetime(frame["start"], utc=True) - origin).dt.total_seconds()
        frame["s1"] = (pd.to_datetime(frame["end"], utc=True) - origin).dt.total_seconds()
    drawn_what = ("whole horizon track per satellite" if args.full_horizon
                  else f"one orbit per satellite, around its {PASS_LABEL[args.pass_choice]} pass")

    stations = select_stations(gateways, args.station, set(slots["station_id"]))
    print(f"Drawing {len(stations)} station maps ({drawn_what})")
    index_rows = []
    for number, station in enumerate(stations.itertuples(index=False), start=1):
        sid = station.station_id
        own = slots[slots["station_id"] == sid]
        mine = pairs[(pairs["station_a"] == sid) | (pairs["station_b"] == sid)]
        partner_ids = np.where(mine["station_a"] == sid, mine["station_b"], mine["station_a"])
        counts = pd.Series(partner_ids).value_counts().rename("pairs")
        partners = all_stations.join(counts, on="station_id", how="inner").sort_values("pairs", ascending=False)
        others = all_stations[~all_stations["station_id"].isin(set(partners["station_id"]) | {sid})]
        clock = time.perf_counter()
        runs, drawn = station_tracks(own, mine, tles, timescale, origin, horizon_seconds, args)
        track_seconds = time.perf_counter() - clock
        span = mine["norad_id"].map(drawn)
        shown = sum(a <= s1 and s0 <= b for (a, b), s0, s1 in zip(span, mine["s0"], mine["s1"]))
        own_hit = set(mine["slot_id_a"]) | set(mine["slot_id_b"])
        own_hit &= set(own["slot_id"])
        top = ", ".join(f"{n} ({c})" for n, c in zip(partners["station_name"].head(4), partners["pairs"].head(4)))
        title = (f"{slots_path.stem}: {station.station_name} ({sid}) - {len(own)} slots on "
                 f"{own['norad_id'].nunique()} satellites ({drawn_what})\n{len(own_hit)} slots "
                 f"({100 * len(own_hit) / max(1, len(own)):.0f}%) overlap in time with a slot of the same satellite "
                 f"at another station: {len(mine)} pairs with {len(partners)} stations, {shown} of them on the "
                 f"drawn orbits" + (f"\n{top}" if top else ""))
        safe = "".join(c if c.isalnum() else "_" for c in station.station_name).strip("_")
        name = f"{sid}_{safe}.png"
        plot_station(station, runs, partners, others, basemap, title, out_dir / name, args.dpi)
        index_rows.append({"station_id": sid, "station_name": station.station_name, "image": name,
                           "slots": len(own), "satellites": own["norad_id"].nunique(),
                           "overlapping_slots": len(own_hit), "overlapping_pairs": len(mine),
                           "conflicting_stations": len(partners), "pairs_on_drawn_orbits": shown})
        print(f"  [{number}/{len(stations)}] {station.station_name} ({sid}): {len(own)} slots, "
              f"{own['norad_id'].nunique()} satellites, {len(mine)} overlapping pairs with {len(partners)} stations "
              f"({shown} on the drawn orbits); tracks {track_seconds:.1f} s")
    if index_rows:
        pd.DataFrame(index_rows).to_csv(out_dir / "stations_index.csv", index=False)
    print(f"Saved network.png, network_zoom.png, {len(index_rows)} station maps and the CSVs to {out_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("slots", type=Path, help="path to the slot CSV")
    parser.add_argument("--station", nargs="+", default=None,
                        help="station ids or parts of station names (default: every station of the slot CSV)")
    parser.add_argument("--satellites", type=Path, default=DEFAULT_SATELLITES,
                        help="TLE table the slot CSV was built from (default: %(default)s)")
    parser.add_argument("--gateways", type=Path, default=None,
                        help="station table with coordinates (default: picked from the slot file name)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: visualization/plot_station_overlaps/<slot file stem>/")
    parser.add_argument("--pass", dest="pass_choice", choices=list(PASS_LABEL), default="overlap",
                        help="the pass each satellite's orbit is centred on: the one with the most overlapping slot "
                             "pairs (default), the highest max elevation, or the earliest")
    parser.add_argument("--full-horizon", action="store_true",
                        help="draw each satellite over the whole horizon instead of one orbit")
    parser.add_argument("--step-seconds", type=float, default=10,
                        help="track sampling step (slot and overlap boundaries are always sampled)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")  # progress at once, readable station names
    if not args.slots.is_file():
        parser.error(f"slot CSV not found: {args.slots}")
    for path, flag in ((args.satellites, "--satellites"), (args.gateways, "--gateways")):
        if path is not None and not path.is_file():
            parser.error(f"{flag} file not found: {path}")

    tles = pd.read_csv(args.satellites).drop_duplicates("norad_id").set_index("norad_id")
    basemap = load_country_outlines()
    timescale = load.timescale()
    out_dir = args.out_dir or DEFAULT_OUT_DIR / args.slots.stem
    draw_dataset(args.slots, args.gateways or default_gateways(args.slots), tles, basemap, timescale, out_dir, args)


if __name__ == "__main__":
    main()
