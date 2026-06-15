"""
API to convert wrangled data into model objects.
"""

from candidate_slot_data_model import GroundStation, Satellite, Antenna, CandidateSlot
from typing import cast
import pandas as pd
from pathlib import Path


def load_satellites(path: Path) -> list[Satellite]:
    """
    Load all satellites from the data source.

    ------
    > path: path to the data source
    """
    df = pd.read_csv(path, usecols=["satellite_name"], delimiter=",")
    return [Satellite(id=name) for name in df["satellite_name"].unique()]


def load_ground_station(path: Path) -> list[GroundStation]:
    """
    Load all ground stations from the data source.

    ------
    > path: path to the data source
    """
    df = pd.read_csv(path, usecols=["station_id", "station_name"], delimiter=",")
    return [
        GroundStation(id=gs_id, name=gs_name)
        for gs_id, gs_name in df[["station_id", "station_name"]].drop_duplicates().itertuples(index=False)
    ]


def load_antenna(
    path: Path,
    gs_dict: dict[str, GroundStation] | None = None,
) -> list[Antenna]:
    """
    Load all antennas from the data source.

    ------
    > path: path to the data source
    > gs_dict: collection of GroundStation indexed by station id; built from path if None
    """
    df = pd.read_csv(
        path,
        usecols=["antenna_id", "station_id", "reconfiguration_time_hours", "station_name"],
        delimiter=",",
    )
    if gs_dict is None:
        gs_dict = {
            str(row.station_id): GroundStation(id=str(row.station_id), name=str(row.station_name))
            for row in df[["station_id", "station_name"]].drop_duplicates().itertuples(index=False)
        }

    ant_list: list[Antenna] = []
    for ant_id, gs_id, rec_t in df[["antenna_id", "station_id", "reconfiguration_time_hours"]].drop_duplicates().itertuples(index=False):
        ant_list.append(Antenna(
            id=str(ant_id),
            description=str(ant_id),
            station=gs_dict[str(gs_id)],
            reconfiguration_time=cast(float, rec_t),
        ))
    return ant_list


def load_candidateslot(
    path: Path,
    ant_dict: dict[str, Antenna] | None = None,
    sat_dict: dict[str, Satellite] | None = None,
) -> list[CandidateSlot]:
    """
    Load all candidate slots from the data source.

    ------
    > path: path to the data source
    > ant_dict: collection of Antenna indexed by antenna id; built from path if None
    > sat_dict: collection of Satellite indexed by satellite name; built from path if None
    """
    df = pd.read_csv(path, delimiter=",")

    if sat_dict is None:
        sat_dict = {str(name): Satellite(id=str(name)) for name in df["satellite_name"].unique()}
    if ant_dict is None:
        gs_dict: dict[str, GroundStation] = {
            str(row.station_id): GroundStation(id=str(row.station_id), name=str(row.station_name))
            for row in df[["station_id", "station_name"]].drop_duplicates().itertuples(index=False)
        }
        ant_dict = {
            str(row.antenna_id): Antenna(
                id=str(row.antenna_id),
                description=str(row.antenna_id),
                station=gs_dict[str(row.station_id)],
                reconfiguration_time=cast(float, row.reconfiguration_time_hours),
            )
            for row in df[["antenna_id", "station_id", "reconfiguration_time_hours"]].drop_duplicates().itertuples(index=False)
        }

    cs_list: list[CandidateSlot] = []
    col_filter = ["slot_id", "source_interval_id", "satellite_name", "antenna_id", "start_time_hours", "end_time_hours", "priority", "urgency"]
    for cs_id, sint_id, sat_id, ant_id, start_t, end_t, prior, urg in df[col_filter].drop_duplicates().itertuples(index=False):
        cs_list.append(CandidateSlot(
            slot_id=cs_id,
            source_interval_id=sint_id,
            satellite=sat_dict[sat_id],
            antenna=ant_dict[ant_id],
            start_time=start_t,
            end_time=end_t,
            priority=prior,
            urgency=urg,
        ))
    return cs_list
