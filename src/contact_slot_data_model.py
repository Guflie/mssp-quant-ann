from dataclasses import dataclass, field

@dataclass
class GroundStation:
    """Represent one ground station."""

    id: str
    name: str

@dataclass
class Satellite:
    """Represent one satellite."""

    id: str
    norad_id: int

@dataclass
class Antenna:
    """
    Represent one ground-station antenna. Multiple antennas might be located in the 
    same GroundStation. 
    """

    id: str
    station: GroundStation
    reconfiguration_time: float
    capacity: int

@dataclass
class ContactSlot:
    """
    A candidate slot is one possible assignment of a satellite to an antenna
    at a ground station over a discrete time interval. 
    """

    # Identity fields
    slot_id: str
    window_id: str
    satellite: Satellite
    antenna: Antenna

    # Time interval fields, expressed in hours from the scheduling horizon start
    start_time: float
    duration: float

    # Stored optimization value, computed once coefficients are chosen
    weight: float

    @property
    def satellite_id(self) -> str:
        """Return the satellite identifier."""
        return self.satellite.id

    @property
    def antenna_id(self) -> str:
        """Return the antenna identifier."""
        return self.antenna.id

    @property
    def station_id(self) -> str:
        """Return the ground-station identifier through the antenna."""
        return self.antenna.station.id

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    def as_dict(self) -> dict[str, float | str]:
        """Return a compact dictionary for readable notebook display."""
        return {
            "slot_id": self.slot_id,
            "parent_widnow_id": self.window_id,
            "satellite_id": self.satellite_id,
            "antenna_id": self.antenna_id,
            "station_id": self.station_id,
            "reconfiguration_time": self.antenna.reconfiguration_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "weight": self.weight,
        }
