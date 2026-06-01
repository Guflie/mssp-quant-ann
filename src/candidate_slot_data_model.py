from dataclasses import dataclass, field

# TODO looks a little redudant, shoudl we merge it in the 
#      raw_visibility_interval_data_model?

@dataclass
class GroundStation:
    """Represent one ground station."""

    id: str
    name: str

@dataclass
class Satellite:
    """Represent one satellite."""

    id: str

@dataclass
class Antenna:
    """
    Represent one ground-station antenna. Multiple antennas might be located in the 
    same GroundStation. 
    """

    id: str
    description: str
    station: GroundStation
    reconfiguration_time: float

@dataclass
class CandidateSlot:
    """
    A candidate slot is one possible assignment of a satellite to an antenna
    at a ground station over a continuous time interval. 
    """

    # Identity fields
    slot_id: str
    source_interval_id: str
    satellite: Satellite
    antenna: Antenna

    # Time interval fields, expressed in hours from the scheduling horizon start
    start_time: float
    end_time: float

    # Scoring input fields
    priority: float
    urgency: float

    # Stored optimization value, computed once coefficients are chosen
    weight: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        """Validate the basic time interval after initialization."""
        if self.end_time <= self.start_time:
            raise ValueError(
                f"Slot {self.slot_id} has invalid times: "
                f"start_time={self.start_time}, end_time={self.end_time}."
            )

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
    def duration(self) -> float:
        """Return the slot duration in hours."""
        return self.end_time - self.start_time

    def compute_weight(self, alpha: float, beta: float, gamma: float) -> float:
        """Compute and store the weighted value of the slot.

        The weight formula is:

            weight = alpha * priority + beta * duration + gamma * urgency
        """
        # Keep the formula explicit so the scoring model is easy to inspect.
        self.weight = alpha * self.priority + beta * self.duration + gamma * self.urgency
        return self.weight

    def as_dict(self) -> dict[str, float | str]:
        """Return a compact dictionary for readable notebook display."""
        return {
            "slot_id": self.slot_id,
            "source_interval_id": self.source_interval_id,
            "satellite_id": self.satellite_id,
            "antenna_id": self.antenna_id,
            "station_id": self.station_id,
            "reconfiguration_time": self.antenna.reconfiguration_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "priority": self.priority,
            "urgency": self.urgency,
            "weight": self.weight,
        }
