"""Data models for raw visibility-window inputs and scheduling horizons."""

from dataclasses import dataclass


@dataclass
class SchedulingHorizon:
    """Represent the global scheduling horizon in hours."""

    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        """Return the horizon duration in hours."""
        return self.end_time - self.start_time


@dataclass
class RawVisibilityInterval:
    """Represent one raw visibility window before discretization.

    A raw visibility interval says that a satellite can potentially communicate
    through an antenna during a visibility window. It is not yet a concrete
    schedulable contact because the required communication duration can be
    placed at different start times within the visibility window.
    """

    # Identity fields
    interval_id: str
    satellite_id: str
    antenna_id: str
    station_id: str

    # Visibility-window fields, expressed in hours from the horizon start
    visibility_start: float
    visibility_end: float

    # Communication requirement, expressed in hours
    required_duration: float

    # Scoring input fields copied into generated candidate slots
    priority: float
    urgency: float

    @property
    def visibility_duration(self) -> float:
        """Return the raw visibility-window duration in hours."""
        return self.visibility_end - self.visibility_start

    def as_dict(self) -> dict[str, float | str]:
        """Return a compact dictionary for readable notebook display."""
        return {
            "interval_id": self.interval_id,
            "satellite_id": self.satellite_id,
            "antenna_id": self.antenna_id,
            "station_id": self.station_id,
            "visibility_start": self.visibility_start,
            "visibility_end": self.visibility_end,
            "visibility_duration": self.visibility_duration,
            "required_duration": self.required_duration,
            "priority": self.priority,
            "urgency": self.urgency,
        }
