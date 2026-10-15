"""
Data models, enums, and schemas for the WALL-E and EVE's Signal simulation.
Enforces strict typing and validation across all components.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator


class ActionType(str, Enum):
    MOVE_NORTH = "MOVE_NORTH"
    MOVE_SOUTH = "MOVE_SOUTH"
    MOVE_EAST = "MOVE_EAST"
    MOVE_WEST = "MOVE_WEST"
    SCAN = "SCAN"
    WAIT = "WAIT"
    RECHARGE = "RECHARGE"
    RETURN_TO_BASE = "RETURN_TO_BASE"
    REPORT_FOUND = "REPORT_FOUND"


class EveStatus(str, Enum):
    OPERATIONAL = "OPERATIONAL"
    STANDBY = "STANDBY"
    RECHARGING = "RECHARGING"
    CRITICAL = "CRITICAL"
    COMPROMISED = "COMPROMISED"
    DORMANT = "DORMANT"


class MissionStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    SUCCESS = "SUCCESS"
    FAILED_BATTERY_DEPLETED = "FAILED_BATTERY_DEPLETED"
    FAILED_TIMEOUT = "FAILED_TIMEOUT"
    FAILED_INCORRECT_REPORT = "FAILED_INCORRECT_REPORT"
    FAILED_ABORTED = "FAILED_ABORTED"


class ClueType(str, Enum):
    BEACON_PING = "BEACON_PING"
    ION_TRAIL = "ION_TRAIL"
    STATUS_LOG = "STATUS_LOG"
    SECTOR_CLEAR = "SECTOR_CLEAR"


class SignalReading(BaseModel):
    """Signal transmission reading with path-loss strength and directional bearing."""
    strength: float = Field(..., description="Signal strength RSSI / normalized [0.0, 1.0]")
    bearing: float = Field(..., description="Estimated bearing towards source in degrees [0, 360)")
    noisy_distance: float = Field(..., description="Estimated distance based on signal attenuation")
    timestamp: int = Field(..., description="Simulation tick when reading was observed")


class Clue(BaseModel):
    """Environmental clue providing evidence about EVE's location or status."""
    clue_id: str
    clue_type: ClueType
    description: str
    data: Dict[str, Any] = Field(default_factory=dict)
    # Coordinates where clue is situated in the environment (internal ground truth)
    coordinates: Optional[Tuple[int, int]] = None

    def public_copy(self) -> Clue:
        """Return a copy of the clue without internal ground-truth coordinates."""
        return Clue(
            clue_id=self.clue_id,
            clue_type=self.clue_type,
            description=self.description,
            data=dict(self.data),
            coordinates=None
        )


class Observation(BaseModel):
    """
    State observable by the agent at the current step.
    Strictly excludes ground-truth coordinates or hidden environmental state.
    """
    position: Tuple[int, int]
    battery_remaining: float
    battery_capacity: float
    time_remaining: int
    step_count: int
    signal_reading: Optional[SignalReading] = None
    discovered_clues: List[Clue] = Field(default_factory=list)
    local_surroundings: Dict[str, bool] = Field(
        default_factory=lambda: {"north": True, "south": True, "east": True, "west": True}
    )
    is_at_recharge_station: bool = False
    is_terminal: bool = False
    termination_reason: Optional[str] = None
    action_result_message: str = "Initialized"


class AgentAction(BaseModel):
    """Action selected by the autonomous agent."""
    action_type: ActionType
    reported_location: Optional[Tuple[int, int]] = None
    reported_status: Optional[EveStatus] = None


class ActionCostConfig(BaseModel):
    """Resource consumption parameters."""
    move_battery_cost: float = Field(default=1.0, ge=0.0)
    move_time_cost: int = Field(default=1, ge=1)
    scan_battery_cost: float = Field(default=3.0, ge=0.0)
    scan_time_cost: int = Field(default=2, ge=1)
    wait_battery_cost: float = Field(default=0.2, ge=0.0)
    wait_time_cost: int = Field(default=1, ge=1)
    recharge_battery_rate: float = Field(default=20.0, ge=1.0)
    recharge_time_cost: int = Field(default=1, ge=1)


class SensorConfig(BaseModel):
    """Sensor range and noise characteristics."""
    scan_radius: int = Field(default=3, ge=1)
    passive_clue_radius: int = Field(default=1, ge=0)
    signal_noise_std: float = Field(default=0.35, ge=0.01)
    bearing_noise_std: float = Field(default=12.0, ge=0.1)


class MissionConfig(BaseModel):
    """
    Complete configuration specification for a simulation run.
    Contains map geometry, resource budget, clues, and ground truth.
    """
    grid_width: int = Field(..., ge=3, description="Width of map grid")
    grid_height: int = Field(..., ge=3, description="Height of map grid")
    walle_start: Tuple[int, int] = Field(..., description="WALL-E start coordinate (x, y)")
    recharge_stations: List[Tuple[int, int]] = Field(default_factory=list)
    obstacles: List[Tuple[int, int]] = Field(default_factory=list)
    initial_signal: Optional[Dict[str, Any]] = None
    battery_capacity: float = Field(default=100.0, gt=0.0)
    starting_battery: float = Field(default=100.0, gt=0.0)
    time_budget: int = Field(default=120, gt=0)
    action_costs: ActionCostConfig = Field(default_factory=ActionCostConfig)
    sensor_config: SensorConfig = Field(default_factory=SensorConfig)
    clues: List[Dict[str, Any]] = Field(default_factory=list)
    eve_location: Tuple[int, int] = Field(..., description="Ground truth EVE location (x, y)")
    eve_status: EveStatus = Field(default=EveStatus.OPERATIONAL, description="Ground truth EVE status")
    confidence_threshold: float = Field(default=0.75, ge=0.1, le=1.0)
    max_steps: int = Field(default=150, ge=5)
    random_seed: Optional[int] = None

    @field_validator("walle_start", "eve_location")
    @classmethod
    def validate_coord(cls, coord: Tuple[int, int]) -> Tuple[int, int]:
        if len(coord) != 2:
            raise ValueError(f"Coordinate {coord} must be an (x, y) pair.")
        return coord

    @model_validator(mode="after")
    def validate_geometry_and_boundaries(self) -> MissionConfig:
        w, h = self.grid_width, self.grid_height
        obstacle_set = set(self.obstacles)

        # Check bounds
        for name, coord in [("walle_start", self.walle_start), ("eve_location", self.eve_location)]:
            x, y = coord
            if not (0 <= x < w and 0 <= y < h):
                raise ValueError(f"{name} {coord} is outside grid bounds ({w}x{h}).")
            if coord in obstacle_set:
                raise ValueError(f"{name} {coord} cannot be placed on an obstacle.")

        for station in self.recharge_stations:
            sx, sy = station
            if not (0 <= sx < w and 0 <= sy < h):
                raise ValueError(f"Recharge station {station} is outside grid bounds ({w}x{h}).")
            if station in obstacle_set:
                raise ValueError(f"Recharge station {station} cannot be placed on an obstacle.")

        for ox, oy in self.obstacles:
            if not (0 <= ox < w and 0 <= oy < h):
                raise ValueError(f"Obstacle ({ox}, {oy}) is outside grid bounds ({w}x{h}).")

        if self.starting_battery > self.battery_capacity:
            raise ValueError(
                f"starting_battery ({self.starting_battery}) cannot exceed battery_capacity ({self.battery_capacity})."
            )

        return self


class TraceStep(BaseModel):
    """Structured record of an individual simulation step for auditing and replay."""
    step_number: int
    agent_position: Tuple[int, int]
    chosen_action: str
    action_args: Dict[str, Any] = Field(default_factory=dict)
    battery_remaining: float
    time_remaining: int
    action_result: str
    observation_summary: Dict[str, Any]
    belief_summary: Dict[str, Any]
    estimated_best_candidate: Tuple[int, int]
    estimated_best_probability: float
    inferred_status: str
    status_confidence: float
    entropy: float
    decision_explanation: str


class MissionReport(BaseModel):
    """Final mission evaluation report produced at termination."""
    mission_status: MissionStatus
    eve_found: bool
    termination_reason: str
    total_steps: int
    scans_performed: int
    battery_consumed: float
    time_consumed: int
    battery_remaining: float
    time_remaining: int
    final_inferred_location: Tuple[int, int]
    final_inferred_status: EveStatus
    location_confidence: float
    status_confidence: float
    ground_truth_location: Tuple[int, int]
    ground_truth_status: EveStatus
    location_error_distance: float
    matches_ground_truth: bool
    evidence_gathered: List[str]
    recommended_next_action: str
