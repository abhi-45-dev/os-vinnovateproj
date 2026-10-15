"""
Simulated grid environment for WALL-E and EVE's Signal.
Enforces physical boundaries, obstacles, resource budgets, sensor noise,
and strict encapsulation of ground-truth data.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from wall_e_agent.models import (
    ActionCostConfig,
    ActionType,
    AgentAction,
    Clue,
    ClueType,
    EveStatus,
    MissionConfig,
    MissionStatus,
    Observation,
    SensorConfig,
    SignalReading,
)


class GridEnvironment:
    """
    Simulation environment modeling the map, obstacles, resource constraints,
    environmental clues, and sensor readings.
    """

    def __init__(self, config: MissionConfig):
        self.config = config
        self.width = config.grid_width
        self.height = config.grid_height
        self.obstacles: Set[Tuple[int, int]] = set(config.obstacles)
        self.recharge_stations: Set[Tuple[int, int]] = set(config.recharge_stations)

        # Ground truth state (isolated from agent observation)
        self._eve_location: Tuple[int, int] = config.eve_location
        self._eve_status: EveStatus = config.eve_status

        # Internal clues with actual coordinates
        self._internal_clues: List[Clue] = []
        for c in config.clues:
            coord = tuple(c.get("coordinates", [0, 0]))  # type: ignore
            clue = Clue(
                clue_id=c.get("clue_id", f"clue_{len(self._internal_clues)}"),
                clue_type=ClueType(c.get("clue_type", ClueType.BEACON_PING)),
                description=c.get("description", ""),
                data=c.get("data", {}),
                coordinates=coord,
            )
            self._internal_clues.append(clue)

        # PRNG instance for reproducibility
        self.rng = random.Random(config.random_seed)

        # Dynamic simulation state
        self.agent_position: Tuple[int, int] = config.walle_start
        self.battery: float = config.starting_battery
        self.time_remaining: int = config.time_budget
        self.step_count: int = 0
        self.scans_performed: int = 0
        self.movement_actions: int = 0
        self.invalid_actions: int = 0
        self.discovered_clue_ids: Set[str] = set()

        self.is_terminal: bool = False
        self.mission_status: MissionStatus = MissionStatus.IN_PROGRESS
        self.termination_reason: Optional[str] = None
        self.last_action_result: str = "Mission initialized"

    def reset(self) -> Observation:
        """Reset the simulation environment to initial configuration."""
        self.rng = random.Random(self.config.random_seed)
        self.agent_position = self.config.walle_start
        self.battery = self.config.starting_battery
        self.time_remaining = self.config.time_budget
        self.step_count = 0
        self.scans_performed = 0
        self.movement_actions = 0
        self.invalid_actions = 0
        self.discovered_clue_ids = set()

        self.is_terminal = False
        self.mission_status = MissionStatus.IN_PROGRESS
        self.termination_reason = None
        self.last_action_result = "Mission initialized"

        # Check for initial clues at start position
        new_clues = self._check_clue_discovery(radius=self.config.sensor_config.passive_clue_radius)
        initial_signal = self._generate_signal_reading()

        return self._build_observation(
            signal_reading=initial_signal,
            discovered_clues=new_clues,
            message="Mission ready. Initial sensor readings recorded."
        )

    def step(self, action: AgentAction) -> Observation:
        """
        Execute an agent action in the environment and return the resulting observation.
        """
        if self.is_terminal:
            return self._build_observation(
                message=f"Simulation already terminated: {self.termination_reason}"
            )

        self.step_count += 1
        action_msg = ""
        costs = self.config.action_costs
        new_clues: List[Clue] = []
        signal_reading: Optional[SignalReading] = None

        if action.action_type in (
            ActionType.MOVE_NORTH,
            ActionType.MOVE_SOUTH,
            ActionType.MOVE_EAST,
            ActionType.MOVE_WEST,
        ):
            action_msg = self._handle_move(action.action_type, costs)
            new_clues = self._check_clue_discovery(radius=self.config.sensor_config.passive_clue_radius)
            # Ambient noisy ping while moving
            signal_reading = self._generate_signal_reading(noise_multiplier=1.2)

        elif action.action_type == ActionType.SCAN:
            action_msg = self._handle_scan(costs)
            new_clues = self._check_clue_discovery(radius=self.config.sensor_config.scan_radius)
            # If in direct proximity to EVE during scan, observe diagnostic status
            ax, ay = self.agent_position
            ex, ey = self._eve_location
            if math.hypot(ex - ax, ey - ay) <= 1.5:
                direct_clue = Clue(
                    clue_id="direct_eve_diagnostic",
                    clue_type=ClueType.STATUS_LOG,
                    description=f"Direct sensor scan confirms EVE diagnostic status: {self._eve_status.value}",
                    data={"status_fragment": self._eve_status.value, "confidence": 1.0},
                    coordinates=self._eve_location,
                )
                if direct_clue.clue_id not in self.discovered_clue_ids:
                    self.discovered_clue_ids.add(direct_clue.clue_id)
                    new_clues.append(direct_clue.public_copy())
            # High-precision scan reading
            signal_reading = self._generate_signal_reading(noise_multiplier=0.6)

        elif action.action_type == ActionType.WAIT:
            action_msg = self._handle_wait(costs)
            signal_reading = self._generate_signal_reading(noise_multiplier=1.0)

        elif action.action_type == ActionType.RECHARGE:
            action_msg = self._handle_recharge(costs)

        elif action.action_type == ActionType.RETURN_TO_BASE:
            action_msg = self._handle_return_to_base(costs)
            new_clues = self._check_clue_discovery(radius=self.config.sensor_config.passive_clue_radius)
            signal_reading = self._generate_signal_reading(noise_multiplier=1.0)

        elif action.action_type == ActionType.REPORT_FOUND:
            action_msg = self._handle_report_found(action)

        else:
            self.invalid_actions += 1
            action_msg = f"Unknown action: {action.action_type}"

        # Resource depletion checks
        if not self.is_terminal:
            if self.battery <= 0.0:
                self.battery = 0.0
                self.is_terminal = True
                self.mission_status = MissionStatus.FAILED_BATTERY_DEPLETED
                self.termination_reason = "WALL-E battery fully depleted."
                action_msg += " [CRITICAL: Battery depleted]"
            elif self.time_remaining <= 0:
                self.time_remaining = 0
                self.is_terminal = True
                self.mission_status = MissionStatus.FAILED_TIMEOUT
                self.termination_reason = "Mission time budget expired."
                action_msg += " [CRITICAL: Time expired]"
            elif self.step_count >= self.config.max_steps:
                self.is_terminal = True
                self.mission_status = MissionStatus.FAILED_TIMEOUT
                self.termination_reason = f"Max step limit reached ({self.config.max_steps})."
                action_msg += " [CRITICAL: Step limit reached]"

        self.last_action_result = action_msg
        return self._build_observation(
            signal_reading=signal_reading,
            discovered_clues=new_clues,
            message=action_msg
        )

    def _handle_move(self, action_type: ActionType, costs: ActionCostConfig) -> str:
        self.battery -= costs.move_battery_cost
        self.time_remaining -= costs.move_time_cost
        self.movement_actions += 1

        dx, dy = 0, 0
        if action_type == ActionType.MOVE_NORTH:
            dy = 1
        elif action_type == ActionType.MOVE_SOUTH:
            dy = -1
        elif action_type == ActionType.MOVE_EAST:
            dx = 1
        elif action_type == ActionType.MOVE_WEST:
            dx = -1

        new_x = self.agent_position[0] + dx
        new_y = self.agent_position[1] + dy

        # Check boundaries
        if not (0 <= new_x < self.width and 0 <= new_y < self.height):
            self.invalid_actions += 1
            return f"Blocked: Move {action_type.value} hit boundary at ({new_x}, {new_y})."

        # Check obstacles
        if (new_x, new_y) in self.obstacles:
            self.invalid_actions += 1
            return f"Blocked: Move {action_type.value} hit obstacle at ({new_x}, {new_y})."

        self.agent_position = (new_x, new_y)
        return f"Moved {action_type.value} to {self.agent_position}."

    def _handle_scan(self, costs: ActionCostConfig) -> str:
        self.battery -= costs.scan_battery_cost
        self.time_remaining -= costs.scan_time_cost
        self.scans_performed += 1
        return f"Scanned radius {self.config.sensor_config.scan_radius} from {self.agent_position}."

    def _handle_wait(self, costs: ActionCostConfig) -> str:
        self.battery -= costs.wait_battery_cost
        self.time_remaining -= costs.wait_time_cost
        return f"Waited at {self.agent_position}. Power conservation mode active."

    def _handle_recharge(self, costs: ActionCostConfig) -> str:
        self.time_remaining -= costs.recharge_time_cost
        if self.agent_position in self.recharge_stations:
            gained = min(costs.recharge_battery_rate, self.config.battery_capacity - self.battery)
            self.battery += gained
            return f"Recharged +{gained:.1f} battery at station {self.agent_position}. (Current: {self.battery:.1f})"
        else:
            # Idle discharge if attempting recharge away from station
            self.battery -= costs.wait_battery_cost
            self.invalid_actions += 1
            return f"Recharge failed: No charging station at current position {self.agent_position}."

    def _handle_return_to_base(self, costs: ActionCostConfig) -> str:
        """Step one unit along shortest path toward closest recharge station / base."""
        if not self.recharge_stations:
            self.invalid_actions += 1
            return "No recharge stations configured."

        if self.agent_position in self.recharge_stations:
            return f"Already at recharge base {self.agent_position}."

        # Find closest station via BFS
        path = self._find_path_to_nearest_station()
        if path and len(path) > 1:
            next_pos = path[1]
            self.battery -= costs.move_battery_cost
            self.time_remaining -= costs.move_time_cost
            self.movement_actions += 1
            self.agent_position = next_pos
            return f"Returning to base: stepped to {self.agent_position}."
        else:
            self.invalid_actions += 1
            return "No navigable path to recharge base found."

    def _handle_report_found(self, action: AgentAction) -> str:
        """Evaluate agent's report against ground truth."""
        self.is_terminal = True
        rep_loc = action.reported_location
        rep_stat = action.reported_status

        if rep_loc is None:
            self.mission_status = MissionStatus.FAILED_INCORRECT_REPORT
            self.termination_reason = "REPORT_FOUND executed without candidate location."
            return "Failed: Reported location is empty."

        loc_match = (rep_loc == self._eve_location)
        status_match = (rep_stat == self._eve_status if rep_stat else True)

        if loc_match and status_match:
            self.mission_status = MissionStatus.SUCCESS
            self.termination_reason = f"EVE successfully located at {rep_loc} with status {rep_stat.value if rep_stat else 'UNKNOWN'}."
            return f"SUCCESS: EVE found at {rep_loc}."
        else:
            self.mission_status = MissionStatus.FAILED_INCORRECT_REPORT
            self.termination_reason = (
                f"Reported {rep_loc} ({rep_stat}) does not match ground truth "
                f"(Location match: {loc_match}, Status match: {status_match})."
            )
            return f"FAILED: Reported {rep_loc} is incorrect."

    def _check_clue_discovery(self, radius: int) -> List[Clue]:
        """Check if any undiscovered clues are within the given radius."""
        new_clues: List[Clue] = []
        ax, ay = self.agent_position

        for clue in self._internal_clues:
            if clue.clue_id in self.discovered_clue_ids:
                continue
            if clue.coordinates is not None:
                cx, cy = clue.coordinates
                dist = math.hypot(cx - ax, cy - ay)
                if dist <= radius:
                    self.discovered_clue_ids.add(clue.clue_id)
                    # Strip internal coordinates before giving to agent
                    new_clues.append(clue.public_copy())
        return new_clues

    def _generate_signal_reading(self, noise_multiplier: float = 1.0) -> SignalReading:
        """Generate noisy signal observation from EVE's actual transmitter."""
        ax, ay = self.agent_position
        ex, ey = self._eve_location

        true_distance = math.hypot(ex - ax, ey - ay)
        true_angle = math.degrees(math.atan2(ey - ay, ex - ax)) % 360.0

        s_cfg = self.config.sensor_config
        dist_noise = self.rng.gauss(0, s_cfg.signal_noise_std * noise_multiplier)
        bearing_noise = self.rng.gauss(0, s_cfg.bearing_noise_std * noise_multiplier)

        noisy_dist = max(0.0, true_distance + dist_noise)
        noisy_bearing = (true_angle + bearing_noise) % 360.0

        # Path-loss model: strength drops with distance squared
        # S = 1 / (1 + 0.08 * d^2) + noise
        attenuation = 1.0 / (1.0 + 0.08 * (noisy_dist ** 2))
        strength_noise = self.rng.gauss(0, 0.04 * noise_multiplier)
        strength = max(0.01, min(1.0, attenuation + strength_noise))

        return SignalReading(
            strength=round(strength, 4),
            bearing=round(noisy_bearing, 2),
            noisy_distance=round(noisy_dist, 2),
            timestamp=self.step_count,
        )

    def _get_local_surroundings(self) -> Dict[str, bool]:
        """Check 4-directional step validity from current position."""
        x, y = self.agent_position
        directions = {
            "north": (x, y + 1),
            "south": (x, y - 1),
            "east": (x + 1, y),
            "west": (x - 1, y),
        }
        res = {}
        for d, (nx, ny) in directions.items():
            valid = (0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in self.obstacles)
            res[d] = valid
        return res

    def _find_path_to_nearest_station(self) -> Optional[List[Tuple[int, int]]]:
        """BFS pathfinding to closest recharge station."""
        if not self.recharge_stations:
            return None

        queue = [[self.agent_position]]
        visited = {self.agent_position}

        while queue:
            path = queue.pop(0)
            curr = path[-1]
            if curr in self.recharge_stations:
                return path

            cx, cy = curr
            for nx, ny in [(cx, cy + 1), (cx, cy - 1), (cx + 1, cy), (cx - 1, cy)]:
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if (nx, ny) not in self.obstacles and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append(path + [(nx, ny)])
        return None

    def _build_observation(
        self,
        signal_reading: Optional[SignalReading] = None,
        discovered_clues: Optional[List[Clue]] = None,
        message: str = ""
    ) -> Observation:
        """Construct the sanitized observation payload for the agent."""
        return Observation(
            position=self.agent_position,
            battery_remaining=round(self.battery, 2),
            battery_capacity=self.config.battery_capacity,
            time_remaining=self.time_remaining,
            step_count=self.step_count,
            signal_reading=signal_reading,
            discovered_clues=discovered_clues or [],
            local_surroundings=self._get_local_surroundings(),
            is_at_recharge_station=(self.agent_position in self.recharge_stations),
            is_terminal=self.is_terminal,
            termination_reason=self.termination_reason,
            action_result_message=message,
        )

    # Evaluation-only interface (used post-mission by MissionReporter)
    def get_evaluation_ground_truth(self) -> Dict[str, Any]:
        """
        Exposes ground truth strictly for post-mission audit and evaluation.
        MUST NEVER be called by the agent during decision planning.
        """
        return {
            "eve_location": self._eve_location,
            "eve_status": self._eve_status,
            "internal_clues": [c.model_dump() for c in self._internal_clues],
            "total_battery_consumed": round(self.config.starting_battery - self.battery, 2),
            "total_time_consumed": self.config.time_budget - self.time_remaining,
            "scans_performed": self.scans_performed,
            "movement_actions": self.movement_actions,
            "invalid_actions": self.invalid_actions,
        }
