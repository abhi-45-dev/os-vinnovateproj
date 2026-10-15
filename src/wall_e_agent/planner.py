"""
Utility-based autonomous planner with Bayesian information gain,
A* pathfinding, and battery safety constraints.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Set, Tuple

from wall_e_agent.belief import BayesianGridBelief
from wall_e_agent.models import (
    ActionType,
    AgentAction,
    MissionConfig,
    Observation,
)


class UtilityPlanner:
    """
    Autonomous planner that balances information gain, goal pursuit,
    travel cost, and energy survival.
    """

    def __init__(self, config: MissionConfig):
        self.config = config
        self.width = config.grid_width
        self.height = config.grid_height
        self.obstacles: Set[Tuple[int, int]] = set(config.obstacles)
        self.recharge_stations: List[Tuple[int, int]] = list(config.recharge_stations)
        self.confidence_threshold: float = config.confidence_threshold
        self.costs = config.action_costs

        # Memory of visited locations with step timestamps to avoid loops
        self.visited_history: Dict[Tuple[int, int], int] = {}
        self.scanned_positions: Set[Tuple[int, int]] = set()

    def record_step(self, position: Tuple[int, int], step_number: int) -> None:
        """Track visited position to apply recency penalties against loops."""
        self.visited_history[position] = step_number

    def record_scan(self, position: Tuple[int, int]) -> None:
        """Track scanned coordinates to avoid redundant scans."""
        self.scanned_positions.add(position)

    def plan_next_action(
        self,
        observation: Observation,
        belief: BayesianGridBelief,
    ) -> Tuple[AgentAction, str]:
        """
        Evaluate current state and select the optimal deterministic action.
        Returns the chosen AgentAction and a detailed explanation.
        """
        pos = observation.position
        battery = observation.battery_remaining
        time_rem = observation.time_remaining

        # 1. Check Goal Condition (High confidence + proximity verification)
        best_coord, best_prob = belief.get_most_likely_location()
        best_stat, stat_conf = belief.get_most_likely_status()
        dist_to_best = math.hypot(best_coord[0] - pos[0], best_coord[1] - pos[1])

        # If probability is exceptionally high (>= threshold) and agent is in vicinity
        if best_prob >= self.confidence_threshold:
            if dist_to_best <= 1.0 or (pos == best_coord):
                # If status is still uncertain and we haven't scanned here, scan to verify status!
                if stat_conf < 0.70 and pos not in self.scanned_positions and battery >= self.costs.scan_battery_cost:
                    explanation = f"Target {best_coord} reached. Conducting proximity verification SCAN to identify EVE status."
                    return AgentAction(action_type=ActionType.SCAN), explanation

                # EVE located and verified!
                action = AgentAction(
                    action_type=ActionType.REPORT_FOUND,
                    reported_location=best_coord,
                    reported_status=best_stat,
                )
                explanation = (
                    f"REPORT_FOUND triggered: Target {best_coord} reached with confidence {best_prob:.2f} "
                    f"(status: {best_stat.value}, conf: {stat_conf:.2f})."
                )
                return action, explanation
            else:
                # High confidence, route directly to candidate location
                path = self._find_path(pos, best_coord)
                if path and len(path) > 1:
                    next_pos = path[1]
                    mv_action = self._coord_to_move_action(pos, next_pos)
                    if mv_action:
                        explanation = (
                            f"Approaching candidate EVE location {best_coord} (P={best_prob:.2f}) "
                            f"via {mv_action.value}. Distance: {len(path)-1} steps."
                        )
                        return AgentAction(action_type=mv_action), explanation

        # 2. Battery Safety Budget & Recharge Management
        dist_to_station, nearest_station = self._nearest_recharge_station(pos)
        safety_buffer = 4.0
        return_battery_needed = (dist_to_station * self.costs.move_battery_cost) + safety_buffer

        is_critical_battery = (battery <= return_battery_needed + 2.0)
        is_at_station = (pos in self.recharge_stations)

        if is_at_station:
            # If battery is not near full capacity and we have time, recharge
            if battery < (self.config.battery_capacity * 0.85) and time_rem > 5:
                explanation = (
                    f"RECHARGE selected at station {pos}: battery at {battery:.1f}/"
                    f"{self.config.battery_capacity:.1f}."
                )
                return AgentAction(action_type=ActionType.RECHARGE), explanation

        if is_critical_battery and nearest_station is not None and not is_at_station:
            path_to_station = self._find_path(pos, nearest_station)
            if path_to_station and len(path_to_station) > 1:
                next_pos = path_to_station[1]
                mv_action = self._coord_to_move_action(pos, next_pos)
                if mv_action:
                    explanation = (
                        f"SAFETY INTERVENTION: Low battery ({battery:.1f} <= {return_battery_needed:.1f}). "
                        f"Routing to recharge station {nearest_station} via {mv_action.value}."
                    )
                    return AgentAction(action_type=mv_action), explanation

        # 3. Scan Evaluation
        # Scan if local probability density is significant or high entropy, and hasn't scanned here yet
        local_mass = self._calculate_local_mass(pos, belief, radius=self.config.sensor_config.scan_radius)
        entropy = belief.get_entropy()
        can_afford_scan = (battery - self.costs.scan_battery_cost) > (return_battery_needed + 2.0)

        if pos not in self.scanned_positions and can_afford_scan and (local_mass >= 0.10 or entropy > 3.0):
            explanation = (
                f"SCAN selected: high local belief mass ({local_mass:.2f}) and spatial entropy ({entropy:.2f}). "
                f"Sufficient energy margin (+{battery - return_battery_needed:.1f})."
            )
            return AgentAction(action_type=ActionType.SCAN), explanation

        # 4. Expected Utility Evaluation of Candidate Moves
        top_candidates = belief.get_top_candidates(k=5)
        valid_moves = self._get_valid_moves(pos)

        if not valid_moves:
            # Trapped or waiting
            return AgentAction(action_type=ActionType.WAIT), "No valid movement moves available; waiting."

        best_move_action: Optional[ActionType] = None
        best_utility: float = -1e9
        best_decision_info: str = ""

        current_step = observation.step_count

        for move_act, target_coord in valid_moves:
            # Check safety margin from target coordinate
            dist_stat, _ = self._nearest_recharge_station(target_coord)
            batt_after = battery - self.costs.move_battery_cost
            if batt_after < (dist_stat * self.costs.move_battery_cost + 1.0):
                continue  # Unsafe move

            # Information pull towards top belief mass
            belief_pull = 0.0
            for cand_pos, cand_prob in top_candidates:
                d = math.hypot(target_coord[0] - cand_pos[0], target_coord[1] - cand_pos[1])
                belief_pull += cand_prob / (1.0 + 0.3 * d)

            # Recency loop penalty
            recency_penalty = 0.0
            if target_coord in self.visited_history:
                steps_ago = current_step - self.visited_history[target_coord]
                recency_penalty = max(0.0, 15.0 - steps_ago)

            utility = (belief_pull * 100.0) - (recency_penalty * 2.5)

            if utility > best_utility:
                best_utility = utility
                best_move_action = move_act
                best_decision_info = (
                    f"Selected {move_act.value} to {target_coord} (Utility: {utility:.2f}, "
                    f"BeliefPull: {belief_pull:.3f}, RecencyPenalty: {recency_penalty:.1f})."
                )

        if best_move_action is not None:
            return AgentAction(action_type=best_move_action), best_decision_info

        # 5. Fallback Policy: safe step to least-recently-visited neighbor
        return self._fallback_action(pos, valid_moves, current_step)

    def _fallback_action(
        self,
        pos: Tuple[int, int],
        valid_moves: List[Tuple[ActionType, Tuple[int, int]]],
        current_step: int,
    ) -> Tuple[AgentAction, str]:
        """Fallback policy when utility evaluation cannot find a preferred candidate."""
        # Pick valid move with lowest visitation timestamp
        best_move = valid_moves[0][0]
        oldest_visit = 1e9

        for act, coord in valid_moves:
            last_visit = self.visited_history.get(coord, -1)
            if last_visit < oldest_visit:
                oldest_visit = last_visit
                best_move = act

        return (
            AgentAction(action_type=best_move),
            f"Fallback policy: Selected {best_move.value} towards least-visited cell.",
        )

    def _calculate_local_mass(
        self,
        center: Tuple[int, int],
        belief: BayesianGridBelief,
        radius: int,
    ) -> float:
        """Sum probability mass within radius of center."""
        cx, cy = center
        grid = belief.get_grid_copy()
        total = 0.0
        for y in range(max(0, cy - radius), min(self.height, cy + radius + 1)):
            for x in range(max(0, cx - radius), min(self.width, cx + radius + 1)):
                if math.hypot(x - cx, y - cy) <= radius:
                    total += float(grid[y, x])
        return total

    def _nearest_recharge_station(
        self,
        pos: Tuple[int, int],
    ) -> Tuple[int, Optional[Tuple[int, int]]]:
        """Find Manhattan distance and coordinate of nearest recharge station."""
        if not self.recharge_stations:
            return 0, None
        best_dist = 1e9
        best_station = None
        for s in self.recharge_stations:
            dist = abs(pos[0] - s[0]) + abs(pos[1] - s[1])
            if dist < best_dist:
                best_dist = dist
                best_station = s
        return int(best_dist), best_station

    def _get_valid_moves(
        self,
        pos: Tuple[int, int],
    ) -> List[Tuple[ActionType, Tuple[int, int]]]:
        """Return list of valid cardinal movements and target coordinates."""
        x, y = pos
        moves = [
            (ActionType.MOVE_NORTH, (x, y + 1)),
            (ActionType.MOVE_SOUTH, (x, y - 1)),
            (ActionType.MOVE_EAST, (x + 1, y)),
            (ActionType.MOVE_WEST, (x - 1, y)),
        ]
        valid = []
        for act, (nx, ny) in moves:
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if (nx, ny) not in self.obstacles:
                    valid.append((act, (nx, ny)))
        return valid

    def _coord_to_move_action(
        self,
        curr: Tuple[int, int],
        target: Tuple[int, int],
    ) -> Optional[ActionType]:
        """Translate single coordinate delta into cardinal action."""
        dx = target[0] - curr[0]
        dy = target[1] - curr[1]
        if dx == 1 and dy == 0:
            return ActionType.MOVE_EAST
        if dx == -1 and dy == 0:
            return ActionType.MOVE_WEST
        if dx == 0 and dy == 1:
            return ActionType.MOVE_NORTH
        if dx == 0 and dy == -1:
            return ActionType.MOVE_SOUTH
        return None

    def _find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """A* search from start to goal avoiding obstacles."""
        if start == goal:
            return [start]

        open_set = [(0.0, start)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            cx, cy = current
            for nx, ny in [(cx, cy + 1), (cx, cy - 1), (cx + 1, cy), (cx - 1, cy)]:
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if (nx, ny) in self.obstacles and (nx, ny) != goal:
                    continue

                tentative_g = g_score[current] + 1.0
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = tentative_g
                    f_score = tentative_g + heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (f_score, (nx, ny)))

        return None
