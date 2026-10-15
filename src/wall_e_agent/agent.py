"""
Autonomous decision-making agent for WALL-E.
Integrates Bayesian belief state tracking with utility-driven planning.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from wall_e_agent.belief import BayesianGridBelief
from wall_e_agent.models import (
    ActionType,
    AgentAction,
    MissionConfig,
    Observation,
)
from wall_e_agent.planner import UtilityPlanner


class AutonomousAgent:
    """
    Autonomous agent controlling WALL-E's search and exploration.
    Strictly isolated from ground truth; relies exclusively on observations and Bayesian belief.
    """

    def __init__(self, config: MissionConfig):
        self.config = config
        self.belief = BayesianGridBelief(config)
        self.planner = UtilityPlanner(config)

        self.step_count: int = 0
        self.current_observation: Optional[Observation] = None
        self.trajectory: List[Tuple[int, int]] = [config.walle_start]
        self.evidence_gathered: List[str] = []
        self.last_action: Optional[AgentAction] = None

    def perceive_and_update(self, observation: Observation) -> None:
        """
        Process the new observation from the environment and update belief state.
        """
        self.current_observation = observation
        self.step_count = observation.step_count
        pos = observation.position

        # Update trajectory
        if not self.trajectory or self.trajectory[-1] != pos:
            self.trajectory.append(pos)

        self.planner.record_step(pos, self.step_count)

        # 1. Update from signal observation
        if observation.signal_reading:
            self.belief.update_from_signal(
                signal=observation.signal_reading,
                observer_pos=pos,
            )

        # 2. Update from newly discovered clues
        for clue in observation.discovered_clues:
            evidence_desc = f"Step {self.step_count}: Found {clue.clue_type.value} - {clue.description}"
            self.evidence_gathered.append(evidence_desc)
            self.belief.update_from_clue(clue, observer_pos=pos)

        # 3. Update negative observations (clearing visited cells)
        if self.last_action and self.last_action.action_type == ActionType.SCAN:
            self.planner.record_scan(pos)
            self.belief.update_from_negative_observation(
                observer_pos=pos,
                radius=self.config.sensor_config.scan_radius,
            )
        else:
            # Current cell has no EVE (since mission hasn't ended via REPORT_FOUND)
            self.belief.update_from_negative_observation(
                observer_pos=pos,
                radius=0,
            )

    def decide_action(self) -> Tuple[AgentAction, str]:
        """
        Select the next action according to the utility planning policy.
        """
        if self.current_observation is None:
            raise RuntimeError("Agent cannot decide action before receiving initial observation.")

        action, explanation = self.planner.plan_next_action(
            observation=self.current_observation,
            belief=self.belief,
        )
        self.last_action = action
        return action, explanation

    def get_trajectory(self) -> List[Tuple[int, int]]:
        """Return the complete sequence of visited coordinates."""
        return list(self.trajectory)

    def get_evidence_gathered(self) -> List[str]:
        """Return list of descriptive strings for all collected evidence."""
        return list(self.evidence_gathered)
