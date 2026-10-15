"""
Unit tests for AutonomousAgent and UtilityPlanner: action selection,
energy safety policies, loop avoidance, and inspectable explanations.
"""

import pytest
from wall_e_agent.agent import AutonomousAgent
from wall_e_agent.environment import GridEnvironment
from wall_e_agent.models import (
    ActionCostConfig,
    ActionType,
    AgentAction,
    EveStatus,
    MissionConfig,
    Observation,
    SignalReading,
)


@pytest.fixture
def agent_config():
    return MissionConfig(
        grid_width=6,
        grid_height=6,
        walle_start=(0, 0),
        recharge_stations=[(0, 0)],
        obstacles=[(2, 0), (2, 1)],
        battery_capacity=100.0,
        starting_battery=100.0,
        time_budget=50,
        eve_location=(5, 5),
        eve_status=EveStatus.OPERATIONAL,
        confidence_threshold=0.75,
    )


def test_agent_perceive_and_decision_cycle(agent_config):
    agent = AutonomousAgent(agent_config)

    obs = Observation(
        position=(0, 0),
        battery_remaining=100.0,
        battery_capacity=100.0,
        time_remaining=50,
        step_count=1,
        signal_reading=SignalReading(
            strength=0.25,
            bearing=45.0,
            noisy_distance=7.0,
            timestamp=1,
        ),
        local_surroundings={"north": True, "south": False, "east": True, "west": False},
        is_at_recharge_station=True,
    )

    agent.perceive_and_update(obs)
    assert agent.trajectory == [(0, 0)]

    action, explanation = agent.decide_action()
    assert isinstance(action, AgentAction)
    assert isinstance(explanation, str)
    assert len(explanation) > 10
    # Must choose a valid action
    assert action.action_type in (ActionType.MOVE_NORTH, ActionType.MOVE_EAST, ActionType.SCAN, ActionType.WAIT)


def test_agent_recharge_priority_when_battery_low(agent_config):
    agent = AutonomousAgent(agent_config)

    # Agent is at station (0, 0) with low battery
    obs = Observation(
        position=(0, 0),
        battery_remaining=25.0,
        battery_capacity=100.0,
        time_remaining=40,
        step_count=10,
        local_surroundings={"north": True, "south": False, "east": True, "west": False},
        is_at_recharge_station=True,
    )

    agent.perceive_and_update(obs)
    action, explanation = agent.decide_action()

    # Low battery at recharge station should trigger RECHARGE
    assert action.action_type == ActionType.RECHARGE
    assert "RECHARGE" in explanation


def test_agent_triggers_report_found_when_confident_and_adjacent(agent_config):
    agent = AutonomousAgent(agent_config)

    # Artificially concentrate belief at (1, 0)
    agent.belief.grid[:, :] = 0.0
    agent.belief.grid[0, 1] = 1.0  # Cell (1, 0) has 100% probability

    obs = Observation(
        position=(0, 0),
        battery_remaining=80.0,
        battery_capacity=100.0,
        time_remaining=30,
        step_count=5,
        local_surroundings={"north": True, "south": False, "east": True, "west": False},
        is_at_recharge_station=False,
    )

    # 1. When status is uncertain, agent executes proximity SCAN to verify status
    agent.perceive_and_update(obs)
    action1, explanation1 = agent.decide_action()
    assert action1.action_type == ActionType.SCAN
    assert "Conducting proximity verification SCAN" in explanation1

    # 2. Once status is confirmed, agent issues REPORT_FOUND
    agent.belief.status_belief[EveStatus.OPERATIONAL] = 1.0
    agent.planner.record_scan((0, 0))
    action2, explanation2 = agent.decide_action()
    assert action2.action_type == ActionType.REPORT_FOUND
    assert action2.reported_location == (1, 0)
    assert action2.reported_status == EveStatus.OPERATIONAL
    assert "REPORT_FOUND triggered" in explanation2
