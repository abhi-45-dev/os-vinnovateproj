"""
Tests for GridEnvironment: boundaries, obstacles, resource accounting,
observation privacy, and action mechanics.
"""

import pytest
from wall_e_agent.environment import GridEnvironment
from wall_e_agent.models import (
    ActionCostConfig,
    ActionType,
    AgentAction,
    EveStatus,
    MissionConfig,
    MissionStatus,
    SensorConfig,
)


@pytest.fixture
def basic_config():
    return MissionConfig(
        grid_width=5,
        grid_height=5,
        walle_start=(1, 1),
        recharge_stations=[(1, 1)],
        obstacles=[(2, 1), (2, 2)],
        battery_capacity=50.0,
        starting_battery=50.0,
        time_budget=40,
        action_costs=ActionCostConfig(
            move_battery_cost=1.0,
            move_time_cost=1,
            scan_battery_cost=2.0,
            scan_time_cost=2,
            wait_battery_cost=0.5,
            wait_time_cost=1,
            recharge_battery_rate=10.0,
            recharge_time_cost=1,
        ),
        sensor_config=SensorConfig(
            scan_radius=2,
            passive_clue_radius=1,
        ),
        clues=[
            {
                "clue_id": "c1",
                "clue_type": "BEACON_PING",
                "description": "Scrap beacon",
                "coordinates": (1, 2),
                "data": {"estimated_distance": 2.0},
            }
        ],
        eve_location=(4, 4),
        eve_status=EveStatus.OPERATIONAL,
        max_steps=30,
        random_seed=42,
    )


def test_environment_initialization_and_privacy(basic_config):
    env = GridEnvironment(basic_config)
    obs = env.reset()

    assert obs.position == (1, 1)
    assert obs.battery_remaining == 50.0
    assert obs.time_remaining == 40
    assert not obs.is_terminal
    assert obs.is_at_recharge_station is True

    # PRIVACY CHECK: Observation must NOT leak ground truth location or status
    obs_dict = obs.model_dump()
    assert "eve_location" not in obs_dict
    assert "eve_status" not in obs_dict
    assert not hasattr(obs, "eve_location")
    assert not hasattr(obs, "eve_status")

    # Clues returned must NOT contain ground-truth coordinates
    for clue in obs.discovered_clues:
        assert clue.coordinates is None


def test_boundary_and_obstacle_collision(basic_config):
    env = GridEnvironment(basic_config)
    env.reset()

    # Move WEST from (1, 1) -> (0, 1)
    obs = env.step(AgentAction(action_type=ActionType.MOVE_WEST))
    assert obs.position == (0, 1)
    assert obs.battery_remaining == 49.0

    # Try moving WEST again: boundary collision
    obs2 = env.step(AgentAction(action_type=ActionType.MOVE_WEST))
    assert obs2.position == (0, 1)  # Did not move
    assert obs2.battery_remaining == 48.0  # Cost still deducted
    assert "Blocked: Move MOVE_WEST hit boundary" in obs2.action_result_message
    assert env.invalid_actions == 1

    # Move EAST to (1, 1)
    env.step(AgentAction(action_type=ActionType.MOVE_EAST))
    # Try moving EAST towards obstacle at (2, 1)
    obs_obs = env.step(AgentAction(action_type=ActionType.MOVE_EAST))
    assert obs_obs.position == (1, 1)  # Blocked by obstacle
    assert "Blocked: Move MOVE_EAST hit obstacle" in obs_obs.action_result_message
    assert env.invalid_actions == 2


def test_battery_depletion_terminates_mission(basic_config):
    basic_config.starting_battery = 2.0
    env = GridEnvironment(basic_config)
    env.reset()

    # Move costs 1.0
    env.step(AgentAction(action_type=ActionType.MOVE_NORTH))
    assert env.battery == 1.0
    assert not env.is_terminal

    # Scan costs 2.0 -> drops below 0 -> terminates
    obs = env.step(AgentAction(action_type=ActionType.SCAN))
    assert env.battery == 0.0
    assert env.is_terminal
    assert env.mission_status == MissionStatus.FAILED_BATTERY_DEPLETED


def test_time_expiration_terminates_mission(basic_config):
    basic_config.time_budget = 2
    env = GridEnvironment(basic_config)
    env.reset()

    env.step(AgentAction(action_type=ActionType.MOVE_NORTH))
    obs = env.step(AgentAction(action_type=ActionType.MOVE_SOUTH))

    assert env.is_terminal
    assert env.mission_status == MissionStatus.FAILED_TIMEOUT


def test_recharge_mechanics(basic_config):
    env = GridEnvironment(basic_config)
    env.reset()

    # Drain some battery by scanning
    env.step(AgentAction(action_type=ActionType.SCAN))  # 48.0
    env.step(AgentAction(action_type=ActionType.SCAN))  # 46.0

    # At (1, 1), which is a recharge station
    obs = env.step(AgentAction(action_type=ActionType.RECHARGE))
    assert obs.battery_remaining > 46.0
    assert "Recharged +" in obs.action_result_message

    # Move away from station to (0, 1)
    env.step(AgentAction(action_type=ActionType.MOVE_WEST))
    # Attempt recharge where there is no station
    obs_fail = env.step(AgentAction(action_type=ActionType.RECHARGE))
    assert "Recharge failed: No charging station" in obs_fail.action_result_message


def test_report_found_success_and_failure(basic_config):
    env = GridEnvironment(basic_config)
    env.reset()

    # Incorrect report
    obs_fail = env.step(
        AgentAction(
            action_type=ActionType.REPORT_FOUND,
            reported_location=(0, 0),
            reported_status=EveStatus.OPERATIONAL,
        )
    )
    assert env.is_terminal
    assert env.mission_status == MissionStatus.FAILED_INCORRECT_REPORT

    # Reset and do correct report
    env2 = GridEnvironment(basic_config)
    env2.reset()
    obs_succ = env2.step(
        AgentAction(
            action_type=ActionType.REPORT_FOUND,
            reported_location=(4, 4),
            reported_status=EveStatus.OPERATIONAL,
        )
    )
    assert env2.is_terminal
    assert env2.mission_status == MissionStatus.SUCCESS
