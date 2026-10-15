"""
Integration and end-to-end tests for mission execution,
deterministic repeatability, and audit artifact generation.
"""

import json
from pathlib import Path
import pytest

from scripts.run_mission import load_config, run_mission
from scripts.run_benchmark import run_single_simulation
from wall_e_agent.agent import AutonomousAgent
from wall_e_agent.environment import GridEnvironment
from wall_e_agent.logging_utils import MissionLogger
from wall_e_agent.models import MissionStatus, TraceStep
from wall_e_agent.reporting import MissionReporter


@pytest.fixture
def sample_config_path(tmp_path):
    # Use repo sample config
    config_file = Path(__file__).resolve().parent.parent / "config" / "sample_mission.json"
    return config_file


def test_end_to_end_sample_mission(sample_config_path, tmp_path):
    config = load_config(sample_config_path)
    output_dir = tmp_path / "outputs"

    exit_code = run_mission(
        config=config,
        output_dir=output_dir,
        verbose=False,
        visualize=True,
        show_ascii=False,
    )

    assert exit_code == 0

    # Verify generated output files
    assert (output_dir / "mission_trace.json").exists()
    assert (output_dir / "mission_trace.csv").exists()
    assert (output_dir / "final_mission_report.json").exists()
    assert (output_dir / "final_mission_report.md").exists()
    assert (output_dir / "mission_visualization.png").exists()

    with open(output_dir / "final_mission_report.json", "r") as f:
        rep_data = json.load(f)

    assert rep_data["mission_status"] == MissionStatus.SUCCESS.value
    assert rep_data["eve_found"] is True
    assert rep_data["final_inferred_location"] == [9, 9]
    assert rep_data["final_inferred_status"] == "STANDBY"
    assert rep_data["matches_ground_truth"] is True


def test_deterministic_repeatability(sample_config_path, tmp_path):
    config = load_config(sample_config_path)
    config.random_seed = 999

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    run_mission(config=config, output_dir=out1, verbose=False, visualize=False, show_ascii=False)
    run_mission(config=config, output_dir=out2, verbose=False, visualize=False, show_ascii=False)

    with open(out1 / "mission_trace.json", "r") as f1, open(out2 / "mission_trace.json", "r") as f2:
        trace1 = json.load(f1)
        trace2 = json.load(f2)

    assert len(trace1) == len(trace2)
    for s1, s2 in zip(trace1, trace2):
        assert s1["step_number"] == s2["step_number"]
        assert s1["chosen_action"] == s2["chosen_action"]
        assert s1["agent_position"] == s2["agent_position"]
        assert s1["battery_remaining"] == s2["battery_remaining"]


def test_benchmark_simulation_execution(sample_config_path):
    config = load_config(sample_config_path)

    # Test that each agent policy can execute cleanly
    for agent_type in ["BayesianAutonomous", "GreedySignal", "RandomWalk"]:
        res = run_single_simulation(config, agent_type=agent_type, seed=42)
        assert "success" in res
        assert "steps" in res
        assert "battery_consumed" in res
        assert "location_error" in res
        assert res["steps"] > 0
