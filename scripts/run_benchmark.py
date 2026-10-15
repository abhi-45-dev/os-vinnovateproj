#!/usr/bin/env python3
"""
Benchmark and Reproducibility Suite.
Compares the full Bayesian Autonomous Agent against baseline policies
(Random Walk and Greedy Heuristic) across multiple deterministic seeds.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wall_e_agent.agent import AutonomousAgent
from wall_e_agent.environment import GridEnvironment
from wall_e_agent.models import (
    ActionType,
    AgentAction,
    EveStatus,
    MissionConfig,
    MissionStatus,
)
from wall_e_agent.reporting import MissionReporter
from scripts.run_mission import load_config


class RandomWalkBaselineAgent:
    """Baseline policy: randomly wanders and randomly guesses."""

    def __init__(self, config: MissionConfig, seed: int):
        self.config = config
        self.rng = random.Random(seed)
        self.step = 0

    def perceive_and_update(self, obs: Any) -> None:
        self.last_obs = obs

    def decide_action(self) -> Tuple[AgentAction, str]:
        self.step += 1
        pos = self.last_obs.position

        # If at battery <= 5, try recharge or wait
        if self.last_obs.battery_remaining <= 5.0 and self.last_obs.is_at_recharge_station:
            return AgentAction(action_type=ActionType.RECHARGE), "Emergency recharge"

        valid = []
        surr = self.last_obs.local_surroundings
        if surr.get("north"):
            valid.append(ActionType.MOVE_NORTH)
        if surr.get("south"):
            valid.append(ActionType.MOVE_SOUTH)
        if surr.get("east"):
            valid.append(ActionType.MOVE_EAST)
        if surr.get("west"):
            valid.append(ActionType.MOVE_WEST)

        # 5% chance of scan
        if self.rng.random() < 0.10:
            return AgentAction(action_type=ActionType.SCAN), "Random scan"

        # Blind guess report after 30 steps
        if self.step > 30 and self.rng.random() < 0.15:
            guess_x = self.rng.randint(0, self.config.grid_width - 1)
            guess_y = self.rng.randint(0, self.config.grid_height - 1)
            return AgentAction(
                action_type=ActionType.REPORT_FOUND,
                reported_location=(guess_x, guess_y),
                reported_status=EveStatus.OPERATIONAL,
            ), "Blind guess report"

        if valid:
            act = self.rng.choice(valid)
            return AgentAction(action_type=act), f"Random move {act.value}"
        return AgentAction(action_type=ActionType.WAIT), "Wait"


class GreedySignalBaselineAgent:
    """Baseline policy: greedily steps towards noisy bearing angle without Bayesian memory or energy safety."""

    def __init__(self, config: MissionConfig, seed: int):
        self.config = config
        self.rng = random.Random(seed)
        self.last_bearing = 45.0
        self.step = 0

    def perceive_and_update(self, obs: Any) -> None:
        self.last_obs = obs
        if obs.signal_reading:
            self.last_bearing = obs.signal_reading.bearing

    def decide_action(self) -> Tuple[AgentAction, str]:
        self.step += 1
        pos = self.last_obs.position

        # If signal is very strong (distance close), report current cell
        if self.last_obs.signal_reading and self.last_obs.signal_reading.strength > 0.85:
            return AgentAction(
                action_type=ActionType.REPORT_FOUND,
                reported_location=pos,
                reported_status=EveStatus.OPERATIONAL,
            ), "Greedy report based on high signal strength"

        # Direction vector from bearing angle
        rad = math.radians(self.last_bearing)
        target_dx = math.cos(rad)
        target_dy = math.sin(rad)

        surr = self.last_obs.local_surroundings
        candidates = []
        if surr.get("east") and target_dx > 0:
            candidates.append((ActionType.MOVE_EAST, target_dx))
        if surr.get("west") and target_dx < 0:
            candidates.append((ActionType.MOVE_WEST, -target_dx))
        if surr.get("north") and target_dy > 0:
            candidates.append((ActionType.MOVE_NORTH, target_dy))
        if surr.get("south") and target_dy < 0:
            candidates.append((ActionType.MOVE_SOUTH, -target_dy))

        if candidates:
            candidates.sort(key=lambda c: c[1], reverse=True)
            chosen = candidates[0][0]
            return AgentAction(action_type=chosen), f"Greedy bearing move {chosen.value}"

        # Otherwise pick any valid move
        for d, act in [("north", ActionType.MOVE_NORTH), ("south", ActionType.MOVE_SOUTH),
                       ("east", ActionType.MOVE_EAST), ("west", ActionType.MOVE_WEST)]:
            if surr.get(d):
                return AgentAction(action_type=act), f"Greedy fallback {act.value}"

        return AgentAction(action_type=ActionType.WAIT), "Wait"


def run_single_simulation(config: MissionConfig, agent_type: str, seed: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg.random_seed = seed

    env = GridEnvironment(cfg)
    obs = env.reset()

    if agent_type == "BayesianAutonomous":
        agent = AutonomousAgent(cfg)
    elif agent_type == "GreedySignal":
        agent = GreedySignalBaselineAgent(cfg, seed)  # type: ignore
    elif agent_type == "RandomWalk":
        agent = RandomWalkBaselineAgent(cfg, seed)  # type: ignore
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    agent.perceive_and_update(obs)

    step = 0
    while not env.is_terminal and step < cfg.max_steps:
        step += 1
        act, _ = agent.decide_action()
        next_obs = env.step(act)
        agent.perceive_and_update(next_obs)
        obs = next_obs

    # Evaluate
    gt = env.get_evaluation_ground_truth()
    eve_loc = gt["eve_location"]

    if agent_type == "BayesianAutonomous":
        best_loc, _ = agent.belief.get_most_likely_location()  # type: ignore
    else:
        best_loc = env.agent_position

    dist_err = math.hypot(best_loc[0] - eve_loc[0], best_loc[1] - eve_loc[1])
    exhausted = env.mission_status in (MissionStatus.FAILED_BATTERY_DEPLETED, MissionStatus.FAILED_TIMEOUT)

    return {
        "success": (env.mission_status == MissionStatus.SUCCESS),
        "steps": env.step_count,
        "scans": env.scans_performed,
        "battery_consumed": gt["total_battery_consumed"],
        "location_error": dist_err,
        "budget_exhausted": exhausted,
        "status": env.mission_status.value,
    }


def run_benchmark(
    config_path: str = "config/sample_mission.json",
    num_runs: int = 20,
    base_seed: int = 100,
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    base_config = load_config(config_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agent_types = ["BayesianAutonomous", "GreedySignal", "RandomWalk"]
    results: Dict[str, List[Dict[str, Any]]] = {at: [] for at in agent_types}

    print(f"\n🔬 Running Benchmark: {num_runs} runs per policy across deterministic seeds {base_seed}..{base_seed+num_runs-1}")
    print("-" * 75)

    for at in agent_types:
        for i in range(num_runs):
            seed = base_seed + i
            res = run_single_simulation(base_config, at, seed)
            results[at].append(res)

    # Compute aggregate statistics
    summary = {}
    for at in agent_types:
        runs = results[at]
        n = len(runs)
        success_count = sum(1 for r in runs if r["success"])
        exhaust_count = sum(1 for r in runs if r["budget_exhausted"])
        avg_steps = sum(r["steps"] for r in runs) / n
        avg_scans = sum(r["scans"] for r in runs) / n
        avg_batt = sum(r["battery_consumed"] for r in runs) / n
        avg_err = sum(r["location_error"] for r in runs) / n

        summary[at] = {
            "runs": n,
            "success_rate_pct": round((success_count / n) * 100.0, 1),
            "budget_exhaustion_rate_pct": round((exhaust_count / n) * 100.0, 1),
            "avg_steps": round(avg_steps, 2),
            "avg_scans": round(avg_scans, 2),
            "avg_battery_consumed": round(avg_batt, 2),
            "avg_location_error": round(avg_err, 2),
        }

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Agent Policy':<22} | {'Success%':<9} | {'Exhaust%':<9} | {'Avg Steps':<10} | {'Avg Scans':<10} | {'Avg Batt':<10} | {'Avg Err':<8}")
    print("=" * 90)
    for at, s in summary.items():
        print(
            f"{at:<22} | {s['success_rate_pct']:>8.1f}% | {s['budget_exhaustion_rate_pct']:>8.1f}% | "
            f"{s['avg_steps']:>10.1f} | {s['avg_scans']:>10.1f} | {s['avg_battery_consumed']:>10.1f} | "
            f"{s['avg_location_error']:>8.2f}"
        )
    print("=" * 90)

    # Save benchmark results
    out_file = out_dir / "benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "runs": results}, f, indent=2)
    print(f"\nBenchmark results saved to: {out_file}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark suite on WALL-E agent")
    parser.add_argument("--config", type=str, default="config/sample_mission.json", help="Mission config path")
    parser.add_argument("--runs", type=int, default=20, help="Number of missions per agent policy")
    parser.add_argument("--base-seed", type=int, default=100, help="Base random seed")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")

    args = parser.parse_args()
    run_benchmark(
        config_path=args.config,
        num_runs=args.runs,
        base_seed=args.base_seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
