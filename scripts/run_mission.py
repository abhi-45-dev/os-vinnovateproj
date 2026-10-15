#!/usr/bin/env python3
"""
CLI entry point to execute an autonomous WALL-E mission.
Usage:
  python3 scripts/run_mission.py --config config/sample_mission.json --visualize --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to sys.path so wall_e_agent can be imported cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml
from wall_e_agent.agent import AutonomousAgent
from wall_e_agent.environment import GridEnvironment
from wall_e_agent.logging_utils import MissionLogger
from wall_e_agent.models import MissionConfig, TraceStep
from wall_e_agent.reporting import MissionReporter
from wall_e_agent.visualization import GridVisualizer


def load_config(config_path: Path | str) -> MissionConfig:
    """Load and validate mission configuration from JSON or YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    return MissionConfig(**data)


def run_mission(
    config: MissionConfig,
    output_dir: Path | str = "outputs",
    verbose: bool = True,
    visualize: bool = True,
    show_ascii: bool = True,
) -> int:
    """Execute a single autonomous mission end-to-end."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("🚀 WALL-E & EVE's Signal: Autonomous Mission Simulation")
    print(f"   Grid Size: {config.grid_width}x{config.grid_height} | Start: {config.walle_start}")
    print(f"   Battery Budget: {config.battery_capacity} | Time Budget: {config.time_budget}")
    print(f"   Deterministic Seed: {config.random_seed}")
    print("=" * 65)

    env = GridEnvironment(config)
    agent = AutonomousAgent(config)
    logger = MissionLogger(verbose=verbose)
    visualizer = GridVisualizer(env, agent)

    # 1. Initialize environment and initial agent perception
    obs = env.reset()
    agent.perceive_and_update(obs)

    # 2. Main Autonomous Decision-Action Cycle
    step_num = 0
    while not env.is_terminal:
        step_num += 1

        # Agent determines action based on belief state
        action, explanation = agent.decide_action()

        best_cand, best_prob = agent.belief.get_most_likely_location()
        inferred_stat, stat_conf = agent.belief.get_most_likely_status()
        entropy_val = agent.belief.get_entropy()

        # Execute action in simulated environment
        next_obs = env.step(action)

        # Log trace step
        sig_info = None
        if next_obs.signal_reading:
            sig_info = {
                "strength": next_obs.signal_reading.strength,
                "bearing": next_obs.signal_reading.bearing,
                "distance": next_obs.signal_reading.noisy_distance,
            }

        trace_step = TraceStep(
            step_number=step_num,
            agent_position=obs.position,
            chosen_action=action.action_type.value,
            action_args={
                "reported_location": action.reported_location,
                "reported_status": action.reported_status.value if action.reported_status else None,
            },
            battery_remaining=next_obs.battery_remaining,
            time_remaining=next_obs.time_remaining,
            action_result=next_obs.action_result_message,
            observation_summary={
                "signal": sig_info,
                "clues_discovered": [c.clue_id for c in next_obs.discovered_clues],
            },
            belief_summary={
                "top_candidate": best_cand,
                "probability": best_prob,
                "inferred_status": inferred_stat.value,
            },
            estimated_best_candidate=best_cand,
            estimated_best_probability=best_prob,
            inferred_status=inferred_stat.value,
            status_confidence=stat_conf,
            entropy=round(entropy_val, 3),
            decision_explanation=explanation,
        )
        logger.log_step(trace_step)

        # Update agent with results of action
        agent.perceive_and_update(next_obs)
        obs = next_obs

    # 3. Post-Mission Evaluation & Reporting
    reporter = MissionReporter(env, agent)
    report = reporter.generate_report()

    # Save artifacts
    trace_json_path = out_dir / "mission_trace.json"
    trace_csv_path = out_dir / "mission_trace.csv"
    report_json_path = out_dir / "final_mission_report.json"
    report_md_path = out_dir / "final_mission_report.md"
    vis_png_path = out_dir / "mission_visualization.png"

    logger.export_json(trace_json_path)
    logger.export_csv(trace_csv_path)
    reporter.export_json(report, report_json_path)
    reporter.export_markdown(report, report_md_path)

    print("\n" + "=" * 65)
    outcome_symbol = "🏆 SUCCESS" if report.eve_found else "❌ FAILED"
    print(f"Mission Finished: {outcome_symbol}")
    print(f"Status: {report.mission_status.value}")
    print(f"Reason: {report.termination_reason}")
    print(f"Steps: {report.total_steps} | Scans: {report.scans_performed}")
    print(f"Battery Consumed: {report.battery_consumed:.1f} | Remaining: {report.battery_remaining:.1f}")
    print(f"Inferred Location: {report.final_inferred_location} (Confidence: {report.location_confidence*100:.1f}%)")
    print(f"Ground Truth Location: {report.ground_truth_location} (Distance Error: {report.location_error_distance:.2f})")
    print(f"Inferred Status: {report.final_inferred_status.value} | GT: {report.ground_truth_status.value}")
    print("=" * 65)

    if show_ascii:
        print("\nFinal Exploration Map (Post-Mission Ground Truth Revealed):")
        print(visualizer.render_ascii(show_ground_truth=True))

    if visualize:
        saved = visualizer.render_matplotlib(
            trace_steps=logger.get_steps(),
            report=report,
            output_path=vis_png_path,
        )
        if saved:
            print(f"\n📊 Visualization saved to: {vis_png_path}")

    print(f"📁 Mission Trace JSON: {trace_json_path}")
    print(f"📁 Mission Trace CSV:  {trace_csv_path}")
    print(f"📁 Final Report JSON:  {report_json_path}")
    print(f"📁 Final Report MD:    {report_md_path}")

    return 0 if report.eve_found else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Agent Simulation: WALL-E and EVE's Signal"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/sample_mission.json",
        help="Path to mission configuration file (JSON or YAML)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed for deterministic simulation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save traces, reports, and charts",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Generate Matplotlib graphical plot (default: True)",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_false",
        dest="visualize",
        help="Disable Matplotlib graphical plot generation",
    )
    parser.add_argument(
        "--no-ascii",
        action="store_true",
        help="Disable ASCII terminal map printing",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step log output",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max steps limit",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config.random_seed = args.seed
    if args.max_steps is not None:
        config.max_steps = args.max_steps

    exit_code = run_mission(
        config=config,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        visualize=args.visualize,
        show_ascii=not args.no_ascii,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
