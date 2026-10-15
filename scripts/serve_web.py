#!/usr/bin/env python3
"""
Lightweight Web Server and Interactive Simulation API.
Serves the futuristic WALL-E & EVE Mission Control Dashboard.
Usage:
  python3 scripts/serve_web.py --port 8000
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wall_e_agent.agent import AutonomousAgent
from wall_e_agent.environment import GridEnvironment
from wall_e_agent.models import MissionConfig, MissionStatus, TraceStep
from wall_e_agent.reporting import MissionReporter
from scripts.run_mission import load_config


WEB_DIR = PROJECT_ROOT / "web"
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


class MissionControlHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler serving web assets and JSON API endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/scenarios":
            self._handle_get_scenarios()
        elif path == "/api/benchmark":
            self._handle_get_benchmark()
        elif path == "/api/load_scenario":
            query = parse_qs(parsed.query)
            name = query.get("name", ["sample_mission"])[0]
            self._handle_load_scenario(name)
        else:
            # Fallback to static file serving
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/run_simulation":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}
            self._handle_run_simulation(data)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "API endpoint not found")

    def _send_json_response(self, data: Any, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _handle_get_scenarios(self):
        scenarios = []
        for file in sorted(CONFIG_DIR.glob("*.json")):
            scenarios.append({
                "id": file.stem,
                "name": file.stem.replace("_", " ").title(),
                "path": str(file.relative_to(PROJECT_ROOT)),
            })
        self._send_json_response({"scenarios": scenarios})

    def _handle_get_benchmark(self):
        bench_file = OUTPUTS_DIR / "benchmark_results.json"
        if bench_file.exists():
            with open(bench_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._send_json_response(data)
        else:
            self._send_json_response({"error": "Benchmark file not found. Run scripts/run_benchmark.py first."}, status=404)

    def _handle_load_scenario(self, scenario_name: str):
        cfg_file = CONFIG_DIR / f"{scenario_name}.json"
        if not cfg_file.exists():
            self._send_json_response({"error": f"Scenario {scenario_name} not found"}, status=404)
            return

        config = load_config(cfg_file)
        # Run simulation with full step-by-step state snapshot
        result = run_simulation_with_snapshots(config)
        self._send_json_response(result)

    def _handle_run_simulation(self, params: Dict[str, Any]):
        scenario_name = params.get("scenario", "sample_mission")
        cfg_file = CONFIG_DIR / f"{scenario_name}.json"
        if not cfg_file.exists():
            self._send_json_response({"error": f"Scenario {scenario_name} not found"}, status=404)
            return

        config = load_config(cfg_file)
        if "seed" in params and params["seed"] is not None:
            config.random_seed = int(params["seed"])
        if "starting_battery" in params and params["starting_battery"]:
            config.starting_battery = float(params["starting_battery"])
        if "confidence_threshold" in params and params["confidence_threshold"]:
            config.confidence_threshold = float(params["confidence_threshold"])

        result = run_simulation_with_snapshots(config)
        self._send_json_response(result)


def run_simulation_with_snapshots(config: MissionConfig) -> Dict[str, Any]:
    """Execute simulation while recording belief grid snapshots for 2D UI playback."""
    env = GridEnvironment(config)
    agent = AutonomousAgent(config)

    obs = env.reset()
    agent.perceive_and_update(obs)

    snapshots = []
    # Initial state (Step 0)
    best_cand, best_prob = agent.belief.get_most_likely_location()
    best_stat, stat_conf = agent.belief.get_most_likely_status()
    snapshots.append({
        "step_number": 0,
        "position": obs.position,
        "action": "INITIALIZE",
        "battery": obs.battery_remaining,
        "time": obs.time_remaining,
        "signal": {
            "strength": obs.signal_reading.strength if obs.signal_reading else None,
            "bearing": obs.signal_reading.bearing if obs.signal_reading else None,
            "distance": obs.signal_reading.noisy_distance if obs.signal_reading else None,
        } if obs.signal_reading else None,
        "clues_found": [c.model_dump() for c in obs.discovered_clues],
        "top_candidate": best_cand,
        "top_probability": round(best_prob, 4),
        "inferred_status": best_stat.value,
        "status_confidence": round(stat_conf, 4),
        "entropy": round(agent.belief.get_entropy(), 3),
        "explanation": "Initial sensor readings received and prior belief initialized.",
        "action_result": obs.action_result_message,
        "belief_grid": agent.belief.get_grid_copy().round(4).tolist(),
    })

    step = 0
    while not env.is_terminal and step < config.max_steps:
        step += 1
        action, explanation = agent.decide_action()
        next_obs = env.step(action)
        agent.perceive_and_update(next_obs)

        best_cand, best_prob = agent.belief.get_most_likely_location()
        best_stat, stat_conf = agent.belief.get_most_likely_status()

        snapshots.append({
            "step_number": step,
            "position": next_obs.position,
            "action": action.action_type.value,
            "battery": next_obs.battery_remaining,
            "time": next_obs.time_remaining,
            "signal": {
                "strength": next_obs.signal_reading.strength if next_obs.signal_reading else None,
                "bearing": next_obs.signal_reading.bearing if next_obs.signal_reading else None,
                "distance": next_obs.signal_reading.noisy_distance if next_obs.signal_reading else None,
            } if next_obs.signal_reading else None,
            "clues_found": [c.model_dump() for c in next_obs.discovered_clues],
            "top_candidate": best_cand,
            "top_probability": round(best_prob, 4),
            "inferred_status": best_stat.value,
            "status_confidence": round(stat_conf, 4),
            "entropy": round(agent.belief.get_entropy(), 3),
            "explanation": explanation,
            "action_result": next_obs.action_result_message,
            "belief_grid": agent.belief.get_grid_copy().round(4).tolist(),
        })

    reporter = MissionReporter(env, agent)
    report = reporter.generate_report()

    return {
        "config": {
            "grid_width": config.grid_width,
            "grid_height": config.grid_height,
            "walle_start": config.walle_start,
            "recharge_stations": config.recharge_stations,
            "obstacles": config.obstacles,
            "battery_capacity": config.battery_capacity,
            "starting_battery": config.starting_battery,
            "time_budget": config.time_budget,
            "random_seed": config.random_seed,
            "clues": [c.model_dump() for c in env._internal_clues],
        },
        "ground_truth": {
            "eve_location": env._eve_location,
            "eve_status": env._eve_status.value,
        },
        "report": report.model_dump(),
        "snapshots": snapshots,
    }


def main():
    parser = argparse.ArgumentParser(description="Serve the WALL-E Mission Control Web Application")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the web server (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    args = parser.parse_args()

    server_address = (args.host, args.port)
    httpd = ThreadingHTTPServer(server_address, MissionControlHandler)
    url = f"http://{args.host}:{args.port}"
    print("=" * 65)
    print("🛰️  WALL-E & EVE Mission Control Web Server Active")
    print(f"👉 Local Access URL: {url}")
    print(f"📁 Web Root: {WEB_DIR}")
    print("   Press Ctrl+C to terminate server")
    print("=" * 65)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down gracefully.")
        httpd.server_close()


if __name__ == "__main__":
    main()
