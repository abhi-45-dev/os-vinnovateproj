import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

try:
    from wall_e_agent.models import MissionConfig
    from scripts.serve_web import run_simulation_with_snapshots
    from scripts.run_mission import load_config
except ImportError:
    run_simulation_with_snapshots = None
    load_config = None


class handler(BaseHTTPRequestHandler):
    """Vercel Serverless Python Handler supporting both direct and rewritten requests."""

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        endpoint = query.get("endpoint", [""])[0] or parsed.path

        if "scenarios" in endpoint:
            scenarios = [
                {"id": "sample_mission", "name": "Sample Mission (12x12 - Earth Ruins)", "path": "config/sample_mission.json"},
                {"id": "challenging_mission", "name": "Challenging Mission (16x16 - Industrial Canyons)", "path": "config/challenging_mission.json"}
            ]
            self._send_json({"scenarios": scenarios})

        elif "benchmark" in endpoint:
            bench_file = ROOT / "web" / "data" / "benchmark_results.json"
            if not bench_file.exists():
                bench_file = ROOT / "outputs" / "benchmark_results.json"
            if bench_file.exists():
                with open(bench_file, "r", encoding="utf-8") as f:
                    self._send_json(json.load(f))
            else:
                self._send_json({"error": "Benchmark data not found"}, status=404)

        elif "load_scenario" in endpoint:
            name = query.get("name", ["sample_mission"])[0]
            data_file = ROOT / "web" / "data" / f"{name}.json"

            if data_file.exists():
                with open(data_file, "r", encoding="utf-8") as f:
                    self._send_json(json.load(f))
            elif load_config and run_simulation_with_snapshots:
                cfg_file = ROOT / "config" / f"{name}.json"
                if cfg_file.exists():
                    config = load_config(cfg_file)
                    res = run_simulation_with_snapshots(config)
                    self._send_json(res)
                else:
                    self._send_json({"error": f"Scenario {name} not found"}, status=404)
            else:
                self._send_json({"error": f"Scenario {name} not found"}, status=404)

        else:
            self._send_json({"status": "API active", "endpoint": endpoint})

    def do_POST(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        endpoint = query.get("endpoint", [""])[0] or parsed.path

        if "run_simulation" in endpoint:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body) if body else {}
            name = data.get("scenario", "sample_mission")

            if load_config and run_simulation_with_snapshots:
                cfg_file = ROOT / "config" / f"{name}.json"
                if cfg_file.exists():
                    config = load_config(cfg_file)
                    if "seed" in data and data["seed"] is not None:
                        config.random_seed = int(data["seed"])
                    res = run_simulation_with_snapshots(config)
                    self._send_json(res)
                    return

            data_file = ROOT / "web" / "data" / f"{name}.json"
            if data_file.exists():
                with open(data_file, "r", encoding="utf-8") as f:
                    self._send_json(json.load(f))
            else:
                self._send_json({"error": f"Scenario {name} not found"}, status=404)

        else:
            self.send_error(404, "Endpoint not found")

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
