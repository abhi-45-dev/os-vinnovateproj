"""
Structured logging and trace generation for mission auditing.
Supports JSON and CSV export formats.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from wall_e_agent.models import TraceStep


class MissionLogger:
    """
    Collects step-by-step simulation traces and exports to structured formats.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.steps: List[TraceStep] = []

    def log_step(self, step: TraceStep) -> None:
        """Record a trace step and optionally print to console."""
        self.steps.append(step)

        if self.verbose:
            sig_info = step.observation_summary.get("signal")
            sig_str = f"Sig: {sig_info}" if sig_info else "No signal"
            print(
                f"[Step {step.step_number:03d}] Pos: {step.agent_position} | "
                f"Act: {step.chosen_action:<14} | Batt: {step.battery_remaining:.1f} | "
                f"Top: {step.estimated_best_candidate} (P={step.estimated_best_probability:.2f}) | "
                f"H={step.entropy:.2f} | {sig_str}"
            )
            print(f"       Reason: {step.decision_explanation}")
            print(f"       Result: {step.action_result}")

    def export_json(self, file_path: str | Path) -> None:
        """Export mission trace as a pretty-printed JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [step.model_dump() for step in self.steps]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def export_csv(self, file_path: str | Path) -> None:
        """Export mission trace as a tabular CSV file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self.steps:
            return

        fieldnames = [
            "step_number",
            "pos_x",
            "pos_y",
            "chosen_action",
            "battery_remaining",
            "time_remaining",
            "action_result",
            "best_cand_x",
            "best_cand_y",
            "best_cand_prob",
            "inferred_status",
            "status_confidence",
            "entropy",
            "decision_explanation",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for step in self.steps:
                writer.writerow({
                    "step_number": step.step_number,
                    "pos_x": step.agent_position[0],
                    "pos_y": step.agent_position[1],
                    "chosen_action": step.chosen_action,
                    "battery_remaining": step.battery_remaining,
                    "time_remaining": step.time_remaining,
                    "action_result": step.action_result,
                    "best_cand_x": step.estimated_best_candidate[0],
                    "best_cand_y": step.estimated_best_candidate[1],
                    "best_cand_prob": step.estimated_best_probability,
                    "inferred_status": step.inferred_status,
                    "status_confidence": step.status_confidence,
                    "entropy": step.entropy,
                    "decision_explanation": step.decision_explanation,
                })

    def get_steps(self) -> List[TraceStep]:
        """Return defensive copy of all recorded steps."""
        return list(self.steps)
