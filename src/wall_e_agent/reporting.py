"""
Post-mission evaluation, reporting, and ground-truth validation.
Evaluates agent performance strictly after simulation termination.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

from wall_e_agent.agent import AutonomousAgent
from wall_e_agent.environment import GridEnvironment
from wall_e_agent.models import (
    EveStatus,
    MissionReport,
    MissionStatus,
)


class MissionReporter:
    """
    Evaluates completed missions against ground-truth parameters
    and generates formal post-mission reports.
    """

    def __init__(self, env: GridEnvironment, agent: AutonomousAgent):
        self.env = env
        self.agent = agent

    def generate_report(self) -> MissionReport:
        """
        Evaluate mission results and compare inferred estimates against ground truth.
        """
        eval_gt = self.env.get_evaluation_ground_truth()
        gt_loc = eval_gt["eve_location"]
        gt_stat = eval_gt["eve_status"]

        best_loc, loc_conf = self.agent.belief.get_most_likely_location()
        best_stat, stat_conf = self.agent.belief.get_most_likely_status()

        loc_error = math.hypot(best_loc[0] - gt_loc[0], best_loc[1] - gt_loc[1])
        matches_gt = (best_loc == gt_loc) and (best_stat == gt_stat)
        eve_found = (self.env.mission_status == MissionStatus.SUCCESS)

        # Recommendation logic based on outcome
        if eve_found:
            rec = (
                f"Deploy transport/extraction crawler directly to coordinates {gt_loc}. "
                f"EVE status confirmed as {gt_stat.value}."
            )
        elif self.env.mission_status == MissionStatus.FAILED_BATTERY_DEPLETED:
            rec = (
                f"Dispatch mobile emergency solar recharge drone to WALL-E's final position "
                f"{self.env.agent_position}. Re-route search to high-probability sector around {best_loc}."
            )
        elif self.env.mission_status == MissionStatus.FAILED_TIMEOUT:
            rec = (
                f"Allocate additional time budget (+50 ticks) and deploy secondary scouting probe "
                f"to sector {best_loc} (posterior confidence: {loc_conf:.2f})."
            )
        else:
            rec = (
                f"Re-calibrate signal sensors to account for multi-path noise. "
                f"Re-verify candidate coordinates {best_loc}."
            )

        return MissionReport(
            mission_status=self.env.mission_status,
            eve_found=eve_found,
            termination_reason=self.env.termination_reason or "Completed",
            total_steps=self.env.step_count,
            scans_performed=self.env.scans_performed,
            battery_consumed=eval_gt["total_battery_consumed"],
            time_consumed=eval_gt["total_time_consumed"],
            battery_remaining=round(self.env.battery, 2),
            time_remaining=self.env.time_remaining,
            final_inferred_location=best_loc,
            final_inferred_status=best_stat,
            location_confidence=round(loc_conf, 4),
            status_confidence=round(stat_conf, 4),
            ground_truth_location=gt_loc,
            ground_truth_status=gt_stat,
            location_error_distance=round(loc_error, 2),
            matches_ground_truth=matches_gt,
            evidence_gathered=self.agent.get_evidence_gathered(),
            recommended_next_action=rec,
        )

    def export_json(self, report: MissionReport, file_path: str | Path) -> None:
        """Export report to a formatted JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2)

    def export_markdown(self, report: MissionReport, file_path: str | Path) -> None:
        """Export report as a formatted Markdown summary document."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        status_emoji = "✅" if report.eve_found else "❌"
        md = f"""# WALL-E Mission Report: EVE Signal Search

## Executive Summary
- **Outcome**: {status_emoji} **{report.mission_status.value}**
- **EVE Located**: {"Yes" if report.eve_found else "No"}
- **Termination Reason**: {report.termination_reason}

## Resource & Performance Metrics
| Metric | Value |
|---|---|
| Total Steps Executed | {report.total_steps} |
| Active Sensor Scans | {report.scans_performed} |
| Battery Consumed | {report.battery_consumed:.1f} / {self.env.config.battery_capacity:.1f} |
| Battery Remaining | {report.battery_remaining:.1f} |
| Time Consumed | {report.time_consumed} / {self.env.config.time_budget} |
| Time Remaining | {report.time_remaining} |

## Inferred State vs. Ground Truth
| Parameter | Inferred (Agent) | Ground Truth (Simulator) | Match |
|---|---|---|---|
| **Location Coordinates** | {report.final_inferred_location} | {report.ground_truth_location} | {"MATCH" if report.final_inferred_location == report.ground_truth_location else f"ERROR: {report.location_error_distance:.2f} tiles"} |
| **Location Confidence** | {report.location_confidence * 100:.1f}% | 100.0% | - |
| **Operational Status** | {report.final_inferred_status.value} | {report.ground_truth_status.value} | {"MATCH" if report.final_inferred_status == report.ground_truth_status else "MISMATCH"} |
| **Status Confidence** | {report.status_confidence * 100:.1f}% | 100.0% | - |

## Environmental Evidence Gathered
"""
        if report.evidence_gathered:
            for ev in report.evidence_gathered:
                md += f"- {ev}\n"
        else:
            md += "- *No environmental clues discovered before mission termination.*\n"

        md += f"""
## Recommended Next Action
> {report.recommended_next_action}
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
