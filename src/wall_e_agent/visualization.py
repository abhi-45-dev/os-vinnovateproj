"""
Visualization tools for WALL-E and EVE's Signal.
Provides ASCII map generation and Matplotlib visual inspection plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np

from wall_e_agent.agent import AutonomousAgent
from wall_e_agent.environment import GridEnvironment
from wall_e_agent.models import MissionReport, TraceStep


class GridVisualizer:
    """
    Renders visual representations of the mission map, agent trajectory,
    Bayesian belief probability distributions, and resource decay curves.
    """

    def __init__(self, env: GridEnvironment, agent: AutonomousAgent):
        self.env = env
        self.agent = agent
        self.width = env.width
        self.height = env.height

    def render_ascii(self, show_ground_truth: bool = False) -> str:
        """
        Generate a terminal-friendly ASCII representation of the map.
        Legend:
          W: WALL-E current position
          ·: Trajectory path
          #: Obstacle
          ⚡: Recharge station
          ?: Discovered clue
          ★: Highest belief candidate
          E: EVE ground truth (post-mission audit only)
        """
        trajectory_set: Set[Tuple[int, int]] = set(self.agent.get_trajectory())
        curr_pos = self.env.agent_position
        best_coord, _ = self.agent.belief.get_most_likely_location()
        recharge_set = set(self.env.recharge_stations)
        obstacle_set = set(self.env.obstacles)

        discovered_coords = set()
        for c in self.env._internal_clues:
            if c.clue_id in self.env.discovered_clue_ids and c.coordinates:
                discovered_coords.add(c.coordinates)

        lines = []
        border = "+" + "---" * self.width + "+"
        lines.append(border)

        for y in range(self.height - 1, -1, -1):
            row_chars = []
            for x in range(self.width):
                coord = (x, y)
                if coord == curr_pos:
                    char = " W "
                elif show_ground_truth and coord == self.env._eve_location:
                    char = " E "
                elif coord == best_coord:
                    char = " ★ "
                elif coord in discovered_coords:
                    char = " ? "
                elif coord in recharge_set:
                    char = " R "
                elif coord in obstacle_set:
                    char = "###"
                elif coord in trajectory_set:
                    char = " · "
                else:
                    char = " . "
                row_chars.append(char)
            lines.append("|" + "".join(row_chars) + "|")

        lines.append(border)
        lines.append("Legend: W=WALL-E, ·=Path, ###=Obstacle, R=Recharge, ?=Clue, ★=Belief Peak" + (", E=EVE Ground Truth" if show_ground_truth else ""))
        return "\n".join(lines)

    def render_matplotlib(
        self,
        trace_steps: List[TraceStep],
        report: MissionReport,
        output_path: str | Path,
    ) -> bool:
        """
        Generate high-resolution Matplotlib inspection charts:
          1. Agent's Bayesian Belief Heatmap & Trajectory
          2. Post-Mission Ground Truth Evaluation Map
          3. Resource (Battery/Time) and Shannon Entropy Progression
        """
        try:
            import matplotlib
            matplotlib.use("Agg")  # Headless backend
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            print("Matplotlib not installed. Skipping graphical plot.")
            return False

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6.5))

        # --- Panel 1: Agent's Live Bayesian Belief Heatmap ---
        grid_belief = self.agent.belief.get_grid_copy()
        # Flip vertically so (0,0) is bottom-left
        im1 = ax1.imshow(grid_belief, origin="lower", cmap="magma", interpolation="nearest")
        fig.colorbar(im1, ax=ax1, label="Posterior Probability P(EVE at cell)")

        # Plot obstacles
        for ox, oy in self.env.obstacles:
            ax1.add_patch(patches.Rectangle((ox - 0.5, oy - 0.5), 1, 1, facecolor="gray", hatch="//"))

        # Plot recharge stations
        for rx, ry in self.env.recharge_stations:
            ax1.plot(rx, ry, marker="s", markersize=10, color="cyan", markeredgecolor="black", label="Recharge Station")

        # Plot Trajectory
        traj = self.agent.get_trajectory()
        if len(traj) > 1:
            tx, ty = zip(*traj)
            ax1.plot(tx, ty, color="white", linestyle="--", linewidth=1.5, alpha=0.8, label="WALL-E Path")
            ax1.plot(tx[0], ty[0], marker="o", markersize=8, color="green", label="Start")
            ax1.plot(tx[-1], ty[-1], marker="o", markersize=8, color="yellow", label="WALL-E Final")

        # Plot peak belief
        best_loc, best_p = self.agent.belief.get_most_likely_location()
        ax1.plot(best_loc[0], best_loc[1], marker="*", markersize=14, color="lime", markeredgecolor="black", label=f"Belief Peak ({best_p:.2f})")

        ax1.set_title(f"Agent's Bayesian Belief Map\nEntropy H = {self.agent.belief.get_entropy():.2f}")
        ax1.set_xlim(-0.5, self.width - 0.5)
        ax1.set_ylim(-0.5, self.height - 0.5)
        ax1.set_xticks(range(self.width))
        ax1.set_yticks(range(self.height))
        ax1.grid(color="white", alpha=0.15, linestyle=":")
        ax1.legend(loc="upper right", fontsize=8)

        # --- Panel 2: Ground Truth Audit Map ---
        ax2.set_facecolor("#f8f9fa")
        first_obs = next(iter(self.env.obstacles)) if self.env.obstacles else None
        for ox, oy in self.env.obstacles:
            ax2.add_patch(patches.Rectangle((ox - 0.5, oy - 0.5), 1, 1, facecolor="#343a40", hatch="//", label="Obstacle" if (ox, oy) == first_obs else ""))

        for rx, ry in self.env.recharge_stations:
            ax2.plot(rx, ry, marker="s", markersize=10, color="#0d6efd", markeredgecolor="black")

        if len(traj) > 1:
            ax2.plot(tx, ty, color="#fd7e14", linestyle="-", linewidth=2.0, marker=".", markersize=5, label="Trajectory")

        # EVE True location
        gt_x, gt_y = self.env._eve_location
        ax2.plot(gt_x, gt_y, marker="P", markersize=14, color="#dc3545", markeredgecolor="black", label=f"EVE True ({gt_x}, {gt_y})")
        # Inferred location
        ax2.plot(best_loc[0], best_loc[1], marker="*", markersize=14, color="#198754", markeredgecolor="black", label=f"Inferred ({best_loc[0]}, {best_loc[1]})")

        ax2.set_title(f"Ground Truth Evaluation\nOutcome: {report.mission_status.value} (Error: {report.location_error_distance:.2f} tiles)")
        ax2.set_xlim(-0.5, self.width - 0.5)
        ax2.set_ylim(-0.5, self.height - 0.5)
        ax2.set_xticks(range(self.width))
        ax2.set_yticks(range(self.height))
        ax2.grid(color="gray", alpha=0.3, linestyle=":")
        ax2.legend(loc="upper right", fontsize=8)

        # --- Panel 3: Energy & Entropy Dynamics ---
        if trace_steps:
            steps_idx = [s.step_number for s in trace_steps]
            batt_levels = [s.battery_remaining for s in trace_steps]
            entropy_levels = [s.entropy for s in trace_steps]

            ax3_color = "tab:blue"
            ax3.set_xlabel("Simulation Step")
            ax3.set_ylabel("Battery Level", color=ax3_color)
            line1 = ax3.plot(steps_idx, batt_levels, color=ax3_color, linewidth=2, label="Battery")
            ax3.tick_params(axis="y", labelcolor=ax3_color)
            ax3.set_ylim(0, self.env.config.battery_capacity * 1.05)

            ax3_right = ax3.twinx()
            ax3_color2 = "tab:purple"
            ax3_right.set_ylabel("Spatial Shannon Entropy (bits)", color=ax3_color2)
            line2 = ax3_right.plot(steps_idx, entropy_levels, color=ax3_color2, linewidth=2, linestyle="--", label="Entropy H(L)")
            ax3_right.tick_params(axis="y", labelcolor=ax3_color2)

            lines_all = line1 + line2
            labels_all = [l.get_label() for l in lines_all]
            ax3.legend(lines_all, labels_all, loc="upper right")
            ax3.set_title("Resource Consumption & Information Gain")
            ax3.grid(alpha=0.3)

        plt.tight_layout()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150)
        plt.close(fig)
        return True
