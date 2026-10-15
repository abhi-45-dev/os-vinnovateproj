"""
Bayesian grid belief system for WALL-E and EVE's Signal.
Maintains a 2D spatial probability distribution over grid cells and
a discrete probability distribution over EVE's possible operational statuses.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from wall_e_agent.models import (
    Clue,
    ClueType,
    EveStatus,
    MissionConfig,
    SignalReading,
)


class BayesianGridBelief:
    """
    Probabilistic state estimator modeling spatial and status uncertainty.
    Uses recursive Bayesian filtering on a discrete 2D spatial grid.
    """

    def __init__(self, config: MissionConfig):
        self.width = config.grid_width
        self.height = config.grid_height
        self.obstacles: Set[Tuple[int, int]] = set(config.obstacles)
        self.sensor_config = config.sensor_config

        # 2D Spatial probability distribution grid: P(L = (x, y))
        # Initialized uniformly over all non-obstacle cells
        self.grid = np.zeros((self.height, self.width), dtype=np.float64)
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) not in self.obstacles:
                    self.grid[y, x] = 1.0

        # Status belief distribution: P(Status = s)
        self.status_belief: Dict[EveStatus, float] = {
            s: 1.0 / len(EveStatus) for s in EveStatus
        }

        self._normalize_grid()
        self._normalize_status()

        # Incorporate initial signal reading if provided in config
        if config.initial_signal:
            init_sig = SignalReading(
                strength=config.initial_signal.get("strength", 0.5),
                bearing=config.initial_signal.get("bearing", 45.0),
                noisy_distance=config.initial_signal.get("noisy_distance", 10.0),
                timestamp=0,
            )
            self.update_from_signal(init_sig, observer_pos=config.walle_start)

    def _normalize_grid(self) -> None:
        """Ensure the spatial grid sums exactly to 1.0."""
        # Zero out obstacles strictly
        for ox, oy in self.obstacles:
            self.grid[oy, ox] = 0.0

        total = np.sum(self.grid)
        if total > 1e-12:
            self.grid /= total
        else:
            # Numerical fallback: reset uniform over non-obstacle cells
            count = 0
            for y in range(self.height):
                for x in range(self.width):
                    if (x, y) not in self.obstacles:
                        self.grid[y, x] = 1.0
                        count += 1
            if count > 0:
                self.grid /= count

    def _normalize_status(self) -> None:
        """Ensure status belief distribution sums exactly to 1.0."""
        total = sum(self.status_belief.values())
        if total > 1e-12:
            for s in self.status_belief:
                self.status_belief[s] /= total
        else:
            uniform = 1.0 / len(EveStatus)
            for s in self.status_belief:
                self.status_belief[s] = uniform

    def update_from_signal(
        self,
        signal: SignalReading,
        observer_pos: Tuple[int, int],
        dist_sigma_scale: float = 1.0,
        bearing_sigma_scale: float = 1.0,
    ) -> None:
        """
        Bayesian update step conditioning spatial probability on a noisy signal reading.
        Computes joint Gaussian likelihood of distance and circular bearing.
        """
        ox, oy = observer_pos
        sigma_d = max(0.5, self.sensor_config.signal_noise_std * dist_sigma_scale * 2.5)
        sigma_b = max(5.0, self.sensor_config.bearing_noise_std * bearing_sigma_scale)

        likelihood = np.zeros_like(self.grid)

        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in self.obstacles:
                    continue

                dx = x - ox
                dy = y - oy
                cand_dist = math.hypot(dx, dy)

                if cand_dist < 1e-6:
                    cand_bearing = signal.bearing
                else:
                    cand_bearing = math.degrees(math.atan2(dy, dx)) % 360.0

                # Radial path-loss distance likelihood
                d_diff = cand_dist - signal.noisy_distance
                p_dist = math.exp(-0.5 * (d_diff / sigma_d) ** 2)

                # Circular bearing difference
                b_diff = abs(cand_bearing - signal.bearing)
                b_diff = min(b_diff, 360.0 - b_diff)
                p_bearing = math.exp(-0.5 * (b_diff / sigma_b) ** 2)

                likelihood[y, x] = p_dist * p_bearing

        # Posterior update: P(L | z) proportional to P(L) * P(z | L)
        self.grid = self.grid * likelihood
        self._normalize_grid()

    def update_from_negative_observation(
        self,
        observer_pos: Tuple[int, int],
        radius: int,
        false_negative_rate: float = 1e-5,
    ) -> None:
        """
        Discount probability of cells in range where EVE was proven absent.
        """
        ox, oy = observer_pos
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in self.obstacles:
                    continue
                dist = math.hypot(x - ox, y - oy)
                if dist <= radius:
                    self.grid[y, x] *= false_negative_rate

        self._normalize_grid()

    def update_from_clue(self, clue: Clue, observer_pos: Tuple[int, int]) -> None:
        """
        Incorporate evidence discovered from environmental clues.
        """
        ox, oy = observer_pos

        if clue.clue_type == ClueType.BEACON_PING:
            est_dist = clue.data.get("estimated_distance", 5.0)
            sigma = clue.data.get("noise_std", 1.5)
            likelihood = np.zeros_like(self.grid)

            for y in range(self.height):
                for x in range(self.width):
                    if (x, y) in self.obstacles:
                        continue
                    d = math.hypot(x - ox, y - oy)
                    likelihood[y, x] = math.exp(-0.5 * ((d - est_dist) / sigma) ** 2)

            self.grid = self.grid * likelihood
            self._normalize_grid()

        elif clue.clue_type == ClueType.ION_TRAIL:
            heading = clue.data.get("heading_degrees", 0.0)
            spread = clue.data.get("spread_degrees", 40.0)

            likelihood = np.zeros_like(self.grid)
            for y in range(self.height):
                for x in range(self.width):
                    if (x, y) in self.obstacles or (x == ox and y == oy):
                        continue
                    ang = math.degrees(math.atan2(y - oy, x - ox)) % 360.0
                    diff = abs(ang - heading)
                    diff = min(diff, 360.0 - diff)
                    if diff <= spread:
                        # Cone confidence
                        likelihood[y, x] = 1.0 - 0.5 * (diff / spread)
                    else:
                        likelihood[y, x] = 0.05

            self.grid = self.grid * likelihood
            self._normalize_grid()

        elif clue.clue_type == ClueType.SECTOR_CLEAR:
            bounds = clue.data.get("bounds", [0, 0, 1, 1])
            x1, y1, x2, y2 = bounds
            for y in range(max(0, y1), min(self.height, y2 + 1)):
                for x in range(max(0, x1), min(self.width, x2 + 1)):
                    self.grid[y, x] *= 1e-6
            self._normalize_grid()

        elif clue.clue_type == ClueType.STATUS_LOG:
            stat_name = clue.data.get("status_fragment")
            conf = clue.data.get("confidence", 0.90)
            if stat_name and stat_name in EveStatus.__members__:
                target_status = EveStatus(stat_name)
                for s in self.status_belief:
                    if s == target_status:
                        self.status_belief[s] *= conf
                    else:
                        self.status_belief[s] *= (1.0 - conf) / (len(EveStatus) - 1)
                self._normalize_status()

    def get_entropy(self) -> float:
        """
        Compute spatial Shannon entropy H = -sum(p * log2(p)).
        Higher values indicate high spatial uncertainty; lower values indicate confidence.
        """
        nonzero = self.grid[self.grid > 1e-12]
        return float(-np.sum(nonzero * np.log2(nonzero)))

    def get_top_candidates(self, k: int = 5) -> List[Tuple[Tuple[int, int], float]]:
        """Return the top-k highest probability grid cells sorted in descending order."""
        flat_indices = np.argsort(self.grid.ravel())[::-1]
        top = []
        for idx in flat_indices[:k]:
            y, x = divmod(idx, self.width)
            prob = float(self.grid[y, x])
            if prob > 0.0:
                top.append(((x, y), round(prob, 4)))
        return top

    def get_most_likely_location(self) -> Tuple[Tuple[int, int], float]:
        """Return the single cell with the highest posterior probability."""
        idx = int(np.argmax(self.grid))
        y, x = divmod(idx, self.width)
        return (x, y), float(self.grid[y, x])

    def get_status_distribution(self) -> Dict[EveStatus, float]:
        """Return the current categorical probability distribution over EVE's status."""
        return {s: round(p, 4) for s, p in self.status_belief.items()}

    def get_most_likely_status(self) -> Tuple[EveStatus, float]:
        """Return the status with highest confidence."""
        best_stat = max(self.status_belief.items(), key=lambda item: item[1])
        return best_stat[0], round(best_stat[1], 4)

    def get_grid_copy(self) -> np.ndarray:
        """Return a defensive copy of the probability density grid."""
        return self.grid.copy()
