"""
Unit tests for BayesianGridBelief: normalization, Bayesian likelihood updates,
negative observations, clue integration, and edge cases.
"""

import math
import numpy as np
import pytest

from wall_e_agent.belief import BayesianGridBelief
from wall_e_agent.models import (
    Clue,
    ClueType,
    EveStatus,
    MissionConfig,
    SignalReading,
)


@pytest.fixture
def belief_model():
    cfg = MissionConfig(
        grid_width=8,
        grid_height=8,
        walle_start=(0, 0),
        obstacles=[(2, 2), (2, 3)],
        eve_location=(6, 6),
        eve_status=EveStatus.OPERATIONAL,
    )
    return BayesianGridBelief(cfg)


def test_initialization_and_normalization(belief_model):
    grid = belief_model.get_grid_copy()

    # Sum of probability over entire grid must equal 1.0
    assert pytest.approx(np.sum(grid), rel=1e-5) == 1.0

    # Obstacles must have zero probability
    assert grid[2, 2] == 0.0
    assert grid[3, 2] == 0.0

    # Status distribution sums to ~1.0 (with roundoff tolerance)
    status_dist = belief_model.get_status_distribution()
    assert pytest.approx(sum(status_dist.values()), abs=1e-3) == 1.0
    assert pytest.approx(sum(belief_model.status_belief.values()), rel=1e-6) == 1.0


def test_signal_reading_update_concentrates_probability(belief_model):
    initial_entropy = belief_model.get_entropy()

    # Signal reading from (0, 0) pointing towards (6, 6)
    # Bearing = 45 degrees, distance = sqrt(36 + 36) = ~8.48
    signal = SignalReading(
        strength=0.35,
        bearing=45.0,
        noisy_distance=8.5,
        timestamp=1,
    )

    belief_model.update_from_signal(signal, observer_pos=(0, 0))

    grid = belief_model.get_grid_copy()
    assert pytest.approx(np.sum(grid), rel=1e-5) == 1.0

    # Cells in the 45-degree direction near distance 8.5 (e.g., (6, 6)) should have much higher probability
    # than cells in the opposite direction (e.g., (7, 0))
    p_target = grid[6, 6]
    p_opposite = grid[0, 7]
    assert p_target > p_opposite * 5.0

    # Entropy should decrease as belief becomes more concentrated
    new_entropy = belief_model.get_entropy()
    assert new_entropy < initial_entropy


def test_negative_observation_discounts_cells(belief_model):
    # Scan from (3, 3) with radius 2
    grid_before = belief_model.get_grid_copy()
    p_center_before = grid_before[3, 3]

    belief_model.update_from_negative_observation(observer_pos=(3, 3), radius=2)

    grid_after = belief_model.get_grid_copy()
    assert pytest.approx(np.sum(grid_after), rel=1e-5) == 1.0

    # (3, 3) was directly scanned, its probability should be near 0
    assert grid_after[3, 3] < 1e-4
    # Cells outside scan radius (e.g. (7, 7)) should have increased relative probability
    assert grid_after[7, 7] > grid_before[7, 7]


def test_clue_updates(belief_model):
    # Sector clear clue: bounds [0, 0, 3, 3]
    sector_clue = Clue(
        clue_id="sc1",
        clue_type=ClueType.SECTOR_CLEAR,
        description="Sector (0-3, 0-3) clear",
        data={"bounds": [0, 0, 3, 3]},
    )
    belief_model.update_from_clue(sector_clue, observer_pos=(0, 0))

    grid = belief_model.get_grid_copy()
    assert pytest.approx(np.sum(grid), rel=1e-5) == 1.0
    # Any cell in cleared sector should be near zero
    assert grid[1, 1] < 1e-4

    # Status clue: fragment indicates STANDBY with 90% confidence
    status_clue = Clue(
        clue_id="st1",
        clue_type=ClueType.STATUS_LOG,
        description="Log standby",
        data={"status_fragment": "STANDBY", "confidence": 0.90},
    )
    belief_model.update_from_clue(status_clue, observer_pos=(0, 0))

    status_dist = belief_model.get_status_distribution()
    assert pytest.approx(sum(status_dist.values()), rel=1e-5) == 1.0
    best_status, conf = belief_model.get_most_likely_status()
    assert best_status == EveStatus.STANDBY
    assert conf > 0.60


def test_contradictory_readings_graceful_fallback(belief_model):
    # If contradictory negative observations zero out all mass,
    # belief system must not crash or yield NaN; it resets to uniform fallback
    belief_model.update_from_negative_observation(observer_pos=(4, 4), radius=100)

    grid = belief_model.get_grid_copy()
    assert not np.isnan(grid).any()
    assert not np.isinf(grid).any()
    assert pytest.approx(np.sum(grid), rel=1e-5) == 1.0
