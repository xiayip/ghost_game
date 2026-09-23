import pytest

from ghost_game_orchestrator.mock_trajectory import (
    minimum_quintic_duration,
    quintic_blend,
    trajectory_reference,
)


def test_quintic_blend_clamps_and_has_smooth_midpoint():
    assert quintic_blend(-1.0) == 0.0
    assert quintic_blend(0.0) == 0.0
    assert quintic_blend(0.5) == pytest.approx(0.5)
    assert quintic_blend(1.0) == 1.0
    assert quintic_blend(2.0) == 1.0


def test_duration_respects_quintic_peak_velocity():
    duration = minimum_quintic_duration(
        [0.0, 0.0], [1.0, 0.2], [0.5, 0.1], 1.0)

    assert duration == pytest.approx(3.75)


def test_reference_moves_only_unlocked_joints():
    reference = trajectory_reference(
        start=[0.0, 1.0, 2.0],
        target=[1.0, 2.0, 3.0],
        locked=[False, True, False],
        locked_hold=[0.0, 1.25, 2.0],
        progress=0.5,
    )

    assert reference == pytest.approx([0.5, 1.25, 2.5])
