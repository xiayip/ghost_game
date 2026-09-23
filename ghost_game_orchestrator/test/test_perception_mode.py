import pytest

from ghost_game_orchestrator.perception_mode import (
    perception_mode_for_phase,
    validate_phase_modes,
)


def test_phase_mode_mapping_is_explicit_and_defaults_off():
    face, gesture = validate_phase_modes(["face_searching"], ["gesture_interaction"])
    assert perception_mode_for_phase("face_searching", face, gesture) == "FACE"
    assert perception_mode_for_phase("gesture_interaction", face, gesture) == "GESTURE"
    assert perception_mode_for_phase("searching", face, gesture) == "OFF"


def test_phase_lists_must_not_overlap():
    with pytest.raises(ValueError):
        validate_phase_modes(["active"], ["active"])

