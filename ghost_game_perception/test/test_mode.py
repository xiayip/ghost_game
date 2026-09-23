import pytest

from ghost_game_perception.mode import FACE, GESTURE, OFF, ModeState, normalize_mode


def test_normalize_mode_accepts_commands_and_aliases():
    assert normalize_mode(" face ") == FACE
    assert normalize_mode({"mode": "gesture"}) == GESTURE
    assert normalize_mode("hands") == GESTURE
    assert normalize_mode("none") == OFF


@pytest.mark.parametrize("value", [None, 1, {}, "camera", []])
def test_normalize_mode_rejects_unknown_values(value):
    with pytest.raises(ValueError):
        normalize_mode(value)


def test_generation_changes_only_when_effective_mode_changes():
    state = ModeState()
    assert state.snapshot() == (OFF, 0)
    assert state.switch("off") == (OFF, 0, False)
    assert state.switch("face") == (FACE, 1, True)
    assert state.switch("FACE") == (FACE, 1, False)
    assert state.switch("gesture") == (GESTURE, 2, True)

