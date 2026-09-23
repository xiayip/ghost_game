"""Map game phases to the perception backend required by that phase."""


def validate_phase_modes(face_phases, gesture_phases):
    face = {str(value).strip() for value in face_phases}
    gesture = {str(value).strip() for value in gesture_phases}
    if "" in face or "" in gesture:
        raise ValueError("perception phase names must be non-empty")
    overlap = face & gesture
    if overlap:
        raise ValueError(
            "perception face/gesture phase lists overlap: " + ", ".join(sorted(overlap))
        )
    return face, gesture


def perception_mode_for_phase(phase, face_phases, gesture_phases):
    if phase in face_phases:
        return "FACE"
    if phase in gesture_phases:
        return "GESTURE"
    return "OFF"

