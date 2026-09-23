"""Thread-safe perception mode and generation tracking."""

from threading import Lock


OFF = "OFF"
FACE = "FACE"
GESTURE = "GESTURE"
MODES = (OFF, FACE, GESTURE)


def normalize_mode(value):
    """Return a canonical mode from a string or a JSON-decoded command."""
    if isinstance(value, dict):
        value = value.get("mode")
    if not isinstance(value, str):
        raise ValueError("perception mode must be OFF, FACE, or GESTURE")
    normalized = value.strip().upper()
    aliases = {
        "NONE": OFF,
        "DISABLED": OFF,
        "HAND": GESTURE,
        "HANDS": GESTURE,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in MODES:
        raise ValueError("perception mode must be OFF, FACE, or GESTURE")
    return normalized


class ModeState:
    """Serialize mode changes and invalidate in-flight work by generation."""

    def __init__(self, initial=OFF):
        self._lock = Lock()
        self._mode = normalize_mode(initial)
        self._generation = 0

    def snapshot(self):
        with self._lock:
            return self._mode, self._generation

    def switch(self, requested):
        requested = normalize_mode(requested)
        with self._lock:
            if requested == self._mode:
                return self._mode, self._generation, False
            self._mode = requested
            self._generation += 1
            return self._mode, self._generation, True

