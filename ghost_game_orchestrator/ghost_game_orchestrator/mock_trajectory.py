"""Pure helpers for the development-only Ghost auto-solve trajectory."""


def quintic_blend(progress):
    """Return a clamped minimum-jerk blend in the range [0, 1]."""
    u = max(0.0, min(1.0, float(progress)))
    return 10.0 * u ** 3 - 15.0 * u ** 4 + 6.0 * u ** 5


def minimum_quintic_duration(start, target, max_velocity, minimum):
    """Choose a duration whose quintic peak velocity stays below the limit.

    The derivative of ``10u^3 - 15u^4 + 6u^5`` peaks at 1.875, so each
    joint needs at least ``1.875 * abs(delta) / velocity_limit`` seconds.
    """
    duration = float(minimum)
    for current, goal, velocity in zip(start, target, max_velocity):
        duration = max(duration, 1.875 * abs(goal - current) / velocity)
    return duration


def trajectory_reference(start, target, locked, locked_hold, progress):
    """Interpolate unlocked joints while preserving achieved lock positions."""
    blend = quintic_blend(progress)
    return [
        locked_hold[index] if locked[index] else
        start[index] + blend * (target[index] - start[index])
        for index in range(len(start))
    ]
