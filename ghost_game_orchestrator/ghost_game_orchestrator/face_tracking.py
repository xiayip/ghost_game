"""ROS-independent helpers for stable face acquisition and image centering."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FaceSample:
    received_at: float
    center_x: float
    center_y: float
    width: float
    height: float
    image_width: int
    image_height: int
    score: float

    @property
    def area_ratio(self):
        return ((self.width * self.height) /
                (self.image_width * self.image_height))

    @property
    def error_x(self):
        """Horizontal center error normalized to the image half-width."""
        return (self.center_x - self.image_width / 2.0) / (
            self.image_width / 2.0)

    @property
    def error_y(self):
        """Vertical center error normalized to the image half-height."""
        return (self.center_y - self.image_height / 2.0) / (
            self.image_height / 2.0)


def make_face_sample(
        received_at, center_x, center_y, width, height,
        image_width, image_height, score):
    values = (
        received_at, center_x, center_y, width, height,
        image_width, image_height, score,
    )
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError('face sample values must be finite')
    if width <= 0.0 or height <= 0.0:
        raise ValueError('face box dimensions must be positive')
    if image_width <= 0 or image_height <= 0:
        raise ValueError('image dimensions must be positive')
    return FaceSample(
        received_at=float(received_at),
        center_x=float(center_x),
        center_y=float(center_y),
        width=float(width),
        height=float(height),
        image_width=int(image_width),
        image_height=int(image_height),
        score=float(score),
    )


def face_is_acceptable(sample, now, max_age, min_area_ratio, max_area_ratio):
    if sample is None:
        return False
    age = now - sample.received_at
    return (
        0.0 <= age <= max_age and
        min_area_ratio <= sample.area_ratio <= max_area_ratio
    )


def face_matches_track(
        previous, current, center_jump_tolerance,
        area_relative_jump_tolerance):
    """Reject a detector identity switch while visual servoing one face."""
    if previous is None or current is None:
        return False
    center_jump = max(
        abs(current.error_x - previous.error_x),
        abs(current.error_y - previous.error_y),
    )
    area_jump = abs(
        current.area_ratio - previous.area_ratio
    ) / max(previous.area_ratio, 1e-9)
    return (
        center_jump <= center_jump_tolerance and
        area_jump <= area_relative_jump_tolerance
    )


class FaceStabilityGate:
    """Require one face to remain spatially stable for a dwell period."""

    def __init__(self, dwell_time, center_tolerance, area_relative_tolerance):
        if dwell_time <= 0.0:
            raise ValueError('face stability dwell_time must be positive')
        if center_tolerance <= 0.0:
            raise ValueError('face stability center_tolerance must be positive')
        if area_relative_tolerance <= 0.0:
            raise ValueError(
                'face stability area_relative_tolerance must be positive')
        self.dwell_time = dwell_time
        self.center_tolerance = center_tolerance
        self.area_relative_tolerance = area_relative_tolerance
        self.reset()

    def reset(self):
        self._reference = None
        self._stable_since = None
        self._last_stable_for = 0.0

    @property
    def stable_for(self):
        return self._last_stable_for

    def update(self, sample, now):
        if sample is None:
            self.reset()
            return False

        if self._reference is not None:
            center_moved = max(
                abs(sample.error_x - self._reference.error_x),
                abs(sample.error_y - self._reference.error_y),
            )
            area_change = abs(
                sample.area_ratio - self._reference.area_ratio
            ) / max(self._reference.area_ratio, 1e-9)
            if (center_moved > self.center_tolerance or
                    area_change > self.area_relative_tolerance):
                self.reset()

        if self._reference is None:
            self._reference = sample
            self._stable_since = now

        self._last_stable_for = max(0.0, now - self._stable_since)
        return self._last_stable_for >= self.dwell_time


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


def scan_target(
        reference, lower_limits, upper_limits, yaw_index, pitch_index,
        yaw_offset, pitch_offset, joint_margin=0.05):
    target = list(reference)
    target[yaw_index] = _clamp(
        reference[yaw_index] + yaw_offset,
        lower_limits[yaw_index] + joint_margin,
        upper_limits[yaw_index] - joint_margin,
    )
    target[pitch_index] = _clamp(
        reference[pitch_index] + pitch_offset,
        lower_limits[pitch_index] + joint_margin,
        upper_limits[pitch_index] - joint_margin,
    )
    return target


def servo_target(
        current_command, reference, lower_limits, upper_limits,
        yaw_index, pitch_index, error_x, error_y,
        yaw_gain, pitch_gain, max_speed, dt,
        yaw_direction, pitch_direction,
        yaw_max_offset, pitch_max_offset, deadband,
        joint_margin=0.05):
    """Integrate one bounded visual-servo command step.

    ``yaw_gain`` and ``pitch_gain`` map normalized image error to joint
    velocity in rad/s. Holding that velocity between detector frames lets a
    higher-rate command loop drive the impedance controller smoothly while
    ``max_speed`` remains an explicit physical safety bound.
    """
    if not math.isfinite(float(dt)) or dt < 0.0:
        raise ValueError('servo dt must be finite and non-negative')
    if max_speed <= 0.0:
        raise ValueError('servo max_speed must be positive')

    target = list(current_command)
    yaw_error = 0.0 if abs(error_x) <= deadband else error_x
    pitch_error = 0.0 if abs(error_y) <= deadband else error_y
    yaw_velocity = _clamp(
        yaw_direction * yaw_gain * yaw_error, -max_speed, max_speed)
    pitch_velocity = _clamp(
        pitch_direction * pitch_gain * pitch_error, -max_speed, max_speed)

    yaw_lower = max(
        lower_limits[yaw_index] + joint_margin,
        reference[yaw_index] - yaw_max_offset,
    )
    yaw_upper = min(
        upper_limits[yaw_index] - joint_margin,
        reference[yaw_index] + yaw_max_offset,
    )
    pitch_lower = max(
        lower_limits[pitch_index] + joint_margin,
        reference[pitch_index] - pitch_max_offset,
    )
    pitch_upper = min(
        upper_limits[pitch_index] - joint_margin,
        reference[pitch_index] + pitch_max_offset,
    )

    target[yaw_index] = _clamp(
        current_command[yaw_index] + yaw_velocity * dt,
        yaw_lower, yaw_upper)
    target[pitch_index] = _clamp(
        current_command[pitch_index] + pitch_velocity * dt,
        pitch_lower, pitch_upper)
    return target
