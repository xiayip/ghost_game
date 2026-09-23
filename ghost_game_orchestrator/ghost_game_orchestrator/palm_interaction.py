"""Depth-to-motion mapping and a bounded Cartesian servo for the palm game."""

from dataclasses import dataclass
import math

import numpy as np


# Joint origins and axes are the commissioned reBot arm URDF values. The fixed
# transforms end at camera_left_color_optical_frame, whose +Z axis points out
# of the camera toward the visitor.
_JOINTS = (
    ((-0.00034283, -0.00098683, 0.075), (0.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    ((0.020343, 0.027237, 0.07), (-1.5708, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((-0.236, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    ((0.228, -0.072746, 0.0045), (0.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    ((0.087, -0.048, -0.03075), (-1.5708, 0.0, 0.0), (0.0, 0.0, -1.0)),
    ((0.0365, 0.0, 0.048), (0.0, 1.5708, 0.0), (0.0, 0.0, -1.0)),
)
_CAMERA_FIXED = (
    ((0.0, 0.0, 0.0), (0.0, -math.pi / 2.0, math.pi)),
    ((0.024, 0.0, 0.05), (0.0, 0.0, 0.0)),
    ((0.008126703388352, 0.009, 0.0210000000000001), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (-math.pi / 2.0, 0.0, -math.pi / 2.0)),
)


def _rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=float)
    ry = np.array(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=float)
    rz = np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=float)
    return rz @ ry @ rx


def _transform(xyz, rpy):
    result = np.eye(4)
    result[:3, :3] = _rpy_matrix(rpy)
    result[:3, 3] = xyz
    return result


def _axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cosine, sine, complement = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array(
        (
            (x * x * complement + cosine, x * y * complement - z * sine,
             x * z * complement + y * sine),
            (y * x * complement + z * sine, y * y * complement + cosine,
             y * z * complement - x * sine),
            (z * x * complement - y * sine, z * y * complement + x * sine,
             z * z * complement + cosine),
        ),
        dtype=float,
    )


def camera_pose_and_jacobian(joints):
    """Return camera optical pose and a base-frame 6x6 geometric Jacobian."""
    if len(joints) != 6 or not all(math.isfinite(float(value)) for value in joints):
        raise ValueError("camera kinematics requires six finite joint positions")
    transform = np.eye(4)
    joint_origins = []
    joint_axes = []
    for value, (xyz, rpy, axis) in zip(joints, _JOINTS):
        transform = transform @ _transform(xyz, rpy)
        joint_origins.append(transform[:3, 3].copy())
        joint_axes.append(transform[:3, :3] @ np.asarray(axis, dtype=float))
        rotation = np.eye(4)
        rotation[:3, :3] = _axis_rotation(axis, float(value))
        transform = transform @ rotation
    for xyz, rpy in _CAMERA_FIXED:
        transform = transform @ _transform(xyz, rpy)

    position = transform[:3, 3].copy()
    jacobian = np.zeros((6, 6), dtype=float)
    for index, (origin, axis) in enumerate(zip(joint_origins, joint_axes)):
        jacobian[:3, index] = np.cross(axis, position - origin)
        jacobian[3:, index] = axis
    return position, transform[:3, :3].copy(), jacobian


@dataclass(frozen=True)
class ServoStep:
    positions: list
    velocities: list
    position_error_m: float
    offset_m: float


def palm_center_error_x(payload):
    """Return horizontal palm-center error normalized to image half-width."""
    if not isinstance(payload, dict) or payload.get("palm_open") is not True:
        return None
    center = payload.get("center")
    if (
        not isinstance(center, (list, tuple))
        or len(center) != 2
        or not all(math.isfinite(float(value)) for value in center)
    ):
        return None
    center_x = float(center[0])
    if not 0.0 <= center_x <= 1.0:
        return None
    return (center_x - 0.5) / 0.5


def bounded_joint_servo_target(
    current_target,
    reference,
    lower_limit,
    upper_limit,
    error,
    gain,
    max_speed,
    dt,
    direction,
    max_offset,
    deadband,
    joint_margin,
):
    """Integrate one bounded image-error command for a single joint."""
    values = (
        current_target, reference, lower_limit, upper_limit, error, gain,
        max_speed, dt, direction, max_offset, deadband, joint_margin,
    )
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("joint servo values must be finite")
    if gain <= 0.0 or max_speed <= 0.0 or max_offset <= 0.0 or dt < 0.0:
        raise ValueError("joint servo gain, speed, and limits are invalid")
    if deadband < 0.0 or joint_margin < 0.0 or direction == 0.0:
        raise ValueError("joint servo deadband, margin, or direction is invalid")
    lower = max(lower_limit + joint_margin, reference - max_offset)
    upper = min(upper_limit - joint_margin, reference + max_offset)
    if lower >= upper:
        raise ValueError("joint servo has no usable range")
    active_error = 0.0 if abs(error) <= deadband else error
    velocity = float(np.clip(
        direction * gain * active_error, -max_speed, max_speed
    ))
    return float(np.clip(current_target + velocity * dt, lower, upper))


class CameraAxisServo:
    """Keep camera orientation while translating along its initial optical axis."""

    def __init__(
        self,
        anchor,
        lower_limits,
        upper_limits,
        joint_max_offsets,
        max_linear_speed=0.06,
        max_joint_speed=0.35,
        position_gain=2.5,
        orientation_gain=2.0,
        damping=0.06,
        joint_margin=0.04,
    ):
        self.anchor = np.asarray(anchor, dtype=float)
        self.lower = np.asarray(lower_limits, dtype=float) + float(joint_margin)
        self.upper = np.asarray(upper_limits, dtype=float) - float(joint_margin)
        offsets = np.asarray(joint_max_offsets, dtype=float)
        if any(array.shape != (6,) for array in (self.anchor, self.lower, self.upper, offsets)):
            raise ValueError("camera servo vectors must contain six values")
        if np.any(offsets <= 0) or np.any(self.lower >= self.upper):
            raise ValueError("camera servo joint bounds are invalid")
        self.lower = np.maximum(self.lower, self.anchor - offsets)
        self.upper = np.minimum(self.upper, self.anchor + offsets)
        if np.any(self.lower >= self.upper):
            raise ValueError("camera servo anchor is outside its bounded joint workspace")
        self.max_linear_speed = float(max_linear_speed)
        self.max_joint_speed = float(max_joint_speed)
        self.position_gain = float(position_gain)
        self.orientation_gain = float(orientation_gain)
        self.damping = float(damping)
        if any(not math.isfinite(value) or value <= 0 for value in (
                self.max_linear_speed, self.max_joint_speed, self.position_gain,
                self.orientation_gain, self.damping)):
            raise ValueError("camera servo gains and limits must be positive")
        self.anchor_position, self.anchor_rotation, _ = camera_pose_and_jacobian(anchor)
        self.axis = self.anchor_rotation[:, 2].copy()

    @staticmethod
    def _limit_norm(vector, limit):
        norm = float(np.linalg.norm(vector))
        return vector if norm <= limit or norm == 0.0 else vector * (limit / norm)

    def step(
        self,
        command,
        target_offset,
        dt,
        reference=None,
        controlled_joint_index=None,
    ):
        command = np.asarray(command, dtype=float)
        if command.shape != (6,) or not np.all(np.isfinite(command)):
            raise ValueError("camera servo command must contain six finite values")
        if not math.isfinite(target_offset) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("camera servo target and dt must be finite and dt positive")
        if reference is None:
            reference = self.anchor
        reference = np.asarray(reference, dtype=float)
        if reference.shape != (6,) or not np.all(np.isfinite(reference)):
            raise ValueError("camera servo reference must contain six finite values")
        if (controlled_joint_index is not None and
                controlled_joint_index not in range(6)):
            raise ValueError("controlled joint index must be in [0, 6)")
        position, rotation, jacobian = camera_pose_and_jacobian(command)
        reference_position, reference_rotation, _ = camera_pose_and_jacobian(
            reference
        )
        desired_position = (
            reference_position
            + reference_rotation[:, 2] * float(target_offset)
        )
        linear = self.position_gain * (desired_position - position)
        linear = self._limit_norm(linear, self.max_linear_speed)
        angular_error = 0.5 * sum(
            np.cross(rotation[:, index], reference_rotation[:, index])
            for index in range(3)
        )
        angular = self._limit_norm(
            self.orientation_gain * angular_error, self.max_joint_speed
        )
        twist = np.concatenate((linear, angular))
        if controlled_joint_index is None:
            regularized = jacobian @ jacobian.T + self.damping**2 * np.eye(6)
            velocity = jacobian.T @ np.linalg.solve(regularized, twist)
        else:
            # Keep the requested gimbal joint explicit. Subtract its known
            # Cartesian contribution, then let the other five joints realize
            # the remaining depth and orientation motion in least squares.
            index = int(controlled_joint_index)
            direct_velocity = float(np.clip(
                (reference[index] - command[index]) / float(dt),
                -self.max_joint_speed,
                self.max_joint_speed,
            ))
            free_indices = [item for item in range(6) if item != index]
            reduced = jacobian[:, free_indices]
            residual = twist - jacobian[:, index] * direct_velocity
            regularized = (
                reduced @ reduced.T + self.damping**2 * np.eye(6)
            )
            free_velocity = reduced.T @ np.linalg.solve(
                regularized, residual
            )
            velocity = np.zeros(6, dtype=float)
            velocity[free_indices] = free_velocity
            velocity[index] = direct_velocity
        velocity = np.clip(velocity, -self.max_joint_speed, self.max_joint_speed)
        next_command = np.clip(command + velocity * float(dt), self.lower, self.upper)
        actual_velocity = (next_command - command) / float(dt)
        return ServoStep(
            positions=next_command.tolist(),
            velocities=actual_velocity.tolist(),
            position_error_m=float(np.linalg.norm(desired_position - position)),
            offset_m=float(target_offset),
        )


class PalmDepthFilter:
    """Establish a neutral palm depth and turn displacement into arm offset.

    A brief bad frame holds the last bounded position target; a sustained loss
    stops motion without necessarily erasing the neutral depth. Gesture
    inference can briefly report no-hand, arming, identity_lost, or an invalid
    depth sample while the same visible hand is still on screen. Keeping both
    the target and neutral through short dropouts removes stop/start jitter.
    """

    def __init__(self, deadzone=0.025, gain=1.0, max_offset=0.08,
                 smoothing_time=0.12, sample_timeout=0.20,
                 reacquire_timeout=0.75):
        self.deadzone = float(deadzone)
        self.gain = float(gain)
        self.max_offset = float(max_offset)
        self.smoothing_time = float(smoothing_time)
        self.sample_timeout = float(sample_timeout)
        self.reacquire_timeout = float(reacquire_timeout)
        if any(not math.isfinite(value) or value <= 0 for value in (
                self.deadzone, self.gain, self.max_offset,
                self.smoothing_time, self.sample_timeout,
                self.reacquire_timeout)):
            raise ValueError("palm filter limits must be finite and positive")
        if self.reacquire_timeout < self.sample_timeout:
            raise ValueError(
                "palm reacquire timeout must be at least the sample timeout")
        self.reset()

    def reset(self):
        self.hand_id = None
        self.baseline = None
        self.filtered = None
        self.last_update = None
        self.last_valid_at = None
        self.last_output = None
        self.last_reason = "reset"

    def _reject(self, reason, now):
        """Hold briefly, then freeze; forget neutral after prolonged loss."""
        if (self.last_valid_at is not None and self.last_output is not None and
                now - self.last_valid_at <= self.sample_timeout):
            self.last_reason = "holding_" + reason
            return self.last_output
        if (self.last_valid_at is not None and
                now - self.last_valid_at > self.reacquire_timeout):
            self.reset()
        self.last_reason = reason
        return None

    def update(self, payload, received_at, now):
        if not isinstance(payload, dict):
            return self._reject("no_payload", now)
        if now - received_at > self.sample_timeout:
            return self._reject("stale_sample", now)
        distance = payload.get("distance_m")
        hand_id = payload.get("hand_id")
        if (payload.get("palm_open") is not True and
                payload.get("label") != "open_palm"):
            return self._reject(
                "label_" + str(payload.get("label", "unknown")), now)
        if payload.get("active") is not True:
            return self._reject(
                str(payload.get("reason", "open_palm_arming")), now)
        if payload.get("distance_valid") is not True:
            return self._reject(
                str(payload.get("depth_reason", "invalid_depth")), now)
        if hand_id is None:
            return self._reject("missing_hand_id", now)
        if (not isinstance(distance, (int, float)) or
                not math.isfinite(float(distance))):
            return self._reject("invalid_distance", now)
        distance = float(distance)
        if (self.last_valid_at is not None and
                now - self.last_valid_at > self.reacquire_timeout):
            self.reset()
        self.last_valid_at = now
        if self.hand_id != hand_id or self.baseline is None:
            self.hand_id = hand_id
            self.baseline = distance
            self.filtered = distance
            self.last_update = now
            self.last_reason = "baseline_captured"
            self.last_output = 0.0
            return self.last_output
        dt = max(0.0, now - self.last_update)
        alpha = 1.0 - math.exp(-dt / self.smoothing_time)
        self.filtered += alpha * (distance - self.filtered)
        self.last_update = now
        displacement = self.filtered - self.baseline
        if abs(displacement) <= self.deadzone:
            self.last_reason = "inside_deadzone"
            self.last_output = 0.0
            return self.last_output
        displacement = math.copysign(abs(displacement) - self.deadzone, displacement)
        self.last_reason = "tracking"
        self.last_output = max(
            -self.max_offset,
            min(self.max_offset, self.gain * displacement))
        return self.last_output
