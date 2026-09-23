#!/usr/bin/env python3
"""
Ghost Game Node - "find the ghost" arm interaction demo.

Flow:
  1. Switch the arm into the impedance chain (zephyr_arm_impedance_controller,
     fed by zephyr_arm_impedance_trajectory_controller), with every joint's
     stiffness set to 0 -> free/compliant ("damping-like") to the touch.
  2. Pick 6 secret joint angles (never published while unfound).
  3. Poll /joint_states; when a joint stays within tolerance of its secret
     target for `match_dwell_time`, that joint is "found": its equilibrium
     reference is pinned there and its stiffness is ramped up so it goes
     rigid (impedance hold), while the rest stay free.
  4. Once all 6 joints are found: open the gripper, activate the impedance
     trajectory controller, and move smoothly to the configured success pose.
  5. At the observation pose, acquire a nearby stable face, scan with bounded
     wrist yaw/pitch motions when none is visible, then continuously servo the
     full-head bbox to image center through direct MIT impedance references.
  6. After capture, orbit-scan the face and perform a smooth tilt-and-nod
     gesture from the captured pose while retaining direct impedance control.
     In parallel, submit the detector's latest expanded full-head crop to the
     FLUX preprocessing service for the later image-to-3D stage.
  7. While the mesh builds, mirror an open palm's aligned depth displacement
     with a bounded camera-axis push/pull motion in direct impedance mode.
  8. Optionally switch to the position-control chain and play a short
     trajectory ("dance").

Stage transitions send cancellable plain-text Actions to ghost_tts. TTS
synthesis and playback remain outside this node so speech cannot block arm
control.

Only ONE controller (mit_impedance_controller) is used for both the free
and locked behavior per joint - "free" is just stiffness=0. No BT is
involved; the whole round is a single background-thread state machine
inside this node, matching the pattern used in
robot_mode_control/robot_state_manager.py.
"""

import json
import math
import random
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import SwitchController
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import JointState
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from vision_msgs.msg import Detection2DArray
from ghost_game_interfaces.action import Speak

from .face_tracking import (
    FaceStabilityGate,
    face_matches_track,
    face_is_acceptable,
    inspection_gesture_target,
    make_face_sample,
    scan_target,
    servo_target,
)
from .mock_trajectory import (
    minimum_quintic_duration,
    quintic_blend,
    trajectory_reference,
)
from .palm_interaction import (
    CameraAxisServo,
    PalmDepthFilter,
    bounded_joint_servo_target,
    palm_center_error_x,
)
from .perception_mode import perception_mode_for_phase, validate_phase_modes

SWITCH_STRICTNESS_BEST_EFFORT = 1


class GhostGameNode(Node):

    def __init__(self):
        super().__init__('ghost_game_node')

        self._declare_parameters()
        self._read_parameters()

        n = len(self.joints)
        if len(self.joint_lower_limits) != n or len(self.joint_upper_limits) != n:
            raise RuntimeError('joint_lower_limits/joint_upper_limits must match joints length')
        if len(self.free_damping) != n or len(self.locked_stiffness) != n or len(self.locked_damping) != n:
            raise RuntimeError('free_damping/locked_stiffness/locked_damping must match joints length')
        if (len(self.mock_solve_stiffness) != n or
                len(self.mock_solve_max_velocity) != n):
            raise RuntimeError(
                'mock_solve_stiffness/mock_solve_max_velocity must match joints length')
        mock_solve_values = (
            self.mock_solve_stiffness + self.mock_solve_max_velocity +
            [self.mock_solve_min_duration, self.mock_solve_settle_timeout])
        if any(not math.isfinite(value) or value <= 0.0
               for value in mock_solve_values):
            raise RuntimeError('mock solve stiffness, velocity, and timing must be positive')
        if len(self.home_positions) != n or len(self.home_stiffness) != n or len(self.home_damping) != n:
            raise RuntimeError('home_positions/home_stiffness/home_damping must match joints length')
        if (len(self.success_positions) != n or
                len(self.success_velocities) != n or
                len(self.success_accelerations) != n or
                len(self.success_position_tolerance) != n or
                len(self.success_stiffness) != n):
            raise RuntimeError(
                'success_positions/success_velocities/'
                'success_accelerations/success_position_tolerance/'
                'success_stiffness '
                'must match joints length')
        if not 0.0 < self.success_seed_time < self.success_time_from_start:
            raise RuntimeError(
                'success timing must satisfy 0 < success_seed_time < success_time_from_start')
        if any(not math.isfinite(value) for value in (
                self.success_positions + self.success_velocities +
                self.success_accelerations + self.success_position_tolerance +
                self.success_stiffness)):
            raise RuntimeError(
                'success pose, velocity, acceleration, tolerance, and '
                'stiffness values '
                'must be finite')
        if any(value <= 0.0 for value in self.success_position_tolerance):
            raise RuntimeError('success_position_tolerance values must be positive')
        if any(value <= 0.0 for value in self.success_stiffness):
            raise RuntimeError('success_stiffness values must be positive')
        if (self.success_validation_timeout <= 0.0 or
                self.success_settle_time <= 0.0 or
                self.success_settle_movement <= 0.0):
            raise RuntimeError(
                'success validation timeout/settle parameters must be positive')
        if any(not self.joint_lower_limits[i] <= self.success_positions[i] <=
               self.joint_upper_limits[i] for i in range(n)):
            raise RuntimeError('success_positions must stay inside the configured joint limits')

        if self.face_yaw_joint not in self.joints:
            raise RuntimeError('face_yaw_joint must name one of joints')
        if self.face_pitch_joint not in self.joints:
            raise RuntimeError('face_pitch_joint must name one of joints')
        if self.face_gesture_roll_joint not in self.joints:
            raise RuntimeError('face_gesture_roll_joint must name one of joints')
        self.face_yaw_index = self.joints.index(self.face_yaw_joint)
        self.face_pitch_index = self.joints.index(self.face_pitch_joint)
        self.face_gesture_roll_index = self.joints.index(
            self.face_gesture_roll_joint)
        if (self.gesture_yaw_max_offset >
                self.gesture_joint_max_offsets[self.face_yaw_index]):
            raise RuntimeError(
                'gesture_yaw_max_offset must fit inside the yaw joint offset')
        if self.face_yaw_index == self.face_pitch_index:
            raise RuntimeError('face yaw and pitch joints must be different')
        if self.face_gesture_roll_index in (
                self.face_yaw_index, self.face_pitch_index):
            raise RuntimeError(
                'face gesture roll joint must differ from face yaw/pitch joints')
        if not (0.0 < self.face_min_bbox_area_ratio <
                self.face_max_bbox_area_ratio <= 1.0):
            raise RuntimeError(
                'face bbox area ratios must satisfy 0 < min < max <= 1')
        positive_face_values = (
            self.face_detection_max_age,
            self.face_stable_time,
            self.face_stability_center_tolerance,
            self.face_stability_area_relative_tolerance,
            self.face_yaw_max_offset,
            self.face_pitch_max_offset,
            self.face_scan_max_joint_speed,
            self.face_scan_min_motion_time,
            self.face_scan_hold_time,
            self.face_center_gain,
            self.face_servo_rate_hz,
            self.face_servo_max_joint_speed,
            self.face_servo_detection_timeout,
            self.face_center_tolerance,
            self.face_center_release_tolerance,
            self.face_center_dwell_time,
            self.face_center_timeout,
            self.face_target_lost_timeout,
            self.face_track_center_jump_tolerance,
            self.face_track_area_relative_jump_tolerance,
            self.face_orbit_pitch_amplitude,
            self.face_orbit_yaw_amplitude,
            self.face_orbit_max_joint_speed,
            self.face_orbit_min_motion_time,
            self.face_gesture_pitch_amplitude,
            self.face_gesture_roll_amplitude,
            self.face_gesture_max_joint_speed,
            self.face_gesture_nod_max_joint_speed,
            self.face_gesture_nod_min_motion_time,
            self.face_gesture_min_motion_time,
            self.face_gesture_trajectory_rate_hz,
        )
        if any(not math.isfinite(value) or value <= 0.0
               for value in positive_face_values):
            raise RuntimeError('face tracking timing/gain/limit values must be positive')
        if self.face_center_release_tolerance < self.face_center_tolerance:
            raise RuntimeError(
                'face_center_release_tolerance must be >= '
                'face_center_tolerance')
        if (not math.isfinite(self.face_follow_duration) or
                self.face_follow_duration < 0.0):
            raise RuntimeError('face_follow_duration must be non-negative')
        if (not math.isfinite(self.face_search_timeout) or
                self.face_search_timeout < 0.0):
            raise RuntimeError('face_search_timeout must be non-negative')
        if (not math.isfinite(self.face_joint_margin) or
                self.face_joint_margin < 0.0):
            raise RuntimeError('face_joint_margin must be non-negative')
        if self.face_gesture_pitch_amplitude > self.face_pitch_max_offset:
            raise RuntimeError(
                'face_gesture_pitch_amplitude must not exceed '
                'face_pitch_max_offset')
        if (self.face_orbit_pitch_amplitude > self.face_pitch_max_offset or
                self.face_orbit_yaw_amplitude > self.face_yaw_max_offset):
            raise RuntimeError(
                'face orbit amplitudes must stay inside face tracking limits')
        if (self.face_orbit_max_joint_speed > 1.5 or
                self.face_gesture_max_joint_speed > 1.5 or
                self.face_gesture_nod_max_joint_speed > 1.5):
            raise RuntimeError(
                'face gesture speeds must not exceed 1.5 rad/s')
        if any(not math.isfinite(value) or value == 0.0 for value in (
                self.face_yaw_direction, self.face_pitch_direction)):
            raise RuntimeError('face yaw/pitch directions must be finite and nonzero')
        if (not self.face_scan_offsets or
                len(self.face_scan_offsets) % 2 != 0 or
                any(not math.isfinite(value)
                    for value in self.face_scan_offsets)):
            raise RuntimeError(
                'face_scan_offsets must contain finite yaw/pitch pairs')
        for offset_index in range(0, len(self.face_scan_offsets), 2):
            yaw_offset = self.face_scan_offsets[offset_index]
            pitch_offset = self.face_scan_offsets[offset_index + 1]
            if (abs(yaw_offset) > self.face_yaw_max_offset or
                    abs(pitch_offset) > self.face_pitch_max_offset):
                raise RuntimeError(
                    'face scan offsets must stay inside local yaw/pitch limits')
        if self.face_scan_hold_time < self.face_stable_time:
            raise RuntimeError(
                'face_scan_hold_time must be >= face_stable_time')

        # Per-joint "distance to secret target" normalizer for ~/state's 0-100%
        # progress bars (see _publish_state): the worst-case separation given
        # fixed_targets are always kept target_margin_ratio away from the
        # joint's hard limits, i.e. confined to a band this wide.
        self._progress_norm = [
            (self.joint_upper_limits[i] - self.joint_lower_limits[i]) * (1.0 - 2.0 * self.target_margin_ratio)
            for i in range(n)
        ]

        # -- Runtime state (guarded by _state_lock) --
        self._state_lock = threading.Lock()
        # idle -> searching -> all_found -> success_move ->
        # success_pose_reached -> face_searching/stabilizing/centering ->
        # face_centered -> face_gesture -> gesture_interaction -> dancing/done
        self._phase = 'idle'
        self._last_perception_command = None
        # Latched when the independently verified success pose is reached.
        # The Web UI consumes this explicit event instead of trying to infer
        # it from short-lived phase names.
        self._camera_ready = False
        self._locked = [False] * n
        self._targets = [0.0] * n       # hold positions used to build JTC goals
        self._secret_targets = [0.0] * n  # immutable per-round ghost targets, for distance debug only
        self._dwell = [0.0] * n
        self._stiffness_state = [0.0] * n
        self._abort_requested = False
        self._go_home_requested = False
        self._mock_solve_requested = False
        self._mock_solve_active = False
        self._mock_palm_transition_pending = False
        self._running = False

        self._joint_positions = {}    # name -> latest position
        self._joint_positions_lock = threading.Lock()
        self._active_goal_lock = threading.Lock()
        self._active_trajectory_goal = None
        self._face_lock = threading.Lock()
        self._camera_size = None
        self._latest_face = None
        self._face_stable_for = 0.0
        self._reconstruction_lock = threading.Lock()
        self._reconstruction_waiting_for_request = False
        self._reconstruction_request_id = ''
        self._reconstruction_state = self._new_reconstruction_state()
        self._reconstruction_state['enabled'] = self.enable_face_reconstruction
        self._palm_lock = threading.Lock()
        self._latest_palm_control = None
        self._latest_palm_received_at = 0.0
        self._mesh_lock = threading.Lock()
        self._mesh_state = {
            'status': 'idle', 'progress': 0, 'request_id': '',
            'received_at': 0.0,
        }
        self._palm_game_state = {
            'active': False,
            'hand_detected': False,
            'distance_m': None,
            'offset_m': 0.0,
            'control_active': False,
            'label': 'unknown',
            'palm_open': False,
            'palm_source': 'none',
            'distance_valid': False,
            'control_reason': 'inactive',
            'depth_reason': '',
            'sample_age_ms': None,
            'depth_age_ms': None,
            'depth_skew_ms': None,
            'inference_ms': None,
            'source_age_ms': None,
            'baseline_m': None,
            'filtered_m': None,
            'tracking_error_rad': 0.0,
            'command_delta_rad': 0.0,
            'yaw_tracking': False,
            'yaw_error': None,
            'yaw_offset_rad': 0.0,
            'mesh_status': 'idle',
            'mesh_progress': 0,
        }

        # -- Callback groups: keep service/action calls off the subscriber/timer group --
        self._io_cb_group = ReentrantCallbackGroup()
        self._call_cb_group = MutuallyExclusiveCallbackGroup()

        # -- Pub/Sub --
        self._joint_state_sub = self.create_subscription(
            JointState, self.joint_states_topic, self._joint_state_cb, 50,
            callback_group=self._io_cb_group)
        self._camera_info_sub = self.create_subscription(
            CameraInfo, self.face_camera_info_topic, self._camera_info_cb,
            qos_profile_sensor_data, callback_group=self._io_cb_group)
        self._face_detection_sub = self.create_subscription(
            Detection2DArray, self.face_detection_topic,
            self._face_detection_cb, 10, callback_group=self._io_cb_group)
        self._palm_control_sub = self.create_subscription(
            String, self.gesture_palm_control_topic,
            self._palm_control_cb, qos_profile_sensor_data,
            callback_group=self._io_cb_group)
        self._gripper_pub = self.create_publisher(JointState, self.gripper_command_topic, 10)
        self._state_pub = self.create_publisher(String, '~/state', 10)
        perception_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._perception_mode_pub = self.create_publisher(
            String, self.perception_mode_topic, perception_qos)
        self._tts_pub = self.create_publisher(String, self.tts_text_topic, 10)
        self._reconstruction_prompt_pub = None
        self._reconstruction_status_sub = None
        if self.enable_face_reconstruction:
            reconstruction_qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self._reconstruction_prompt_pub = self.create_publisher(
                String, self.face_reconstruction_prompt_topic, 10)
            self._reconstruction_status_sub = self.create_subscription(
                String, self.face_reconstruction_status_topic,
                self._face_reconstruction_status_cb, reconstruction_qos,
                callback_group=self._io_cb_group)
        mesh_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._mesh_status_sub = self.create_subscription(
            String, self.gesture_mesh_status_topic,
            self._mesh_status_cb, mesh_qos,
            callback_group=self._io_cb_group)
        self._state_timer = self.create_timer(0.2, self._publish_state, callback_group=self._io_cb_group)

        # -- Clients --
        self._switch_client = self.create_client(
            SwitchController, self.controller_manager_service, callback_group=self._call_cb_group)
        self._impedance_param_client = self.create_client(
            SetParameters, f'/{self.impedance_controller}/set_parameters',
            callback_group=self._call_cb_group)
        self._tts_stop_client = self.create_client(
            Trigger, self.tts_stop_service, callback_group=self._call_cb_group)
        self._tts_action_client = ActionClient(
            self, Speak, self.tts_action_name,
            callback_group=self._call_cb_group)
        # Direct topic command (not the JTC) so the reference always reflects
        # the live measured pose - see _publish_impedance_command for why.
        self._impedance_command_pub = self.create_publisher(
            JointTrajectoryPoint, f'/{self.impedance_controller}/commands', 10)
        # JTC used for the success move and _return_home. The search phase
        # deliberately avoids it to prevent stale-setpoint backswing on lock.
        self._impedance_traj_client = ActionClient(
            self, FollowJointTrajectory,
            f'/{self.impedance_trajectory_controller}/follow_joint_trajectory',
            callback_group=self._call_cb_group)
        self._position_traj_client = ActionClient(
            self, FollowJointTrajectory,
            f'/{self.position_trajectory_controller}/follow_joint_trajectory',
            callback_group=self._call_cb_group)

        # -- Services --
        self.create_service(Trigger, '~/start', self._on_start, callback_group=self._io_cb_group)
        self.create_service(
            Trigger, '~/mock_solve', self._on_mock_solve,
            callback_group=self._io_cb_group)
        self.create_service(
            Trigger, '~/mock_palm_interaction',
            self._on_mock_palm_interaction,
            callback_group=self._io_cb_group)
        self.create_service(Trigger, '~/abort', self._on_abort, callback_group=self._io_cb_group)
        self.create_service(Trigger, '~/return_home', self._on_return_home, callback_group=self._io_cb_group)

        self.get_logger().info(
            f'GhostGameNode ready: {n} joints, tolerance={self.match_tolerance} rad, '
            f'dwell={self.match_dwell_time} s')
        if self.enable_face_reconstruction:
            self.get_logger().info(
                'Face reconstruction enabled: prompt='
                f'{self.face_reconstruction_prompt_topic}, status='
                f'{self.face_reconstruction_status_topic}')

        if self.autostart:
            threading.Thread(target=self._start_round, daemon=True).start()

    # ------------------------------------------------------------------ #
    #  Parameters
    # ------------------------------------------------------------------ #

    def _declare_parameters(self):
        self.declare_parameter('joints', ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])
        self.declare_parameter('joint_lower_limits', [-2.8, 0.0, 0.0, -1.57, -1.57, -3.14])
        self.declare_parameter('joint_upper_limits', [2.8, 3.14, 3.14, 1.57, 1.57, 3.14])
        # NaN sentinel (not an empty list - rclpy can't infer the element type of
        # an empty list) means "no fixed targets, randomize each round".
        self.declare_parameter('fixed_targets', [float('nan')] * 6)
        self.declare_parameter('target_margin_ratio', 0.2)
        self.declare_parameter('match_tolerance', 0.12)
        self.declare_parameter('match_dwell_time', 0.3)
        self.declare_parameter('loop_rate_hz', 30.0)
        # Development auto-solver. It follows a smooth reference inside the
        # normal search loop, so measured-position dwell/locking still proves
        # every hit exactly as it does when a visitor moves the arm by hand.
        self.declare_parameter(
            'mock_solve_stiffness', [10.0, 8.0, 8.0, 6.0, 4.0, 4.0])
        self.declare_parameter(
            'mock_solve_max_velocity', [0.35, 0.25, 0.25, 0.35, 0.35, 0.35])
        self.declare_parameter('mock_solve_min_duration', 4.0)
        self.declare_parameter('mock_solve_settle_timeout', 4.0)
        self.declare_parameter('free_damping', [2.0, 4.0, 3.5, 3.0, 2.5, 1.5])
        self.declare_parameter('locked_stiffness', [30.0, 25.0, 25.0, 18.0, 12.0, 12.0])
        self.declare_parameter('locked_damping', [3.0, 5.0, 4.5, 4.0, 3.0, 2.0])
        self.declare_parameter('stiffness_ramp_steps', 6)
        self.declare_parameter('stiffness_ramp_step_period', 0.05)
        self.declare_parameter('controller_manager_service', '/controller_manager/switch_controller')
        self.declare_parameter('switch_controller_timeout', 5.0)
        self.declare_parameter('impedance_controller', 'zephyr_arm_impedance_controller')
        self.declare_parameter('impedance_trajectory_controller', 'zephyr_arm_impedance_trajectory_controller')
        self.declare_parameter('position_adapter_controller', 'zephyr_arm_position_adapter')
        self.declare_parameter('position_trajectory_controller', 'zephyr_arm_trajectory_controller')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('gripper_command_topic', '/omni_picker/gripper_command')
        # OmniPicker protocol: normalized [0, 1], 0.0 = fully open, 1.0 = fully closed.
        self.declare_parameter('gripper_open_position', 0.0)
        self.declare_parameter('gripper_closed_position', 1.0)
        self.declare_parameter('gripper_settle_time', 1.0)
        self.declare_parameter('tts_enabled', True)
        self.declare_parameter('tts_text_topic', '/ghost/tts/text')
        self.declare_parameter('tts_stop_service', '/ghost/tts/stop')
        self.declare_parameter('tts_action_name', '/ghost/tts/speak')
        self.declare_parameter(
            'tts_setup_text', '壳层自检完成。意识端口正在接入。')
        self.declare_parameter(
            'tts_searching_text',
            '六枚意识碎片已经写入关节。用你的双手，找到它们。')
        self.declare_parameter(
            'tts_joint_found_texts', [
                '第一枚碎片，已固定。',
                '同步率上升。第二枚碎片，已回收。',
                '运动记忆正在重组。第三枚碎片，已上线。',
                '壳层响应稳定。第四枚碎片，已锁定。',
                '意识边界正在闭合。第五枚碎片，已归位。',
                '第六枚碎片，已接入。',
            ])
        self.declare_parameter(
            'tts_all_found_text', '六轴同步完成。Ghost，已经苏醒。')
        self.declare_parameter(
            'tts_success_pose_text',
            '视觉神经接管。请看向我。正在扫描访客信号。')
        self.declare_parameter(
            'tts_face_searching_text', '目标尚未锁定。视觉阵列开始扫描。')
        self.declare_parameter(
            'tts_face_locked_text', '人脸信号稳定。正在校准视觉轴。')
        self.declare_parameter(
            'tts_face_centered_text',
            '视觉轴校准完成。正在潜入深网，生成访客的特工档案。')
        self.declare_parameter(
            'tts_face_not_found_text', '目标信号丢失。视觉扫描已经停止。')
        self.declare_parameter(
            'tts_dancing_text', '运动协议解除。开始执行意识重构序列。')
        self.declare_parameter(
            'tts_done_text', '档案写入完成。身份未知。Ghost 信号，可信。')
        self.declare_parameter(
            'tts_returning_home_text', '本轮连接结束。意识碎片正在回收。')
        self.declare_parameter(
            'tts_idle_text', '壳层进入待机。等待下一次接入。')
        self.declare_parameter(
            'tts_stuck_text',
            '运动链路受阻。安全协议已经释放关节。请清除障碍。')
        self.declare_parameter(
            'tts_aborted_text', '信号中断。当前交互已经终止。')
        self.declare_parameter(
            'success_positions',
            [-1.5137020349502563, 0.596736490726471, 1.2563692331314087,
             -0.2887808680534363, 0.012655341997742653,
             -0.02569492720067501])
        self.declare_parameter(
            'success_velocities', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter(
            'success_accelerations', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('success_seed_time', 0.2)
        self.declare_parameter('success_time_from_start', 8.0)
        # The display/vision pose needs a little more authority than the
        # visitor-facing joint locks, especially on load-bearing joint3.
        # This is ramped only after the game has been solved, so it does not
        # change the interaction feel while visitors search for fragments.
        self.declare_parameter(
            'success_stiffness', [30.0, 30.0, 35.0, 18.0, 18.0, 12.0])
        # The impedance JTC can finish with a small static load-dependent
        # residual and report GOAL_TOLERANCE_VIOLATED even though the arm has
        # safely reached the intended display pose. Validate the measured,
        # settled pose independently instead of treating every action abort
        # as a failed turn. Hardware logs show repeatable loaded residuals of
        # about 0.048 rad on joint2 and one 0.061 rad outlier on joint3; these
        # limits remain tighter than the game's match tolerance.
        self.declare_parameter(
            'success_position_tolerance', [0.05, 0.08, 0.07, 0.04, 0.04, 0.04])
        # The real impedance arm can continue converging for several seconds
        # after the JTC action completes. Keep the tight position tolerances,
        # but allow that physical settling instead of reporting a false abort.
        self.declare_parameter('success_validation_timeout', 6.0)
        self.declare_parameter('success_settle_time', 0.5)
        self.declare_parameter('success_settle_movement', 0.01)
        self.declare_parameter('enable_face_tracking', True)
        self.declare_parameter(
            'perception_mode_topic', '/ghost/perception/mode')
        self.declare_parameter('perception_face_phases', [
            'success_pose_reached', 'face_searching', 'face_stabilizing',
            'face_centering', 'face_following', 'face_centered',
        ])
        self.declare_parameter(
            'perception_gesture_phases', ['gesture_interaction'])
        self.declare_parameter(
            'face_detection_topic', '/nearest_face/detection')
        self.declare_parameter(
            'face_camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('face_detection_max_age', 1.0)
        self.declare_parameter('face_min_bbox_area_ratio', 0.015)
        self.declare_parameter('face_max_bbox_area_ratio', 0.65)
        self.declare_parameter('face_stable_time', 0.30)
        self.declare_parameter('face_stability_center_tolerance', 0.08)
        self.declare_parameter(
            'face_stability_area_relative_tolerance', 0.35)
        self.declare_parameter('face_yaw_joint', 'joint5')
        self.declare_parameter('face_pitch_joint', 'joint4')
        self.declare_parameter('face_yaw_direction', -1.0)
        self.declare_parameter('face_pitch_direction', -1.0)
        self.declare_parameter('face_yaw_max_offset', 0.50)
        self.declare_parameter('face_pitch_max_offset', 0.30)
        self.declare_parameter('face_joint_margin', 0.05)
        self.declare_parameter('face_scan_offsets', [
            0.0, 0.0,
            -0.25, 0.0,
            -0.50, 0.0,
            -0.50, -0.20,
            -0.25, -0.20,
            0.0, -0.20,
            0.25, -0.20,
            0.50, -0.20,
            0.50, 0.20,
            0.25, 0.20,
            0.0, 0.20,
            -0.25, 0.20,
            -0.50, 0.20,
        ])
        self.declare_parameter('face_scan_max_joint_speed', 0.25)
        self.declare_parameter('face_scan_min_motion_time', 0.6)
        self.declare_parameter('face_scan_hold_time', 2.0)
        self.declare_parameter('face_search_timeout', 0.0)
        self.declare_parameter('face_center_gain', 1.80)
        self.declare_parameter('face_servo_rate_hz', 60.0)
        self.declare_parameter('face_servo_max_joint_speed', 0.70)
        self.declare_parameter('face_servo_detection_timeout', 0.20)
        self.declare_parameter('face_center_tolerance', 0.08)
        self.declare_parameter('face_center_release_tolerance', 0.12)
        self.declare_parameter('face_center_dwell_time', 0.40)
        self.declare_parameter('face_follow_duration', 3.0)
        self.declare_parameter('face_center_timeout', 20.0)
        self.declare_parameter('face_target_lost_timeout', 1.5)
        self.declare_parameter('face_track_center_jump_tolerance', 0.25)
        self.declare_parameter(
            'face_track_area_relative_jump_tolerance', 0.75)
        self.declare_parameter('face_gesture_enabled', True)
        self.declare_parameter('face_gesture_roll_joint', 'joint6')
        self.declare_parameter('face_orbit_pitch_amplitude', 0.22)
        self.declare_parameter('face_orbit_yaw_amplitude', 0.36)
        self.declare_parameter('face_orbit_max_joint_speed', 1.20)
        self.declare_parameter('face_orbit_min_motion_time', 0.18)
        self.declare_parameter('face_gesture_pitch_amplitude', 0.18)
        self.declare_parameter('face_gesture_roll_amplitude', 0.30)
        self.declare_parameter('face_gesture_max_joint_speed', 0.80)
        self.declare_parameter('face_gesture_nod_max_joint_speed', 1.25)
        self.declare_parameter('face_gesture_nod_min_motion_time', 0.18)
        self.declare_parameter('face_gesture_min_motion_time', 0.32)
        self.declare_parameter('face_gesture_trajectory_rate_hz', 40.0)
        self.declare_parameter('enable_face_reconstruction', True)
        self.declare_parameter(
            'face_reconstruction_prompt_topic',
            '/ghost/reconstruction/prompt')
        self.declare_parameter(
            'face_reconstruction_status_topic',
            '/ghost/reconstruction/status')
        self.declare_parameter(
            'face_reconstruction_prompt',
            'Create a clean reconstruction-ready cyberpunk operative portrait '
            'from this full-head crop. Preserve identity, facial geometry, '
            'head pose, hair silhouette, ears, and neck. Follow a strict source-only '
            'accessory policy: do not add glasses, sunglasses, goggles, visors, '
            'earrings, ear studs, piercings, jewelry, hats, headsets, masks, facial '
            'hardware, cybernetic implants, tattoos, or any other accessory or body '
            'modification unless that exact item is clearly visible in the source. '
            'If it is absent or uncertain, render none. Never infer accessories from '
            'the cyberpunk theme. Preserve clearly visible existing items exactly; '
            'do not redesign, replace, duplicate, or embellish them. Express the '
            'subtle cyberpunk style only through color grading, lighting, background, '
            'clothing texture, and non-structural surface treatment. Keep the face '
            'unobstructed and use even neutral lighting on a plain dark background.'
        )
        # While FLUX and Tripo build the visitor model, an open palm controls
        # a short camera-axis push/pull game using aligned Orbbec depth.
        self.declare_parameter('enable_gesture_interaction', True)
        self.declare_parameter(
            'gesture_palm_control_topic', '/gestures/palm_control')
        self.declare_parameter(
            'gesture_mesh_status_topic', '/ghost/reconstruction/mesh_status')
        self.declare_parameter('gesture_interaction_min_duration', 8.0)
        self.declare_parameter('gesture_interaction_max_duration', 90.0)
        self.declare_parameter('gesture_finish_on_mesh_ready', True)
        self.declare_parameter('gesture_control_rate_hz', 80.0)
        self.declare_parameter('gesture_depth_deadzone', 0.010)
        self.declare_parameter('gesture_depth_gain', 1.0)
        self.declare_parameter('gesture_max_offset', 0.08)
        self.declare_parameter('gesture_depth_smoothing_time', 0.04)
        self.declare_parameter('gesture_sample_timeout', 0.20)
        self.declare_parameter('gesture_reacquire_timeout', 1.00)
        self.declare_parameter('gesture_position_gain', 6.5)
        self.declare_parameter('gesture_max_linear_speed', 0.22)
        self.declare_parameter('gesture_max_joint_speed', 0.80)
        self.declare_parameter('gesture_tracking_error_limit', 0.18)
        self.declare_parameter('gesture_yaw_enabled', True)
        self.declare_parameter('gesture_yaw_gain', 2.50)
        self.declare_parameter('gesture_yaw_max_speed', 1.00)
        self.declare_parameter('gesture_yaw_max_offset', 0.45)
        self.declare_parameter('gesture_yaw_deadband', 0.08)
        self.declare_parameter(
            'gesture_joint_max_offsets', [0.12, 0.35, 0.35, 0.35, 0.45, 0.18])
        self.declare_parameter(
            'tts_gesture_interaction_text',
            '特工档案正在重构。张开手掌，向我推进或撤离，测试幽灵镜像协议。')
        self.declare_parameter(
            'tts_model_ready_text',
            '神经几何体已经生成。新的壳，正在显现。')
        self.declare_parameter('autostart', False)
        self.declare_parameter('enable_dance', True)
        self.declare_parameter('publish_debug_distances', True)
        self.declare_parameter('debug_reveal_targets', False)
        self.declare_parameter('dance_time_from_start', [1.5, 3.0, 4.5, 6.0])
        self.declare_parameter(
            'dance_positions', [1.5707, 0.0, 0.0, 0.0, 0.0, 0.0] * 4)
        self.declare_parameter(
            'home_positions', [1.5707, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Softer than the compliant baseline on purpose: stiffness_torque_limit
        # ([3.0, 4.0, 4.0, 2.1, 1.4, 1.2], read_only on the controller, can't be
        # lowered from here) is the absolute torque ceiling regardless of
        # stiffness, but a LOWER stiffness means that ceiling is only reached
        # at a much larger position error - for a typical hand-blocking
        # deviation, torque = stiffness * error stays well below the ceiling.
        self.declare_parameter('home_stiffness', [10.0, 8.0, 8.0, 6.0, 4.0, 4.0])
        self.declare_parameter('home_damping', [2.0, 4.0, 3.5, 3.0, 2.5, 1.5])
        self.declare_parameter('home_position_tolerance', 0.16)
        self.declare_parameter('home_time_from_start', 4.0)
        # Seed goal duration before the real return-home move - see _return_home.
        self.declare_parameter('home_seed_time', 0.1)
        # If the JTC doesn't actually reach home_positions (arm stuck/blocked -
        # the trajectory action itself reports success even when physically
        # blocked, since no goal tolerances are configured, so we verify with
        # measured position instead), back off a bit and release stiffness
        # instead of holding at full force or pretending the move succeeded.
        self.declare_parameter('stuck_retreat_fraction', 0.2)
        self.declare_parameter('stuck_retreat_time', 1.0)
        # Poll for arrival/stall during the move instead of waiting the full
        # home_time_from_start before checking anything - see _wait_home_or_stall.
        self.declare_parameter('stuck_check_period', 0.15)
        self.declare_parameter('stuck_stall_window', 0.5)
        self.declare_parameter('stuck_stall_movement', 0.02)
        # A soft home_stiffness makes the arm lag its own commanded
        # trajectory at the start of the move (small real movement even with
        # nothing blocking it) - don't judge stall until this much time has
        # elapsed, or a normal slow move gets flagged "stuck" immediately.
        self.declare_parameter('stuck_grace_period', 1.5)

    def _read_parameters(self):
        p = self.get_parameter
        self.joints = list(p('joints').value)
        self.joint_lower_limits = list(p('joint_lower_limits').value)
        self.joint_upper_limits = list(p('joint_upper_limits').value)
        self.fixed_targets = list(p('fixed_targets').value)
        self.target_margin_ratio = float(p('target_margin_ratio').value)
        self.match_tolerance = float(p('match_tolerance').value)
        self.match_dwell_time = float(p('match_dwell_time').value)
        self.loop_rate_hz = float(p('loop_rate_hz').value)
        self.mock_solve_stiffness = list(p('mock_solve_stiffness').value)
        self.mock_solve_max_velocity = list(
            p('mock_solve_max_velocity').value)
        self.mock_solve_min_duration = float(
            p('mock_solve_min_duration').value)
        self.mock_solve_settle_timeout = float(
            p('mock_solve_settle_timeout').value)
        self.free_damping = list(p('free_damping').value)
        self.locked_stiffness = list(p('locked_stiffness').value)
        self.locked_damping = list(p('locked_damping').value)
        self.stiffness_ramp_steps = int(p('stiffness_ramp_steps').value)
        self.stiffness_ramp_step_period = float(p('stiffness_ramp_step_period').value)
        self.controller_manager_service = p('controller_manager_service').value
        self.switch_controller_timeout = float(p('switch_controller_timeout').value)
        self.impedance_controller = p('impedance_controller').value
        self.impedance_trajectory_controller = p('impedance_trajectory_controller').value
        self.position_adapter_controller = p('position_adapter_controller').value
        self.position_trajectory_controller = p('position_trajectory_controller').value
        self.joint_states_topic = p('joint_states_topic').value
        self.gripper_command_topic = p('gripper_command_topic').value
        self.gripper_open_position = float(p('gripper_open_position').value)
        self.gripper_closed_position = float(p('gripper_closed_position').value)
        self.gripper_settle_time = float(p('gripper_settle_time').value)
        self.tts_enabled = bool(p('tts_enabled').value)
        self.tts_text_topic = p('tts_text_topic').value
        self.tts_stop_service = p('tts_stop_service').value
        self.tts_action_name = p('tts_action_name').value
        self.tts_setup_text = p('tts_setup_text').value
        self.tts_searching_text = p('tts_searching_text').value
        self.tts_joint_found_texts = list(p('tts_joint_found_texts').value)
        self.tts_all_found_text = p('tts_all_found_text').value
        self.tts_success_pose_text = p('tts_success_pose_text').value
        self.tts_face_searching_text = p('tts_face_searching_text').value
        self.tts_face_locked_text = p('tts_face_locked_text').value
        self.tts_face_centered_text = p('tts_face_centered_text').value
        self.tts_face_not_found_text = p('tts_face_not_found_text').value
        self.tts_dancing_text = p('tts_dancing_text').value
        self.tts_done_text = p('tts_done_text').value
        self.tts_returning_home_text = p('tts_returning_home_text').value
        self.tts_idle_text = p('tts_idle_text').value
        self.tts_stuck_text = p('tts_stuck_text').value
        self.tts_aborted_text = p('tts_aborted_text').value
        if len(self.tts_joint_found_texts) != len(self.joints):
            raise RuntimeError('tts_joint_found_texts must match joints length')
        self.success_positions = list(p('success_positions').value)
        self.success_velocities = list(p('success_velocities').value)
        self.success_accelerations = list(p('success_accelerations').value)
        self.success_seed_time = float(p('success_seed_time').value)
        self.success_time_from_start = float(p('success_time_from_start').value)
        self.success_stiffness = list(p('success_stiffness').value)
        self.success_position_tolerance = list(
            p('success_position_tolerance').value)
        self.success_validation_timeout = float(
            p('success_validation_timeout').value)
        self.success_settle_time = float(p('success_settle_time').value)
        self.success_settle_movement = float(
            p('success_settle_movement').value)
        self.enable_face_tracking = bool(p('enable_face_tracking').value)
        self.perception_mode_topic = str(p('perception_mode_topic').value)
        self.perception_face_phases, self.perception_gesture_phases = (
            validate_phase_modes(
                list(p('perception_face_phases').value),
                list(p('perception_gesture_phases').value)))
        if not self.perception_mode_topic.strip():
            raise RuntimeError('perception_mode_topic must be non-empty')
        self.face_detection_topic = p('face_detection_topic').value
        self.face_camera_info_topic = p('face_camera_info_topic').value
        self.face_detection_max_age = float(
            p('face_detection_max_age').value)
        self.face_min_bbox_area_ratio = float(
            p('face_min_bbox_area_ratio').value)
        self.face_max_bbox_area_ratio = float(
            p('face_max_bbox_area_ratio').value)
        self.face_stable_time = float(p('face_stable_time').value)
        self.face_stability_center_tolerance = float(
            p('face_stability_center_tolerance').value)
        self.face_stability_area_relative_tolerance = float(
            p('face_stability_area_relative_tolerance').value)
        self.face_yaw_joint = p('face_yaw_joint').value
        self.face_pitch_joint = p('face_pitch_joint').value
        self.face_yaw_direction = float(p('face_yaw_direction').value)
        self.face_pitch_direction = float(p('face_pitch_direction').value)
        self.face_yaw_max_offset = float(p('face_yaw_max_offset').value)
        self.face_pitch_max_offset = float(p('face_pitch_max_offset').value)
        self.face_joint_margin = float(p('face_joint_margin').value)
        self.face_scan_offsets = list(p('face_scan_offsets').value)
        self.face_scan_max_joint_speed = float(
            p('face_scan_max_joint_speed').value)
        self.face_scan_min_motion_time = float(
            p('face_scan_min_motion_time').value)
        self.face_scan_hold_time = float(p('face_scan_hold_time').value)
        self.face_search_timeout = float(p('face_search_timeout').value)
        self.face_center_gain = float(p('face_center_gain').value)
        self.face_servo_rate_hz = float(p('face_servo_rate_hz').value)
        self.face_servo_max_joint_speed = float(
            p('face_servo_max_joint_speed').value)
        self.face_servo_detection_timeout = float(
            p('face_servo_detection_timeout').value)
        self.face_center_tolerance = float(
            p('face_center_tolerance').value)
        self.face_center_release_tolerance = float(
            p('face_center_release_tolerance').value)
        self.face_center_dwell_time = float(
            p('face_center_dwell_time').value)
        self.face_follow_duration = float(p('face_follow_duration').value)
        self.face_center_timeout = float(p('face_center_timeout').value)
        self.face_target_lost_timeout = float(
            p('face_target_lost_timeout').value)
        self.face_track_center_jump_tolerance = float(
            p('face_track_center_jump_tolerance').value)
        self.face_track_area_relative_jump_tolerance = float(
            p('face_track_area_relative_jump_tolerance').value)
        self.face_gesture_enabled = bool(p('face_gesture_enabled').value)
        self.face_gesture_roll_joint = p('face_gesture_roll_joint').value
        self.face_orbit_pitch_amplitude = float(
            p('face_orbit_pitch_amplitude').value)
        self.face_orbit_yaw_amplitude = float(
            p('face_orbit_yaw_amplitude').value)
        self.face_orbit_max_joint_speed = float(
            p('face_orbit_max_joint_speed').value)
        self.face_orbit_min_motion_time = float(
            p('face_orbit_min_motion_time').value)
        self.face_gesture_pitch_amplitude = float(
            p('face_gesture_pitch_amplitude').value)
        self.face_gesture_roll_amplitude = float(
            p('face_gesture_roll_amplitude').value)
        self.face_gesture_max_joint_speed = float(
            p('face_gesture_max_joint_speed').value)
        self.face_gesture_nod_max_joint_speed = float(
            p('face_gesture_nod_max_joint_speed').value)
        self.face_gesture_nod_min_motion_time = float(
            p('face_gesture_nod_min_motion_time').value)
        self.face_gesture_min_motion_time = float(
            p('face_gesture_min_motion_time').value)
        self.face_gesture_trajectory_rate_hz = float(
            p('face_gesture_trajectory_rate_hz').value)
        self.enable_face_reconstruction = bool(
            p('enable_face_reconstruction').value)
        self.face_reconstruction_prompt_topic = str(
            p('face_reconstruction_prompt_topic').value)
        self.face_reconstruction_status_topic = str(
            p('face_reconstruction_status_topic').value)
        self.face_reconstruction_prompt = str(
            p('face_reconstruction_prompt').value)
        if self.enable_face_reconstruction and not all((
                self.face_reconstruction_prompt_topic.strip(),
                self.face_reconstruction_status_topic.strip(),
                self.face_reconstruction_prompt.strip())):
            raise RuntimeError(
                'face reconstruction topics and prompt must be non-empty')
        self.enable_gesture_interaction = bool(
            p('enable_gesture_interaction').value)
        self.gesture_palm_control_topic = str(
            p('gesture_palm_control_topic').value)
        self.gesture_mesh_status_topic = str(
            p('gesture_mesh_status_topic').value)
        self.gesture_interaction_min_duration = float(
            p('gesture_interaction_min_duration').value)
        self.gesture_interaction_max_duration = float(
            p('gesture_interaction_max_duration').value)
        self.gesture_finish_on_mesh_ready = bool(
            p('gesture_finish_on_mesh_ready').value)
        self.gesture_control_rate_hz = float(
            p('gesture_control_rate_hz').value)
        self.gesture_depth_deadzone = float(
            p('gesture_depth_deadzone').value)
        self.gesture_depth_gain = float(p('gesture_depth_gain').value)
        self.gesture_max_offset = float(p('gesture_max_offset').value)
        self.gesture_depth_smoothing_time = float(
            p('gesture_depth_smoothing_time').value)
        self.gesture_sample_timeout = float(
            p('gesture_sample_timeout').value)
        self.gesture_reacquire_timeout = float(
            p('gesture_reacquire_timeout').value)
        self.gesture_position_gain = float(
            p('gesture_position_gain').value)
        self.gesture_max_linear_speed = float(
            p('gesture_max_linear_speed').value)
        self.gesture_max_joint_speed = float(
            p('gesture_max_joint_speed').value)
        self.gesture_tracking_error_limit = float(
            p('gesture_tracking_error_limit').value)
        self.gesture_yaw_enabled = bool(p('gesture_yaw_enabled').value)
        self.gesture_yaw_gain = float(p('gesture_yaw_gain').value)
        self.gesture_yaw_max_speed = float(
            p('gesture_yaw_max_speed').value)
        self.gesture_yaw_max_offset = float(
            p('gesture_yaw_max_offset').value)
        self.gesture_yaw_deadband = float(p('gesture_yaw_deadband').value)
        self.gesture_joint_max_offsets = list(
            p('gesture_joint_max_offsets').value)
        self.tts_gesture_interaction_text = str(
            p('tts_gesture_interaction_text').value)
        self.tts_model_ready_text = str(p('tts_model_ready_text').value)
        gesture_positive = (
            self.gesture_interaction_min_duration,
            self.gesture_interaction_max_duration,
            self.gesture_control_rate_hz,
            self.gesture_depth_deadzone,
            self.gesture_depth_gain,
            self.gesture_max_offset,
            self.gesture_depth_smoothing_time,
            self.gesture_sample_timeout,
            self.gesture_reacquire_timeout,
            self.gesture_position_gain,
            self.gesture_max_linear_speed,
            self.gesture_max_joint_speed,
            self.gesture_tracking_error_limit,
            self.gesture_yaw_gain,
            self.gesture_yaw_max_speed,
            self.gesture_yaw_max_offset,
        )
        if any(not math.isfinite(value) or value <= 0.0
               for value in gesture_positive):
            raise RuntimeError(
                'gesture interaction timing, gain, and limits must be positive')
        if (self.gesture_interaction_min_duration >
                self.gesture_interaction_max_duration):
            raise RuntimeError(
                'gesture interaction minimum duration must not exceed maximum')
        if self.gesture_reacquire_timeout < self.gesture_sample_timeout:
            raise RuntimeError(
                'gesture_reacquire_timeout must be at least '
                'gesture_sample_timeout')
        if (not math.isfinite(self.gesture_yaw_deadband) or
                not 0.0 <= self.gesture_yaw_deadband < 1.0):
            raise RuntimeError(
                'gesture_yaw_deadband must be finite and in [0, 1)')
        if (len(self.gesture_joint_max_offsets) != len(self.joints) or
                any(not math.isfinite(value) or value <= 0.0
                    for value in self.gesture_joint_max_offsets)):
            raise RuntimeError(
                'gesture_joint_max_offsets must contain one positive value per joint')
        if self.enable_gesture_interaction and not all((
                self.gesture_palm_control_topic.strip(),
                self.gesture_mesh_status_topic.strip(),
                self.tts_gesture_interaction_text.strip())):
            raise RuntimeError(
                'gesture interaction topics and TTS cue must be non-empty')
        self.autostart = bool(p('autostart').value)
        self.enable_dance = bool(p('enable_dance').value)
        self.publish_debug_distances = bool(p('publish_debug_distances').value)
        self.debug_reveal_targets = bool(p('debug_reveal_targets').value)
        self.dance_time_from_start = list(p('dance_time_from_start').value)
        self.dance_positions = list(p('dance_positions').value)
        self.home_positions = list(p('home_positions').value)
        self.home_stiffness = list(p('home_stiffness').value)
        self.home_damping = list(p('home_damping').value)
        self.home_position_tolerance = float(
            p('home_position_tolerance').value)
        self.home_time_from_start = float(p('home_time_from_start').value)
        self.home_seed_time = float(p('home_seed_time').value)
        self.stuck_retreat_fraction = float(p('stuck_retreat_fraction').value)
        self.stuck_retreat_time = float(p('stuck_retreat_time').value)
        self.stuck_check_period = float(p('stuck_check_period').value)
        self.stuck_stall_window = float(p('stuck_stall_window').value)
        self.stuck_stall_movement = float(p('stuck_stall_movement').value)
        self.stuck_grace_period = float(p('stuck_grace_period').value)

    # ------------------------------------------------------------------ #
    #  Subscriptions / state publishing
    # ------------------------------------------------------------------ #

    def _joint_state_cb(self, msg: JointState):
        with self._joint_positions_lock:
            for name, pos in zip(msg.name, msg.position):
                self._joint_positions[name] = pos

    def _camera_info_cb(self, msg: CameraInfo):
        if msg.width <= 0 or msg.height <= 0:
            return
        with self._face_lock:
            self._camera_size = (int(msg.width), int(msg.height))

    def _face_detection_cb(self, msg: Detection2DArray):
        now = time.monotonic()
        with self._face_lock:
            camera_size = self._camera_size
        if camera_size is None or not msg.detections:
            with self._face_lock:
                self._latest_face = None
            return

        detection = max(
            msg.detections,
            key=lambda item: float(item.bbox.size_x * item.bbox.size_y),
        )
        score = (
            float(detection.results[0].hypothesis.score)
            if detection.results else 0.0
        )
        try:
            sample = make_face_sample(
                received_at=now,
                center_x=detection.bbox.center.position.x,
                center_y=detection.bbox.center.position.y,
                width=detection.bbox.size_x,
                height=detection.bbox.size_y,
                image_width=camera_size[0],
                image_height=camera_size[1],
                score=score,
            )
        except ValueError:
            sample = None
        with self._face_lock:
            self._latest_face = sample

    def _palm_control_cb(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            self.get_logger().warn(
                'Ignoring malformed palm-control JSON',
                throttle_duration_sec=2.0)
            return
        if not isinstance(payload, dict):
            return
        with self._palm_lock:
            self._latest_palm_control = payload
            self._latest_palm_received_at = time.monotonic()

    def _mesh_status_cb(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            self.get_logger().warn(
                'Ignoring malformed mesh-status JSON',
                throttle_duration_sec=2.0)
            return
        if not isinstance(payload, dict) or not payload.get('status'):
            return
        try:
            progress = max(0, min(100, int(payload.get('progress', 0) or 0)))
        except (TypeError, ValueError):
            return
        with self._mesh_lock:
            self._mesh_state = {
                'status': str(payload['status']),
                'progress': progress,
                'request_id': str(payload.get('request_id', '')),
                'received_at': time.monotonic(),
            }

    @staticmethod
    def _new_reconstruction_state():
        return {
            'enabled': False,
            'status': 'idle',
            'request_id': '',
            'input_path': '',
            'latest_input_path': '',
            'output_path': '',
            'width': 0,
            'height': 0,
            'total_sec': 0.0,
            'error': '',
        }

    def _reset_reconstruction_state(self):
        with self._reconstruction_lock:
            self._reconstruction_waiting_for_request = False
            self._reconstruction_request_id = ''
            self._reconstruction_state = self._new_reconstruction_state()
            self._reconstruction_state['enabled'] = (
                self.enable_face_reconstruction)

    def _face_reconstruction_status_cb(self, message):
        """Mirror only the current round's asynchronous FLUX request."""
        try:
            update = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            self.get_logger().warn(
                'Ignoring malformed face reconstruction status JSON',
                throttle_duration_sec=2.0)
            return
        if not isinstance(update, dict):
            return
        status = str(update.get('status', ''))
        request_id = str(update.get('request_id', ''))
        if not status or not request_id:
            return
        try:
            width = int(update.get('width', 0) or 0)
            height = int(update.get('height', 0) or 0)
            total_sec = float(update.get('total_sec', 0.0) or 0.0)
        except (TypeError, ValueError):
            return
        with self._reconstruction_lock:
            if self._reconstruction_waiting_for_request:
                if status != 'accepted':
                    return
                self._reconstruction_waiting_for_request = False
                self._reconstruction_request_id = request_id
            elif (not self._reconstruction_request_id or
                  request_id != self._reconstruction_request_id):
                return
            self._reconstruction_state = {
                'enabled': True,
                'status': status,
                'request_id': request_id,
                'input_path': str(update.get('input_path', '')),
                'latest_input_path': str(
                    update.get('latest_input_path', '')),
                'output_path': str(update.get('output_path', '')),
                'width': width,
                'height': height,
                'total_sec': total_sec,
                'error': str(update.get('error', '')),
            }

    def _submit_face_reconstruction(self):
        """Trigger FLUX once; the worker and any later 3D stage stay async."""
        if self._reconstruction_prompt_pub is None:
            return
        with self._reconstruction_lock:
            self._reconstruction_waiting_for_request = True
            self._reconstruction_request_id = ''
            self._reconstruction_state = self._new_reconstruction_state()
            self._reconstruction_state.update({
                'enabled': True,
                'status': 'submitted',
            })
        with self._mesh_lock:
            self._mesh_state = {
                'status': 'idle', 'progress': 0, 'request_id': '',
                'received_at': time.monotonic(),
            }
        self._reconstruction_prompt_pub.publish(String(
            data=self.face_reconstruction_prompt.strip()))
        self.get_logger().info(
            'Submitted latest stable full-head crop for FLUX preprocessing')

    def _positions_snapshot(self):
        with self._joint_positions_lock:
            return [self._joint_positions.get(j) for j in self.joints]

    def _face_sample_snapshot(self):
        with self._face_lock:
            return self._latest_face

    def _clear_face_sample(self):
        with self._face_lock:
            self._latest_face = None
            self._face_stable_for = 0.0

    def _speak(self, text):
        """Queue one stage cue without ever blocking arm control."""
        if not self.tts_enabled or not isinstance(text, str) or not text.strip():
            return
        text = text.strip()
        if not self._send_speech_action(text,interrupt=False):
            self._tts_pub.publish(String(data=text))

    def _send_speech_action(self,text,interrupt):
        """Submit speech asynchronously; return False for legacy fallback."""
        client = getattr(self,'_tts_action_client',None)
        if client is None or not client.server_is_ready():
            return False
        goal = Speak.Goal()
        goal.text = text
        goal.interrupt = bool(interrupt)
        client.send_goal_async(goal)
        return True

    def _interrupt_speech(self, text):
        """Best-effort cancellation for safety/return-home announcements."""
        if not self.tts_enabled:
            return
        text = text.strip()
        if self._send_speech_action(text,interrupt=True):
            return
        if self._tts_stop_client.service_is_ready():
            future = self._tts_stop_client.call_async(Trigger.Request())
            threading.Thread(
                target=self._speak_after_stop,
                args=(future, text),
                daemon=True,
            ).start()
            return
        self._speak(text)

    def _speak_after_stop(self, future, text):
        """Wait off the arm-control path, then queue the replacement cue."""
        self._wait_for_future(future, 0.75)
        self._speak(text)

    def _publish_state(self):
        with self._state_lock:
            phase = self._phase
            payload = {
                'phase': phase,
                'camera_ready': self._camera_ready,
                'mock_solve_active': self._mock_solve_active,
                'mock_palm_transition_pending':
                    self._mock_palm_transition_pending,
                'locked': list(self._locked),
                'found_count': sum(self._locked),
                'total': len(self.joints),
            }
            secret_targets = list(self._secret_targets)
            locked = list(self._locked)
        perception_mode = perception_mode_for_phase(
            phase, self.perception_face_phases,
            self.perception_gesture_phases)
        payload['perception_mode'] = perception_mode
        perception_command = (phase, perception_mode)
        if perception_command != self._last_perception_command:
            command = String()
            command.data = json.dumps({
                'mode': perception_mode,
                'phase': phase,
                'source': 'ghost_game_node',
            }, separators=(',', ':'))
            self._perception_mode_pub.publish(command)
            self._last_perception_command = perception_command
        if self.publish_debug_distances:
            positions = self._positions_snapshot()
            payload['joints'] = self.joints
            payload['positions'] = positions
            distances = [
                None if pos is None else abs(pos - secret_targets[i])
                for i, pos in enumerate(positions)
            ]
            payload['distances'] = distances
            # 0-100% "getting warmer" bar: 100% at/inside match_tolerance,
            # 0% at the worst-case separation for that joint (_progress_norm),
            # linear in between. Locked joints are always pinned to 100%.
            payload['progress'] = [
                100.0 if locked[i] else (
                    None if dist is None else
                    max(0.0, min(1.0, 1.0 - (dist - self.match_tolerance) /
                                max(self._progress_norm[i] - self.match_tolerance, 1e-6))) * 100.0
                )
                for i, dist in enumerate(distances)
            ]
            if self.debug_reveal_targets:
                payload['targets'] = secret_targets
        if self.enable_face_tracking:
            now = time.monotonic()
            face = self._face_sample_snapshot()
            with self._face_lock:
                stable_for = self._face_stable_for
            payload['face'] = {
                'detected': face is not None,
                'accepted': face_is_acceptable(
                    face, now, self.face_detection_max_age,
                    self.face_min_bbox_area_ratio,
                    self.face_max_bbox_area_ratio),
                'area_ratio': None if face is None else face.area_ratio,
                'error_x': None if face is None else face.error_x,
                'error_y': None if face is None else face.error_y,
                'score': None if face is None else face.score,
                'stable_for': stable_for,
            }
        with self._reconstruction_lock:
            payload['reconstruction'] = dict(self._reconstruction_state)
        with self._palm_lock:
            payload['palm_interaction'] = dict(self._palm_game_state)
        msg = String()
        msg.data = json.dumps(payload)
        self._state_pub.publish(msg)

    # ------------------------------------------------------------------ #
    #  Services
    # ------------------------------------------------------------------ #

    def _on_start(self, request, response):
        with self._state_lock:
            if self._running:
                response.success = False
                response.message = f'Round already running (phase={self._phase})'
                return response
            self._running = True
            self._abort_requested = False
            self._mock_solve_requested = False
            self._mock_solve_active = False
            self._mock_palm_transition_pending = False
        threading.Thread(target=self._start_round, daemon=True).start()
        response.success = True
        response.message = 'Round started'
        return response

    def _on_mock_solve(self, request, response):
        del request
        with self._state_lock:
            if not self._running or self._phase != 'searching':
                response.success = False
                response.message = (
                    'Mock solve requires an active round in searching phase '
                    f'(phase={self._phase})')
                return response
            if self._mock_solve_requested or self._mock_solve_active:
                response.success = False
                response.message = 'Mock solve is already active'
                return response
            self._mock_solve_requested = True
        response.success = True
        response.message = 'Mock trajectory queued for the secret pose'
        return response

    def _on_mock_palm_interaction(self, request, response):
        """Preempt any game stage, move to success pose, then enter palm play."""
        del request
        with self._state_lock:
            if self._mock_palm_transition_pending:
                response.success = True
                response.message = 'Direct palm interaction is already queued'
                return response
            if self._running:
                previous_phase = self._phase
                self._mock_palm_transition_pending = True
                self._abort_requested = True
                self._go_home_requested = False
                self._camera_ready = False
                queue_after_preemption = True
            else:
                self._running = True
                self._abort_requested = False
                self._go_home_requested = False
                self._mock_solve_requested = False
                self._mock_solve_active = False
                self._mock_palm_transition_pending = False
                self._camera_ready = False
                queue_after_preemption = False
        if queue_after_preemption:
            # Cancel a success/inspection/home JTC immediately. Direct loops
            # observe _abort_requested on their next tick.
            self._cancel_active_trajectory()
            threading.Thread(
                target=self._start_mock_palm_after_preemption,
                daemon=True,
            ).start()
            response.success = True
            response.message = (
                'Current stage is stopping; success-pose palm interaction queued '
                f'(phase={previous_phase})')
            return response
        threading.Thread(
            target=self._start_mock_palm_interaction, daemon=True).start()
        response.success = True
        response.message = (
            'Moving to success pose before direct palm interaction')
        return response

    def _on_abort(self, request, response):
        with self._state_lock:
            self._abort_requested = True
            self._mock_palm_transition_pending = False
        response.success = True
        response.message = 'Abort requested'
        return response

    def _on_return_home(self, request, response):
        with self._state_lock:
            self._mock_palm_transition_pending = False
            if not self._running:
                self._running = True
                start_new_thread = True
            elif self._phase == 'returning_home':
                response.success = False
                response.message = 'Already returning home'
                return response
            else:
                # A round is mid-flight: ask its background thread to bail
                # out of the search loop and go home itself, instead of
                # racing a second thread against it on the same arm.
                self._go_home_requested = True
                start_new_thread = False
                phase = self._phase
        if start_new_thread:
            threading.Thread(target=self._start_return_home, daemon=True).start()
            response.success = True
            response.message = 'Returning home'
        else:
            response.success = True
            response.message = f'Go-home requested, will take effect shortly (phase={phase})'
        return response

    def _start_return_home(self):
        try:
            self._return_home()
        except Exception:
            self.get_logger().error('return_home crashed', exc_info=True)
            self._safe_abort()
        finally:
            with self._state_lock:
                self._running = False

    def _start_mock_palm_interaction(self):
        """Move to the show pose, then run only the gesture stage."""
        try:
            if not self._set_impedance_params(
                    **{'gravity_compensation.factor': 1.0}):
                self._safe_abort()
                return
            if not self._move_to_success_pose():
                request = self._face_control_request()
                if request == 'home':
                    self._return_home()
                else:
                    self._safe_abort()
                return
            request = self._face_control_request()
            if request == 'abort':
                self._safe_abort()
                return
            if request == 'home':
                self._return_home()
                return
            with self._state_lock:
                self._camera_ready = True
            result = self._run_palm_interaction(finish_on_mesh_ready=False)
            if result == 'abort':
                self._safe_abort()
            elif result == 'home':
                self._return_home()
            elif result == 'failed':
                self._safe_abort()
            else:
                with self._state_lock:
                    self._phase = 'done'
                    self._camera_ready = False
                self.get_logger().info(
                    'Direct palm-interaction test complete')
        except Exception:
            self.get_logger().error(
                'direct palm interaction crashed', exc_info=True)
            self._safe_abort()
        finally:
            with self._state_lock:
                self._running = False

    def _start_mock_palm_after_preemption(self):
        """Wait for the current game thread to release control, then start."""
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            with self._state_lock:
                if not self._mock_palm_transition_pending:
                    return
                if not self._running:
                    self._running = True
                    self._abort_requested = False
                    self._go_home_requested = False
                    self._mock_solve_requested = False
                    self._mock_solve_active = False
                    self._mock_palm_transition_pending = False
                    self._camera_ready = False
                    break
            time.sleep(0.02)
        else:
            with self._state_lock:
                self._mock_palm_transition_pending = False
            self.get_logger().error(
                'Timed out preempting the current stage for palm interaction')
            return
        self._start_mock_palm_interaction()

    # ------------------------------------------------------------------ #
    #  Blocking helpers (run only from the background game thread)
    # ------------------------------------------------------------------ #

    def _wait_for_future(self, future, timeout_sec):
        deadline = time.monotonic() + timeout_sec
        while not future.done():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True

    def _switch_controllers(self, activate, deactivate):
        if not self._switch_client.wait_for_service(timeout_sec=self.switch_controller_timeout):
            self.get_logger().error('switch_controller service unavailable')
            return False
        req = SwitchController.Request()
        req.activate_controllers = activate
        req.deactivate_controllers = deactivate
        req.strictness = SWITCH_STRICTNESS_BEST_EFFORT
        req.activate_asap = True
        req.timeout = Duration(sec=int(self.switch_controller_timeout))
        future = self._switch_client.call_async(req)
        if not self._wait_for_future(future, self.switch_controller_timeout + 1.0):
            self.get_logger().error('switch_controller call timed out')
            return False
        result = future.result()
        if result is None or not result.ok:
            self.get_logger().error(f'switch_controller failed: activate={activate} deactivate={deactivate}')
            return False
        return True

    @staticmethod
    def _impedance_param(name, value):
        param = Parameter()
        param.name = name
        if isinstance(value, (list, tuple)):
            param.value = ParameterValue(
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                double_array_value=[float(v) for v in value])
        else:
            param.value = ParameterValue(
                type=ParameterType.PARAMETER_DOUBLE, double_value=float(value))
        return param

    def _set_impedance_params(self, **params):
        if not self._impedance_param_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('impedance controller set_parameters service unavailable')
            return False
        req = SetParameters.Request()
        req.parameters = [self._impedance_param(name, value) for name, value in params.items()]
        future = self._impedance_param_client.call_async(req)
        if not self._wait_for_future(future, 2.0):
            self.get_logger().error('set_parameters call timed out')
            return False
        result = future.result()
        if result is None or not all(r.successful for r in result.results):
            self.get_logger().error(f'set_parameters rejected: {params.keys()}')
            return False
        return True

    def _send_trajectory(
            self, client: ActionClient, positions, time_from_start_list,
            wait_result=False, velocities=None, accelerations=None):
        if not client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('trajectory action server unavailable')
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = list(self.joints)
        if isinstance(positions[0], (list, tuple)):
            points_positions = positions
        else:
            points_positions = [positions]
        point_count = len(points_positions)
        if len(time_from_start_list) != point_count:
            self.get_logger().error('trajectory positions/time_from_start length mismatch')
            return False

        def normalize_optional(values, field_name):
            if values is None:
                return [None] * point_count
            points = values if isinstance(values[0], (list, tuple)) else [values]
            if len(points) != point_count:
                self.get_logger().error(
                    f'trajectory {field_name}/positions length mismatch')
                return None
            return points

        points_velocities = normalize_optional(velocities, 'velocities')
        points_accelerations = normalize_optional(accelerations, 'accelerations')
        if points_velocities is None or points_accelerations is None:
            return False

        for index, (pts, t) in enumerate(
                zip(points_positions, time_from_start_list)):
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in pts]
            if points_velocities[index] is not None:
                point.velocities = [float(v) for v in points_velocities[index]]
            if points_accelerations[index] is not None:
                point.accelerations = [float(v) for v in points_accelerations[index]]
            point.time_from_start = Duration(sec=int(t), nanosec=int((t % 1.0) * 1e9))
            goal.trajectory.points.append(point)

        send_future = client.send_goal_async(goal)
        if not self._wait_for_future(send_future, 3.0):
            self.get_logger().error('trajectory goal send timed out')
            return False
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('trajectory goal rejected')
            return False
        with self._active_goal_lock:
            self._active_trajectory_goal = goal_handle
        if wait_result:
            result_future = goal_handle.get_result_async()
            max_t = max(time_from_start_list) if time_from_start_list else 5.0
            if not self._wait_for_future(result_future, max_t + 5.0):
                self._clear_active_trajectory_goal(goal_handle)
                self.get_logger().error('trajectory result timed out')
                return False
            wrapped_result = result_future.result()
            self._clear_active_trajectory_goal(goal_handle)
            if (wrapped_result is None or
                    wrapped_result.result.error_code !=
                    FollowJointTrajectory.Result.SUCCESSFUL):
                error_code = (
                    None if wrapped_result is None
                    else wrapped_result.result.error_code)
                error_string = (
                    '' if wrapped_result is None
                    else wrapped_result.result.error_string)
                self.get_logger().error(
                    f'trajectory failed: error_code={error_code}, error={error_string}')
                return False
        else:
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(
                lambda future: self._clear_active_trajectory_goal(goal_handle))
        return True

    def _clear_active_trajectory_goal(self, goal_handle):
        with self._active_goal_lock:
            if self._active_trajectory_goal is goal_handle:
                self._active_trajectory_goal = None

    def _cancel_active_trajectory(self):
        with self._active_goal_lock:
            goal_handle = self._active_trajectory_goal
            self._active_trajectory_goal = None
        if goal_handle is not None:
            goal_handle.cancel_goal_async()

    def _move_to_success_pose(self):
        """Move smoothly from the live impedance reference to the success pose.

        The impedance JTC receives explicit velocity and acceleration
        endpoints so its interpolator produces a smooth start and stop.
        """
        n = len(self.joints)
        with self._state_lock:
            self._phase = 'success_move'

        self.get_logger().info(
            f'Moving to success pose via impedance JTC over '
            f'{self.success_time_from_start:.1f} s')

        # The game locks deliberately remain compliant. Once all fragments
        # are found, ramp to a separate success-pose stiffness before moving.
        # A ramp avoids a torque step and gives joint3 enough authority to
        # overcome its observed load-dependent static residual.
        self._ramp_stiffness(self.success_stiffness, self.locked_damping)
        if not self._switch_controllers(
                activate=[self.impedance_controller,
                          self.impedance_trajectory_controller],
                deactivate=[self.position_adapter_controller,
                            self.position_trajectory_controller]):
            return False

        deadline = time.monotonic() + 3.0
        start_positions = self._positions_snapshot()
        while (any(position is None for position in start_positions) and
               time.monotonic() < deadline):
            time.sleep(0.05)
            start_positions = self._positions_snapshot()
        if any(position is None for position in start_positions):
            self.get_logger().error(
                'Never received full /joint_states, aborting success move')
            return False

        zero_motion = [0.0] * n
        if not self._send_trajectory(
                self._impedance_traj_client, start_positions,
                [self.success_seed_time], wait_result=True,
                velocities=zero_motion, accelerations=zero_motion):
            self.get_logger().error('Failed to seed impedance JTC at live pose')
            return False

        start_positions = self._positions_snapshot()
        if any(position is None for position in start_positions):
            self.get_logger().error('Lost /joint_states before success move')
            return False
        action_succeeded = self._send_trajectory(
            self._impedance_traj_client,
            [start_positions, self.success_positions],
            [self.success_seed_time, self.success_time_from_start],
            wait_result=True,
            velocities=[zero_motion, self.success_velocities],
            accelerations=[zero_motion, self.success_accelerations])
        if not action_succeeded:
            self.get_logger().warn(
                'Success-pose trajectory action did not report success; '
                'checking the measured settled pose before aborting')

        # Do not reveal the camera merely because trajectory time elapsed.
        # Require every measured joint to be close to the intended success
        # pose and remain settled for a short window. This also handles the
        # real arm's repeatable small static residual without weakening the
        # shared JTC tolerances for unrelated motions.
        return self._wait_for_success_pose_settled()

    def _wait_for_success_pose_settled(self):
        deadline = time.monotonic() + self.success_validation_timeout
        settled_since = None
        settled_reference = None
        last_positions = self._positions_snapshot()

        while time.monotonic() < deadline:
            positions = self._positions_snapshot()
            if all(position is not None for position in positions):
                errors = [
                    abs(positions[i] - self.success_positions[i])
                    for i in range(len(self.joints))
                ]
                within_tolerance = all(
                    errors[i] <= self.success_position_tolerance[i]
                    for i in range(len(self.joints))
                )

                if within_tolerance:
                    now = time.monotonic()
                    if settled_reference is None:
                        settled_reference = list(positions)
                        settled_since = now
                    elif max(
                            abs(positions[i] - settled_reference[i])
                            for i in range(len(self.joints))) > self.success_settle_movement:
                        # It entered the position window while still moving;
                        # start the settle timer again from the new pose.
                        settled_reference = list(positions)
                        settled_since = now
                    elif now - settled_since >= self.success_settle_time:
                        self.get_logger().info(
                            'Success pose independently verified from measured '
                            f'joint positions; errors={errors}')
                        return True
                else:
                    settled_since = None
                    settled_reference = None
                last_positions = positions
            time.sleep(0.05)

        errors = [
            None if last_positions[i] is None else
            abs(last_positions[i] - self.success_positions[i])
            for i in range(len(self.joints))
        ]
        self.get_logger().error(
            'Measured arm did not settle at success pose: '
            f'errors={errors}, tolerances={self.success_position_tolerance}')
        return False

    def _face_control_request(self):
        """Return an outstanding safety transition requested by ROS service."""
        with self._state_lock:
            if self._abort_requested:
                return 'abort'
            if self._go_home_requested:
                self._go_home_requested = False
                return 'home'
        return None

    def _enter_face_servo_mode(self, positions):
        """Hand the live pose from the JTC to the direct MIT reference input."""
        # Prime the subscriber buffer before changing chained mode, then send
        # the same reference again afterward. The impedance controller sees
        # the measured pose on both sides of the switch and cannot jump toward
        # an old trajectory setpoint.
        self._publish_impedance_command(positions)
        if not self._switch_controllers(
                activate=[self.impedance_controller],
                deactivate=[self.impedance_trajectory_controller,
                            self.position_adapter_controller,
                            self.position_trajectory_controller]):
            self.get_logger().error(
                'Failed to enter direct impedance mode for face tracking')
            return False
        self._publish_impedance_command(positions)
        return True

    def _stream_face_scan_target(self, target):
        """Stream one smooth scan move and stop early when a face appears."""
        start = self._positions_snapshot()
        if any(position is None for position in start):
            self.get_logger().error('Lost /joint_states during face scan')
            return 'failed'
        start = list(start)
        duration = minimum_quintic_duration(
            start, target,
            [self.face_scan_max_joint_speed] * len(self.joints),
            self.face_scan_min_motion_time)
        period = 1.0 / self.face_servo_rate_hz
        started_at = time.monotonic()
        next_tick = started_at

        while True:
            request = self._face_control_request()
            if request is not None:
                return request
            now = time.monotonic()
            progress = min(1.0, (now - started_at) / duration)
            command = trajectory_reference(
                start, target, [False] * len(self.joints), target, progress)
            self._publish_impedance_command(command)
            if self._face_sample_if_acceptable(
                    self.face_servo_detection_timeout) is not None:
                return 'detected'
            if progress >= 1.0:
                return 'ok'
            next_tick += period
            next_tick = max(next_tick, time.monotonic())
            time.sleep(max(0.0, next_tick - time.monotonic()))

    def _set_face_stability(self, stable_for):
        with self._face_lock:
            self._face_stable_for = stable_for

    def _face_sample_if_acceptable(self, max_age=None):
        now = time.monotonic()
        sample = self._face_sample_snapshot()
        if max_age is None:
            max_age = self.face_detection_max_age
        if face_is_acceptable(
                sample, now, max_age,
                self.face_min_bbox_area_ratio,
                self.face_max_bbox_area_ratio):
            return sample
        return None

    def _center_face(self, reference, tracked_sample):
        """Continuously servo the wrist until the head bbox is centered."""
        deadline = time.monotonic() + self.face_center_timeout
        centered_since = None
        following_since = None
        lost_since = None
        command = self._positions_snapshot()
        if any(position is None for position in command):
            return 'failed'
        command = list(command)
        period = 1.0 / self.face_servo_rate_hz
        previous_tick = time.monotonic()
        next_tick = previous_tick

        with self._state_lock:
            self._phase = 'face_centering'

        while time.monotonic() < deadline:
            request = self._face_control_request()
            if request is not None:
                return request

            now = time.monotonic()
            dt = min(max(0.0, now - previous_tick), 2.0 * period)
            previous_tick = now
            sample = self._face_sample_if_acceptable(
                self.face_servo_detection_timeout)
            if sample is None:
                centered_since = None
                if lost_since is None:
                    lost_since = now
                elif (now - lost_since >=
                      self.face_target_lost_timeout):
                    self.get_logger().warn(
                        'Stable face was lost during centering; resuming scan')
                    return 'lost'
                # Hold the last safe equilibrium. Never keep integrating an
                # image error after the detector has stopped refreshing.
                self._publish_impedance_command(command)
                next_tick += period
                next_tick = max(next_tick, time.monotonic())
                time.sleep(max(0.0, next_tick - time.monotonic()))
                continue

            if not face_matches_track(
                    tracked_sample, sample,
                    self.face_track_center_jump_tolerance,
                    self.face_track_area_relative_jump_tolerance):
                self.get_logger().warn(
                    'Detected a different face during centering; '
                    'resuming scan instead of chasing the new target')
                return 'lost'
            tracked_sample = sample

            lost_since = None
            error_x = sample.error_x
            error_y = sample.error_y
            inside_center = (
                abs(error_x) <= self.face_center_tolerance and
                abs(error_y) <= self.face_center_tolerance)
            inside_release = (
                abs(error_x) <= self.face_center_release_tolerance and
                abs(error_y) <= self.face_center_release_tolerance)
            if centered_since is not None and not inside_release:
                centered_since = None
            if centered_since is None and inside_center:
                centered_since = now
            if (centered_since is not None and
                    now - centered_since >= self.face_center_dwell_time):
                if following_since is None:
                    following_since = now
                    with self._state_lock:
                        self._phase = 'face_following'
                    self.get_logger().info(
                        'Face centered; maintaining continuous tracking for '
                        f'{self.face_follow_duration:.1f} s')
                if (now - following_since >= self.face_follow_duration and
                        inside_release):
                    self.get_logger().info(
                        'Face follow complete: '
                        f'error_x={error_x:.3f}, '
                        f'error_y={error_y:.3f}, '
                        f'area_ratio={sample.area_ratio:.4f}')
                    return 'centered'

            previous_command = command
            command = servo_target(
                current_command=command,
                reference=reference,
                lower_limits=self.joint_lower_limits,
                upper_limits=self.joint_upper_limits,
                yaw_index=self.face_yaw_index,
                pitch_index=self.face_pitch_index,
                error_x=error_x,
                error_y=error_y,
                yaw_gain=self.face_center_gain,
                pitch_gain=self.face_center_gain,
                max_speed=self.face_servo_max_joint_speed,
                dt=dt,
                yaw_direction=self.face_yaw_direction,
                pitch_direction=self.face_pitch_direction,
                yaw_max_offset=self.face_yaw_max_offset,
                pitch_max_offset=self.face_pitch_max_offset,
                deadband=self.face_center_tolerance,
                joint_margin=self.face_joint_margin,
            )
            command_velocity = [0.0] * len(self.joints)
            if dt > 0.0:
                command_velocity[self.face_yaw_index] = (
                    command[self.face_yaw_index] -
                    previous_command[self.face_yaw_index]) / dt
                command_velocity[self.face_pitch_index] = (
                    command[self.face_pitch_index] -
                    previous_command[self.face_pitch_index]) / dt
            self._publish_impedance_command(command, command_velocity)
            next_tick += period
            next_tick = max(next_tick, time.monotonic())
            time.sleep(max(0.0, next_tick - time.monotonic()))

        self.get_logger().warn('Face centering timed out; resuming scan')
        return 'lost'

    def _run_face_tracking(self):
        """Scan for a nearby stable face, then center it in the RGB image."""
        reference = self._positions_snapshot()
        if any(position is None for position in reference):
            self.get_logger().error(
                'Lost /joint_states before starting face tracking')
            return 'failed'
        reference = list(reference)
        if not self._enter_face_servo_mode(reference):
            return 'failed'
        scan_pairs = list(zip(
            self.face_scan_offsets[0::2],
            self.face_scan_offsets[1::2],
        ))
        gate = FaceStabilityGate(
            self.face_stable_time,
            self.face_stability_center_tolerance,
            self.face_stability_area_relative_tolerance,
        )
        started_at = time.monotonic()
        scan_index = 0
        self._clear_face_sample()
        self._speak(self.tts_face_searching_text)

        while True:
            request = self._face_control_request()
            if request is not None:
                return request
            if (self.face_search_timeout > 0.0 and
                    time.monotonic() - started_at >=
                    self.face_search_timeout):
                with self._state_lock:
                    self._phase = 'face_not_found'
                self._speak(self.tts_face_not_found_text)
                self.get_logger().warn('Face search timed out')
                return 'not_found'

            yaw_offset, pitch_offset = scan_pairs[scan_index]
            scan_index = (scan_index + 1) % len(scan_pairs)
            target = scan_target(
                reference,
                self.joint_lower_limits,
                self.joint_upper_limits,
                self.face_yaw_index,
                self.face_pitch_index,
                yaw_offset,
                pitch_offset,
                self.face_joint_margin,
            )
            with self._state_lock:
                self._phase = 'face_searching'
            gate.reset()
            self._set_face_stability(0.0)
            result = self._stream_face_scan_target(target)
            if result not in ('ok', 'detected'):
                return result

            hold_deadline = time.monotonic() + self.face_scan_hold_time
            while time.monotonic() < hold_deadline:
                request = self._face_control_request()
                if request is not None:
                    return request
                sample = self._face_sample_if_acceptable(
                    self.face_servo_detection_timeout)
                if sample is None:
                    gate.reset()
                    self._set_face_stability(0.0)
                    with self._state_lock:
                        self._phase = 'face_searching'
                else:
                    with self._state_lock:
                        self._phase = 'face_stabilizing'
                    stable = gate.update(sample, time.monotonic())
                    self._set_face_stability(gate.stable_for)
                    if stable:
                        self.get_logger().info(
                            'Stable nearby face acquired: '
                            f'area_ratio={sample.area_ratio:.4f}, '
                            f'error_x={sample.error_x:.3f}, '
                            f'error_y={sample.error_y:.3f}')
                        self._speak(self.tts_face_locked_text)
                        center_result = self._center_face(reference, sample)
                        if center_result == 'centered':
                            with self._state_lock:
                                self._phase = 'face_centered'
                            self._speak(self.tts_face_centered_text)
                            return 'centered'
                        if center_result != 'lost':
                            return center_result
                        gate.reset()
                        self._set_face_stability(0.0)
                        break
                time.sleep(0.05)

    def _build_face_inspection_trajectory(self, anchor):
        """Build one dense, velocity-continuous JTC gesture trajectory."""
        n = len(self.joints)
        rate = self.face_gesture_trajectory_rate_hz
        points = [list(anchor)]
        velocities = [[0.0] * n]
        times = [0.15]  # seed the reactivated JTC at the live captured pose
        current = list(anchor)
        time_cursor = times[0]

        def append_quintic(target, duration):
            nonlocal current, time_cursor
            start = list(current)
            steps = max(2, int(math.ceil(duration * rate)))
            for step in range(1, steps + 1):
                u = step / steps
                blend = quintic_blend(u)
                blend_rate = 30.0 * u ** 2 * (1.0 - u) ** 2 / duration
                points.append([
                    start[i] + blend * (target[i] - start[i])
                    for i in range(n)
                ])
                velocities.append([
                    blend_rate * (target[i] - start[i])
                    for i in range(n)
                ])
                times.append(time_cursor + u * duration)
            current = list(target)
            time_cursor += duration

        orbit_start = scan_target(
            anchor, self.joint_lower_limits, self.joint_upper_limits,
            self.face_yaw_index, self.face_pitch_index,
            self.face_orbit_yaw_amplitude, 0.0,
            self.face_joint_margin)
        entry_duration = minimum_quintic_duration(
            current, orbit_start,
            [self.face_orbit_max_joint_speed] * n,
            self.face_orbit_min_motion_time)
        append_quintic(orbit_start, entry_duration)

        # Traverse one ellipse with quintic angle timing. The angular speed is
        # zero at both ends, matching the entry/exit segment velocities while
        # staying continuous through every dense interior point.
        orbit_duration = max(
            self.face_orbit_min_motion_time,
            (2.0 * math.pi * 1.875 *
             max(self.face_orbit_yaw_amplitude,
                 self.face_orbit_pitch_amplitude) /
             self.face_orbit_max_joint_speed))
        orbit_steps = max(8, int(math.ceil(orbit_duration * rate)))
        orbit_started_at = time_cursor
        for step in range(1, orbit_steps + 1):
            u = step / orbit_steps
            angle = 2.0 * math.pi * quintic_blend(u)
            angle_rate = (
                2.0 * math.pi * 30.0 * u ** 2 * (1.0 - u) ** 2 /
                orbit_duration)
            yaw_offset = self.face_orbit_yaw_amplitude * math.cos(angle)
            pitch_offset = self.face_orbit_pitch_amplitude * math.sin(angle)
            target = scan_target(
                anchor, self.joint_lower_limits, self.joint_upper_limits,
                self.face_yaw_index, self.face_pitch_index,
                yaw_offset, pitch_offset, self.face_joint_margin)
            velocity = [0.0] * n
            velocity[self.face_yaw_index] = (
                -self.face_orbit_yaw_amplitude * math.sin(angle) *
                angle_rate)
            velocity[self.face_pitch_index] = (
                self.face_orbit_pitch_amplitude * math.cos(angle) *
                angle_rate)
            points.append(target)
            velocities.append(velocity)
            times.append(orbit_started_at + u * orbit_duration)
        current = list(orbit_start)
        time_cursor += orbit_duration

        exit_duration = minimum_quintic_duration(
            current, anchor,
            [self.face_orbit_max_joint_speed] * n,
            self.face_orbit_min_motion_time)
        append_quintic(anchor, exit_duration)

        def append_pitch_roll(pitch_offset, roll_offset, speed, minimum):
            target = inspection_gesture_target(
                anchor, self.joint_lower_limits, self.joint_upper_limits,
                self.face_pitch_index, self.face_gesture_roll_index,
                pitch_offset, roll_offset, self.face_joint_margin)
            duration = minimum_quintic_duration(
                current, target, [speed] * n, minimum)
            append_quintic(target, duration)

        # One tilt, followed by two crisp nod cycles.
        append_pitch_roll(
            0.0, self.face_gesture_roll_amplitude,
            self.face_gesture_max_joint_speed,
            self.face_gesture_min_motion_time)
        append_pitch_roll(
            0.0, 0.0, self.face_gesture_max_joint_speed,
            self.face_gesture_min_motion_time)
        for _ in range(2):
            append_pitch_roll(
                self.face_gesture_pitch_amplitude, 0.0,
                self.face_gesture_nod_max_joint_speed,
                self.face_gesture_nod_min_motion_time)
            append_pitch_roll(
                0.0, 0.0, self.face_gesture_nod_max_joint_speed,
                self.face_gesture_nod_min_motion_time)
        return points, times, velocities

    def _run_face_inspection_gesture(self):
        """Run orbit, tilt, and two nods as one multi-point JTC goal."""
        anchor = self._positions_snapshot()
        if any(position is None for position in anchor):
            self.get_logger().error(
                'Lost /joint_states before face inspection gesture')
            return 'failed'
        anchor = list(anchor)
        with self._state_lock:
            self._phase = 'face_gesture'
        points, times, velocities = self._build_face_inspection_trajectory(
            anchor)
        self.get_logger().info(
            'Face profile captured; sending one smooth multi-point JTC goal '
            f'({len(points)} points, orbit yaw/pitch='
            f'{self.face_orbit_yaw_amplitude:.2f}/'
            f'{self.face_orbit_pitch_amplitude:.2f} rad, nod speed='
            f'{self.face_gesture_nod_max_joint_speed:.2f} rad/s, '
            f'duration={times[-1]:.2f} s)')

        # The centering loop left the impedance controller in direct mode.
        # Prime it at the measured pose, then reactivate its chained JTC; the
        # first point in this single goal keeps that pose for 0.15 s so no
        # stale JTC reference can produce a backswing.
        self._publish_impedance_command(anchor)
        if not self._switch_controllers(
                activate=[self.impedance_controller,
                          self.impedance_trajectory_controller],
                deactivate=[self.position_adapter_controller,
                            self.position_trajectory_controller]):
            self.get_logger().error(
                'Failed to enter impedance JTC mode for face gesture')
            return 'failed'
        if not self._send_trajectory(
                self._impedance_traj_client, points, times,
                velocities=velocities):
            self.get_logger().error('Face gesture trajectory goal failed')
            return 'failed'

        deadline = time.monotonic() + times[-1] + 2.0
        while time.monotonic() < deadline:
            request = self._face_control_request()
            if request is not None:
                self._cancel_active_trajectory()
                return request
            with self._active_goal_lock:
                active = self._active_trajectory_goal is not None
            if not active:
                self.get_logger().info(
                    'Face inspection JTC trajectory complete')
                return 'ok'
            time.sleep(0.02)
        self._cancel_active_trajectory()
        self.get_logger().error('Face inspection JTC trajectory timed out')
        return 'failed'

    def _run_palm_interaction(self, finish_on_mesh_ready=None):
        """Mirror open-palm depth motion around the current arm pose."""
        if finish_on_mesh_ready is None:
            finish_on_mesh_ready = self.gesture_finish_on_mesh_ready
        anchor = self._positions_snapshot()
        if any(position is None for position in anchor):
            self.get_logger().error(
                'Lost /joint_states before palm interaction')
            return 'failed'
        anchor = list(anchor)
        try:
            servo = CameraAxisServo(
                anchor,
                self.joint_lower_limits,
                self.joint_upper_limits,
                self.gesture_joint_max_offsets,
                max_linear_speed=self.gesture_max_linear_speed,
                max_joint_speed=self.gesture_max_joint_speed,
                position_gain=self.gesture_position_gain,
            )
            depth_filter = PalmDepthFilter(
                deadzone=self.gesture_depth_deadzone,
                gain=self.gesture_depth_gain,
                max_offset=self.gesture_max_offset,
                smoothing_time=self.gesture_depth_smoothing_time,
                sample_timeout=self.gesture_sample_timeout,
                reacquire_timeout=self.gesture_reacquire_timeout,
            )
        except ValueError as error:
            self.get_logger().error(
                f'Palm interaction configuration is invalid: {error}')
            return 'failed'

        if not self._enter_face_servo_mode(anchor):
            return 'failed'
        # Face inspection used the same success-phase gains, but set them
        # explicitly so this stage also works when the inspection gesture is
        # disabled or a previous tuning session changed controller parameters.
        if not self._set_impedance_params(
                stiffness=self.success_stiffness,
                damping_coefficients=self.locked_damping,
                **{'gravity_compensation.factor': 1.0}):
            return 'failed'
        request = self._prepare_gripper_for_palm_interaction()
        if request is not None:
            return request
        with self._state_lock:
            self._stiffness_state = list(self.success_stiffness)
            self._phase = 'gesture_interaction'
        with self._palm_lock:
            self._latest_palm_control = None
            self._latest_palm_received_at = 0.0
            self._palm_game_state = {
                'active': True,
                'hand_detected': False,
                'distance_m': None,
                'offset_m': 0.0,
                'control_active': False,
                'label': 'unknown',
                'palm_open': False,
                'palm_source': 'none',
                'distance_valid': False,
                'control_reason': 'waiting_for_open_palm',
                'depth_reason': '',
                'sample_age_ms': None,
                'depth_age_ms': None,
                'depth_skew_ms': None,
                'inference_ms': None,
                'source_age_ms': None,
                'baseline_m': None,
                'filtered_m': None,
                'tracking_error_rad': 0.0,
                'command_delta_rad': 0.0,
                'yaw_tracking': False,
                'yaw_error': None,
                'yaw_offset_rad': 0.0,
                'mesh_status': 'idle',
                'mesh_progress': 0,
            }
        self._speak(self.tts_gesture_interaction_text)
        self.get_logger().info(
            'Palm push/pull interaction active: open_palm + aligned depth, '
            f'travel=+/-{self.gesture_max_offset:.3f} m, '
            f'speed<={self.gesture_max_linear_speed:.3f} m/s, '
            f'J5 palm yaw={self.gesture_yaw_enabled}, '
            f'gain={self.gesture_yaw_gain:.2f}, '
            f'speed<={self.gesture_yaw_max_speed:.2f} rad/s')

        command = list(anchor)
        pose_reference = list(anchor)
        yaw_target = anchor[self.face_yaw_index]
        servo_offset = 0.0
        period = 1.0 / self.gesture_control_rate_hz
        started_at = time.monotonic()
        previous_tick = started_at
        next_tick = started_at
        finish_reason = 'timeout'
        with self._mesh_lock:
            mesh = dict(self._mesh_state)

        while True:
            request = self._face_control_request()
            if request is not None:
                finish_reason = request
                break
            now = time.monotonic()
            elapsed = now - started_at
            with self._mesh_lock:
                mesh = dict(self._mesh_state)
            with self._reconstruction_lock:
                reconstruction_status = self._reconstruction_state['status']
            if (elapsed >= self.gesture_interaction_min_duration and
                    finish_on_mesh_ready and
                    mesh['status'] in (
                        'success', 'error', 'failed', 'cancelled')):
                finish_reason = (
                    'model_ready' if mesh['status'] == 'success'
                    else 'model_failed')
                break
            if (elapsed >= self.gesture_interaction_min_duration and
                    finish_on_mesh_ready and
                    reconstruction_status == 'error'):
                finish_reason = 'model_failed'
                break
            if elapsed >= self.gesture_interaction_max_duration:
                break

            dt = min(max(now - previous_tick, 0.0), 2.0 * period)
            previous_tick = now
            with self._palm_lock:
                payload = self._latest_palm_control
                received_at = self._latest_palm_received_at
            target_offset = depth_filter.update(payload, received_at, now)
            hand_detected = target_offset is not None
            sample_age_ms = (
                None if received_at <= 0.0 else
                max(0.0, (now - received_at) * 1000.0))
            diagnostic_payload = payload if isinstance(payload, dict) else {}
            yaw_error = palm_center_error_x(diagnostic_payload)
            yaw_visible = (
                self.gesture_yaw_enabled and
                yaw_error is not None and
                received_at > 0.0 and
                now - received_at <= self.gesture_sample_timeout)
            depth_age_ms = (
                float(diagnostic_payload['depth_age_sec']) * 1000.0
                if isinstance(diagnostic_payload.get('depth_age_sec'), (int, float))
                else None)
            depth_skew_ms = (
                float(diagnostic_payload['depth_skew_sec']) * 1000.0
                if isinstance(diagnostic_payload.get('depth_skew_sec'), (int, float))
                else None)
            distance = None
            if hand_detected:
                try:
                    distance = float(payload['distance_m'])
                except (KeyError, TypeError, ValueError):
                    hand_detected = False
                    target_offset = None
            if target_offset is not None:
                servo_offset = target_offset
            if yaw_visible and dt > 0.0:
                try:
                    yaw_target = bounded_joint_servo_target(
                        current_target=yaw_target,
                        reference=anchor[self.face_yaw_index],
                        lower_limit=self.joint_lower_limits[
                            self.face_yaw_index],
                        upper_limit=self.joint_upper_limits[
                            self.face_yaw_index],
                        error=yaw_error,
                        gain=self.gesture_yaw_gain,
                        max_speed=self.gesture_yaw_max_speed,
                        dt=dt,
                        direction=self.face_yaw_direction,
                        max_offset=self.gesture_yaw_max_offset,
                        deadband=self.gesture_yaw_deadband,
                        joint_margin=self.face_joint_margin,
                    )
                except ValueError as error:
                    self.get_logger().error(
                        f'Palm yaw servo failed: {error}')
                    finish_reason = 'failed'
                    break
                pose_reference[self.face_yaw_index] = yaw_target

            measured = self._positions_snapshot()
            if any(position is None for position in measured):
                self.get_logger().error(
                    'Lost /joint_states during palm interaction')
                finish_reason = 'failed'
                break
            tracking_error = max(
                abs(measured[index] - command[index])
                for index in range(len(command)))
            command_delta = 0.0
            if tracking_error > self.gesture_tracking_error_limit:
                # A hand or obstacle can block this compliant game. Rebase the
                # command at the measured pose instead of building more error.
                command = list(measured)
                self._publish_impedance_command(command)
                self.get_logger().warn(
                    'Palm servo tracking error exceeded limit; holding the '
                    'measured pose before resuming',
                    throttle_duration_sec=1.0)
            elif (hand_detected or yaw_visible) and dt > 0.0:
                try:
                    step = servo.step(
                        command, servo_offset, dt,
                        reference=pose_reference,
                        controlled_joint_index=(
                            self.face_yaw_index
                            if self.gesture_yaw_enabled else None),
                    )
                except (ValueError, np.linalg.LinAlgError) as error:
                    self.get_logger().error(
                        f'Palm Cartesian servo failed: {error}')
                    finish_reason = 'failed'
                    break
                previous_command = command
                command = step.positions
                command_delta = max(
                    abs(command[index] - previous_command[index])
                    for index in range(len(command)))
                self._publish_impedance_command(command, step.velocities)
            else:
                # A lost/closed hand freezes the last equilibrium immediately.
                self._publish_impedance_command(command)

            if hand_detected:
                self.get_logger().info(
                    'Palm servo accepted: '
                    f'distance={distance:.3f} m, '
                    f'baseline={depth_filter.baseline:.3f} m, '
                    f'offset={target_offset:+.3f} m, '
                    f'joint_step={command_delta:.5f} rad, '
                    f'tracking_error={tracking_error:.3f} rad, '
                    f'input_age={sample_age_ms:.1f} ms, '
                    f'depth_age={depth_age_ms} ms, '
                    f'depth_skew={depth_skew_ms} ms',
                    throttle_duration_sec=1.0)
            else:
                self.get_logger().warn(
                    'Palm servo waiting: '
                    f'reason={depth_filter.last_reason}, '
                    f'label={diagnostic_payload.get("label", "unknown")}, '
                    f'active={diagnostic_payload.get("active", False)}, '
                    'distance_valid='
                    f'{diagnostic_payload.get("distance_valid", False)}, '
                    f'sample_age_ms={sample_age_ms}, '
                    f'depth_age_ms={depth_age_ms}, '
                    f'depth_skew_ms={depth_skew_ms}',
                    throttle_duration_sec=2.0)

            with self._palm_lock:
                self._palm_game_state = {
                    'active': True,
                    'hand_detected': hand_detected,
                    'distance_m': distance,
                    'offset_m': 0.0 if target_offset is None else target_offset,
                    'control_active': diagnostic_payload.get('active') is True,
                    'label': str(diagnostic_payload.get('label', 'unknown')),
                    'palm_open': diagnostic_payload.get('palm_open') is True,
                    'palm_source': str(
                        diagnostic_payload.get('palm_source', 'none')),
                    'distance_valid': (
                        diagnostic_payload.get('distance_valid') is True),
                    'control_reason': depth_filter.last_reason,
                    'depth_reason': str(
                        diagnostic_payload.get('depth_reason', '')),
                    'sample_age_ms': sample_age_ms,
                    'depth_age_ms': depth_age_ms,
                    'depth_skew_ms': depth_skew_ms,
                    'inference_ms': diagnostic_payload.get('inference_ms'),
                    'source_age_ms': diagnostic_payload.get('source_age_ms'),
                    'baseline_m': depth_filter.baseline,
                    'filtered_m': depth_filter.filtered,
                    'tracking_error_rad': tracking_error,
                    'command_delta_rad': command_delta,
                    'yaw_tracking': yaw_visible,
                    'yaw_error': yaw_error,
                    'yaw_offset_rad': (
                        yaw_target - anchor[self.face_yaw_index]),
                    'mesh_status': mesh['status'],
                    'mesh_progress': mesh['progress'],
                }
            next_tick += period
            next_tick = max(next_tick, time.monotonic())
            time.sleep(max(0.0, next_tick - time.monotonic()))

        self._publish_impedance_command(command)
        with self._palm_lock:
            final_state = dict(self._palm_game_state)
            final_state['active'] = False
            final_state['hand_detected'] = False
            final_state['mesh_status'] = mesh['status']
            final_state['mesh_progress'] = mesh['progress']
            self._palm_game_state = final_state
        self.get_logger().info(
            f'Palm push/pull interaction complete: reason={finish_reason}, '
            f'mesh={mesh["status"]}, elapsed={time.monotonic() - started_at:.1f} s')
        if finish_reason == 'model_ready':
            self._speak(self.tts_model_ready_text)
            return 'ok'
        if finish_reason in ('abort', 'home'):
            return finish_reason
        if finish_reason == 'failed':
            return 'failed'
        return 'ok'

    def _publish_impedance_command(self, positions, velocities=None):
        """
        Direct topic command to zephyr_arm_impedance_controller (unchained
        mode - the impedance_trajectory_controller/JTC is kept deactivated).
        Publishing this every loop tick keeps the controller's own reference
        glued to the live measured pose while a joint is free (stiffness=0,
        so it has no force effect); this avoids the JTC-staleness bug where
        locking a joint after a long free period made it swing back toward
        a stale trajectory setpoint before settling on the real target.
        """
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in positions]
        point.velocities = (
            [0.0] * len(positions) if velocities is None else
            [float(v) for v in velocities]
        )
        self._impedance_command_pub.publish(point)

    def _open_gripper(self, position):
        msg = JointState()
        msg.name = ['omni_picker_finger_joint']
        msg.position = [float(position)]
        msg.velocity = [0.2] * len(msg.position)
        self._gripper_pub.publish(msg)

    def _prepare_gripper_for_palm_interaction(self):
        """Open and settle the gripper before gesture perception becomes active."""
        self.get_logger().info('Opening gripper before palm interaction')
        self._open_gripper(self.gripper_open_position)
        time.sleep(self.gripper_settle_time)
        return self._face_control_request()

    def _generate_targets(self):
        n = len(self.joints)
        if len(self.fixed_targets) == n and not any(math.isnan(v) for v in self.fixed_targets):
            self.get_logger().info('Using fixed_targets for this round')
            return list(self.fixed_targets)
        targets = []
        for i in range(n):
            lo, hi = self.joint_lower_limits[i], self.joint_upper_limits[i]
            margin = self.target_margin_ratio * (hi - lo)
            targets.append(random.uniform(lo + margin, hi - margin))
        return targets

    def _within_hard_limits(self, positions):
        for i, pos in enumerate(positions):
            if pos is None:
                continue
            if pos < self.joint_lower_limits[i] - 0.05 or pos > self.joint_upper_limits[i] + 0.05:
                return False
        return True

    # ------------------------------------------------------------------ #
    #  Game sequence (background thread)
    # ------------------------------------------------------------------ #

    def _start_round(self):
        try:
            self._run_round()
        except Exception:
            self.get_logger().error('ghost round crashed', exc_info=True)
            self._safe_abort()
        finally:
            with self._state_lock:
                self._running = False

    def _run_round(self):
        n = len(self.joints)
        with self._state_lock:
            self._phase = 'setup'
            self._camera_ready = False
            self._locked = [False] * n
            self._dwell = [0.0] * n
            self._stiffness_state = [0.0] * n
            # A mock-palm request can arrive immediately after ~/start but
            # before this background thread acquires the lock.
            self._abort_requested = self._mock_palm_transition_pending
            self._go_home_requested = False
        self._reset_reconstruction_state()
        self._speak(self.tts_setup_text)

        self.get_logger().info('Switching into impedance interaction chain')
        if not self._switch_controllers(
                activate=[self.impedance_controller],
                deactivate=[self.impedance_trajectory_controller, self.position_adapter_controller,
                            self.position_trajectory_controller]):
            self._safe_abort()
            return

        if not self._set_impedance_params(
                stiffness=[0.0] * n,
                damping_coefficients=self.free_damping,
                **{'gravity_compensation.factor': 1.0}):  # restore in case a prior round fully relaxed it
            self._safe_abort()
            return

        # Seed the reference at the current pose so nothing jumps on activation.
        deadline = time.monotonic() + 3.0
        positions = self._positions_snapshot()
        while any(p is None for p in positions) and time.monotonic() < deadline:
            time.sleep(0.05)
            positions = self._positions_snapshot()
        if any(p is None for p in positions):
            self.get_logger().error('Never received full /joint_states, aborting round')
            self._safe_abort()
            return
        self._publish_impedance_command(positions)

        targets = self._generate_targets()
        with self._state_lock:
            self._targets = list(targets)
            self._secret_targets = list(targets)
            self._phase = 'searching'
        self._speak(self.tts_searching_text)
        self.get_logger().debug(f'Secret targets: {targets}')

        period = 1.0 / self.loop_rate_hz
        mock_start_positions = None
        mock_started_at = None
        mock_duration = None
        mock_deadline = None
        while True:
            with self._state_lock:
                abort_requested = self._abort_requested
                if self._go_home_requested:
                    self._go_home_requested = False
                    go_home = True
                else:
                    go_home = False
                mock_requested = self._mock_solve_requested
                self._mock_solve_requested = False
                locked_snapshot = list(self._locked)
            if abort_requested:
                self.get_logger().warn('Abort requested, stopping round')
                self._safe_abort()
                return
            if go_home:
                self.get_logger().info('Go-home requested mid-round, heading home instead')
                self._return_home()
                return
            if all(locked_snapshot):
                break

            positions = self._positions_snapshot()
            if not self._within_hard_limits(positions):
                self.get_logger().error('Joint position outside safe range, aborting round')
                self._safe_abort()
                return

            if mock_requested:
                if any(position is None for position in positions):
                    self.get_logger().error(
                        'Cannot start mock trajectory without full joint state')
                else:
                    mock_start_positions = list(positions)
                    mock_duration = minimum_quintic_duration(
                        mock_start_positions, targets,
                        self.mock_solve_max_velocity,
                        self.mock_solve_min_duration)
                    mock_target_stiffness = [
                        self._stiffness_state[index]
                        if locked_snapshot[index]
                        else self.mock_solve_stiffness[index]
                        for index in range(n)
                    ]
                    mock_damping = [
                        self.locked_damping[index]
                        if locked_snapshot[index]
                        else self.free_damping[index]
                        for index in range(n)
                    ]
                    # Raise stiffness while the equilibrium reference is still
                    # at the measured pose, then begin the motion. This avoids
                    # a torque step when development automation is engaged.
                    self._ramp_stiffness(
                        mock_target_stiffness, mock_damping)
                    mock_started_at = time.monotonic()
                    mock_deadline = (
                        mock_started_at + mock_duration +
                        self.mock_solve_settle_timeout)
                    with self._state_lock:
                        self._mock_solve_active = True
                    self.get_logger().info(
                        'Mock solve following a soft quintic trajectory to '
                        f'the secret pose over {mock_duration:.2f} s')

            newly_locked = []
            for i in range(n):
                if locked_snapshot[i] or positions[i] is None:
                    continue
                if abs(positions[i] - targets[i]) <= self.match_tolerance:
                    self._dwell[i] += period
                    if self._dwell[i] >= self.match_dwell_time:
                        newly_locked.append(i)
                else:
                    self._dwell[i] = 0.0

            for i in newly_locked:
                self._lock_joint(i, positions)

            # Keep the reference glued to the live pose every tick (locked
            # joints hold their achieved target) - this is what prevents the
            # old JTC-based stale-setpoint backswing on lock.
            with self._state_lock:
                locked_snapshot = list(self._locked)
                locked_hold = list(self._targets)
                hold_positions = [
                    self._targets[j] if self._locked[j] else positions[j] for j in range(n)
                ]
                mock_active = self._mock_solve_active

            if mock_active:
                now = time.monotonic()
                if now <= mock_deadline:
                    hold_positions = trajectory_reference(
                        mock_start_positions, targets, locked_snapshot,
                        locked_hold,
                        (now - mock_started_at) / mock_duration)
                else:
                    # Leave the ordinary hand-play mode usable if the soft
                    # trajectory cannot pull the real arm into every target.
                    self.get_logger().warn(
                        'Mock solve timed out before every joint locked; '
                        'returning remaining joints to free mode')
                    release_stiffness = [
                        self._stiffness_state[index]
                        if locked_snapshot[index] else 0.0
                        for index in range(n)
                    ]
                    release_damping = [
                        self.locked_damping[index]
                        if locked_snapshot[index]
                        else self.free_damping[index]
                        for index in range(n)
                    ]
                    self._ramp_stiffness(
                        release_stiffness, release_damping)
                    with self._state_lock:
                        self._mock_solve_active = False
                    hold_positions = [
                        locked_hold[index]
                        if locked_snapshot[index] else positions[index]
                        for index in range(n)
                    ]
            self._publish_impedance_command(hold_positions)

            time.sleep(period)

        with self._state_lock:
            self._phase = 'all_found'
            self._mock_solve_active = False
        self._speak(self.tts_all_found_text)
        self.get_logger().info('All joints found! Opening gripper')

        self._open_gripper(self.gripper_open_position)
        time.sleep(self.gripper_settle_time)

        if not self._move_to_success_pose():
            self.get_logger().error('Failed to reach the post-game success pose')
            self._safe_abort()
            return

        # The web dashboard uses this state to reveal the camera only after
        # the arm has actually completed the turn, not merely when all ghosts
        # have been found.
        with self._state_lock:
            self._phase = 'success_pose_reached'
            self._camera_ready = True
        self._speak(self.tts_success_pose_text)

        if self.enable_face_tracking:
            face_result = self._run_face_tracking()
            if face_result == 'abort':
                self.get_logger().warn('Abort requested during face tracking')
                self._safe_abort()
                return
            if face_result == 'home':
                self.get_logger().info(
                    'Go-home requested during face tracking')
                self._return_home()
                return
            if face_result == 'failed':
                self.get_logger().error('Face tracking motion failed')
                self._safe_abort()
                return
            if face_result == 'not_found':
                return
            self._submit_face_reconstruction()
            if self.face_gesture_enabled:
                gesture_result = self._run_face_inspection_gesture()
                if gesture_result == 'abort':
                    self.get_logger().warn(
                        'Abort requested during face inspection gesture')
                    self._safe_abort()
                    return
                if gesture_result == 'home':
                    self.get_logger().info(
                        'Go-home requested during face inspection gesture')
                    self._return_home()
                    return
                if gesture_result == 'failed':
                    self.get_logger().error(
                        'Face inspection gesture failed')
                    self._safe_abort()
                    return
            if self.enable_gesture_interaction:
                interaction_result = self._run_palm_interaction()
                if interaction_result == 'abort':
                    self.get_logger().warn(
                        'Abort requested during palm interaction')
                    self._safe_abort()
                    return
                if interaction_result == 'home':
                    self.get_logger().info(
                        'Go-home requested during palm interaction')
                    self._return_home()
                    return
                if interaction_result == 'failed':
                    self.get_logger().error(
                        'Palm interaction motion failed')
                    self._safe_abort()
                    return

        if not self.enable_dance:
            with self._state_lock:
                self._phase = 'done'
            self._speak(self.tts_done_text)
            self.get_logger().info('Dance disabled (enable_dance:=false), round complete')
            return

        if not self._switch_controllers(
                activate=[self.position_adapter_controller, self.position_trajectory_controller],
                deactivate=[self.impedance_trajectory_controller, self.impedance_controller]):
            self._safe_abort()
            return

        with self._state_lock:
            self._phase = 'dancing'
        self._speak(self.tts_dancing_text)
        n_points = len(self.dance_time_from_start)
        dance_points = [
            self.dance_positions[i * n:(i + 1) * n] for i in range(n_points)
        ]
        if not self._send_trajectory(
                self._position_traj_client, dance_points,
                self.dance_time_from_start, wait_result=True):
            self.get_logger().error('Dance trajectory failed')
            self._safe_abort()
            return

        with self._state_lock:
            self._phase = 'done'
        self._speak(self.tts_done_text)
        self.get_logger().info('Ghost game round complete')

    def _ramp_stiffness(self, target_stiffness, damping):
        """Ramp every joint's stiffness toward target_stiffness (same damping
        for all joints throughout) over stiffness_ramp_steps. Updates
        self._stiffness_state as it goes."""
        with self._state_lock:
            start = list(self._stiffness_state)
        for step in range(1, self.stiffness_ramp_steps + 1):
            fraction = step / self.stiffness_ramp_steps
            snapshot = [start[i] + fraction * (target_stiffness[i] - start[i]) for i in range(len(start))]
            self._set_impedance_params(stiffness=snapshot, damping_coefficients=damping)
            time.sleep(self.stiffness_ramp_step_period)
        with self._state_lock:
            self._stiffness_state = list(target_stiffness)

    def _wait_home_or_stall(self):
        """Poll measured position instead of blocking for the whole
        home_time_from_start: returns as soon as every joint is within
        home_position_tolerance of home_positions (arrived), or as soon as any joint
        that isn't there yet has barely moved over the last
        stuck_stall_window seconds (stalled/blocked) - whichever is sooner.
        This is what makes "stuck" detection fast instead of only ever
        finding out after the full nominal move duration elapses.

        Stall checks don't start until stuck_grace_period has elapsed: with
        a soft home_stiffness the arm genuinely lags its own commanded
        trajectory for the first fraction of a second (near-zero real
        movement even with nothing blocking it), which was tripping this as
        a false "stuck" almost immediately."""
        n = len(self.joints)
        move_start = time.monotonic()
        deadline = move_start + self.home_time_from_start + 1.0
        last_check_time = None
        last_positions = None
        while time.monotonic() < deadline:
            time.sleep(self.stuck_check_period)
            positions = self._positions_snapshot()
            if all(pos is not None and
                   abs(pos - self.home_positions[i]) <= self.home_position_tolerance
                   for i, pos in enumerate(positions)):
                return True, positions
            now = time.monotonic()
            if now - move_start < self.stuck_grace_period:
                continue  # still in the trajectory's natural ramp-up, too early to judge
            if last_check_time is None:
                last_check_time, last_positions = now, positions
                continue
            if now - last_check_time >= self.stuck_stall_window:
                stalled = any(
                    positions[i] is not None and last_positions[i] is not None
                    and abs(positions[i] - self.home_positions[i]) >
                    self.home_position_tolerance
                    and abs(positions[i] - last_positions[i]) < self.stuck_stall_movement
                    for i in range(n)
                )
                if stalled:
                    return False, positions
                last_check_time = now
                last_positions = positions
        return False, self._positions_snapshot()

    def _return_home(self):
        """Glide back to the resting pose using the impedance JTC
        (zephyr_arm_impedance_trajectory_controller). The JTC holds its last
        commanded setpoint while idle rather than tracking the live measured
        pose, so if we jumped straight to a "go to home_positions" goal, it
        would interpolate FROM that stale old setpoint - producing a visible
        backswing through wherever the arm used to be. Fix: first send a
        near-instant seed goal at the CURRENT live pose to resync the JTC's
        internal setpoint, then send the real move - see home_seed_time.

        The JTC action itself reports success once the trajectory time
        elapses regardless of whether the goal was actually reached (no goal
        tolerances are configured on zephyr_arm_impedance_trajectory_controller),
        so a blocked/stuck arm still looks "successful" to the action client.
        We independently check the measured position against home_positions;
        if it didn't get there, we back off a bit and release stiffness
        instead of holding at full force or pretending the move worked."""
        n = len(self.joints)
        self._cancel_active_trajectory()
        with self._state_lock:
            self._phase = 'returning_home'
            self._camera_ready = False
            self._mock_solve_requested = False
            self._mock_solve_active = False
        self._interrupt_speech(self.tts_returning_home_text)

        self.get_logger().info('Returning home via impedance JTC')
        if not self._switch_controllers(
                activate=[self.impedance_trajectory_controller, self.impedance_controller],
                deactivate=[self.position_adapter_controller, self.position_trajectory_controller]):
            self._safe_abort()
            return

        # Ramp to a gentle holding stiffness before moving, so joints that
        # were still free (0 stiffness) don't get a torque step.
        self._ramp_stiffness(self.home_stiffness, self.home_damping)

        deadline = time.monotonic() + 3.0
        start_positions = self._positions_snapshot()
        while any(p is None for p in start_positions) and time.monotonic() < deadline:
            time.sleep(0.05)
            start_positions = self._positions_snapshot()
        if any(p is None for p in start_positions):
            self.get_logger().error('Never received full /joint_states, aborting return_home')
            self._safe_abort()
            return

        self._send_trajectory(self._impedance_traj_client, start_positions, [self.home_seed_time])
        time.sleep(self.home_seed_time)
        if not self._send_trajectory(
                self._impedance_traj_client, self.home_positions, [self.home_time_from_start]):
            self.get_logger().error('Failed to send return-home trajectory goal')
            self._safe_abort()
            return

        # Poll instead of blocking for the full home_time_from_start, so a
        # stall is caught within stuck_stall_window seconds, not 4+ seconds.
        arrived, positions = self._wait_home_or_stall()

        if not arrived:
            self.get_logger().warn(
                'Return-home looks stuck/blocked (did not reach home_positions within '
                f'{self.home_position_tolerance} rad) - backing off and releasing stiffness')
            retreat_positions = [
                start_positions[i] if pos is None else pos + self.stuck_retreat_fraction * (start_positions[i] - pos)
                for i, pos in enumerate(positions)
            ]
            self._send_trajectory(
                self._impedance_traj_client, retreat_positions, [self.stuck_retreat_time], wait_result=True)
            self._ramp_stiffness([0.0] * n, self.free_damping)
            with self._state_lock:
                self._phase = 'stuck'
                self._locked = [False] * n
                self._dwell = [0.0] * n
            self._interrupt_speech(self.tts_stuck_text)
            self.get_logger().warn(
                'Backed off and released. Clear the obstruction, then call ~/return_home or ~/start again.')
            return

        self._open_gripper(self.gripper_closed_position)
        time.sleep(self.gripper_settle_time)

        # Fully relax: drop stiffness back to 0, then let gravity_compensation.factor
        # ramp to 0 (same intent as zephyr_bt's ArmRelaxBT - minimum current, no hold).
        self._ramp_stiffness([0.0] * n, self.home_damping)
        self._set_impedance_params(**{'gravity_compensation.factor': 0.0})
        self.get_logger().info('Gravity compensation ramping to 0, arm fully relaxed')

        with self._state_lock:
            self._phase = 'idle'
            self._locked = [False] * n
            self._dwell = [0.0] * n
        self._speak(self.tts_idle_text)
        self.get_logger().info('At home pose, ready for a new round (~/start)')

    def _lock_joint(self, i, positions):
        with self._state_lock:
            self._targets[i] = positions[i]
            self._locked[i] = True
        self.get_logger().info(f'Joint {self.joints[i]} found at {positions[i]:.3f} rad')

        n = len(self.joints)
        for step in range(1, self.stiffness_ramp_steps + 1):
            fraction = step / self.stiffness_ramp_steps
            with self._state_lock:
                self._stiffness_state[i] = fraction * self.locked_stiffness[i]
                stiffness_snapshot = list(self._stiffness_state)
                damping_snapshot = [
                    self.locked_damping[j] if self._locked[j] else self.free_damping[j]
                    for j in range(n)
                ]
            self._set_impedance_params(stiffness=stiffness_snapshot, damping_coefficients=damping_snapshot)
            time.sleep(self.stiffness_ramp_step_period)
        self._speak(self.tts_joint_found_texts[i])

    def _safe_abort(self):
        n = len(self.joints)
        self._cancel_active_trajectory()
        with self._state_lock:
            self._phase = 'aborted'
            self._camera_ready = False
            self._mock_solve_requested = False
            self._mock_solve_active = False
            damping_snapshot = [
                self.locked_damping[j] if self._locked[j] else self.free_damping[j]
                for j in range(n)
            ]
            stiffness_snapshot = list(self._stiffness_state)
        self._interrupt_speech(self.tts_aborted_text)
        self._set_impedance_params(stiffness=stiffness_snapshot, damping_coefficients=damping_snapshot)


def main(args=None):
    rclpy.init(args=args)
    node = GhostGameNode()
    # All long-running game work already runs in a background thread. ROS
    # callbacks only update snapshots or complete async clients, so one
    # blocking executor thread avoids the high idle CPU and scheduling jitter
    # observed with rclpy's MultiThreadedExecutor on this host.
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
