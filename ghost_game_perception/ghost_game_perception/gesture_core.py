"""Gesture intent, temporal debouncing and image-plane palm control.

This module has no ROS, camera or inference dependencies. Coordinates are
normalized image coordinates (including ``center``); ``center_px`` is pixels.
``control`` contains dimensionless displacement hints, never robot velocities.
Its dz is positive when the apparent palm grows relative to its neutral size
(toward the camera), negative when it shrinks. It is not metric depth: changing
the hand angle also changes apparent size. dz uses an 8% ratio deadband and
saturates at a size ratio of 1.5 or its reciprocal.
Heuristic labels express a proposed interaction, not verified contact or pose.

All timestamps must use the same clock. Zero is valid at simulated-clock start;
negative, future, duplicate and backward timestamps are rejected. Call process
on a watchdog tick with the last image stamp to expire old control outputs.
"""

from collections import deque
from dataclasses import dataclass
import math
from uuid import uuid4


@dataclass
class HandObservation:
    landmarks: object
    label: str
    score: float
    handedness: str = ''
    world_landmarks: object = None


_LABELS = {
    'Open_Palm': 'open_palm', 'Closed_Fist': 'closed_fist',
    'Pointing_Up': 'pointing', 'Thumb_Up': 'thumb_up',
    'Thumb_Down': 'thumb_down', 'Victory': 'victory', 'ILoveYou': 'ilove_you',
    'handshake_offer': 'handshake_offer', 'fist_bump_offer': 'fist_bump_offer',
}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _angle(a, b, c):
    u = [x - y for x, y in zip(a, b)]
    v = [x - y for x, y in zip(c, b)]
    denominator = math.sqrt(sum(x*x for x in u) * sum(x*x for x in v))
    if denominator < 1e-10:
        return 0.0
    cosine = sum(x*y for x, y in zip(u, v)) / denominator
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _points_valid(points):
    try:
        return len(points) == 21 and all(
            len(point) == 3 and all(_finite(v) for v in point) for point in points)
    except (TypeError, AttributeError):
        return False


class GestureEngine:
    """Classify one tracked hand; ambiguous hands fail closed.

    A stable label emits once, and must be released before emitting again.
    Missing/stale frames disable palm control immediately. Stale frames alone
    do not count as a witnessed release and cannot rearm a latched event.
    """

    def __init__(self, threshold=.65, hold_seconds=.35, release_seconds=.25,
                 cooldown_seconds=1.5, max_gap=.25, move_deadzone=.06, move_scale=.25):
        values = (threshold, hold_seconds, release_seconds, cooldown_seconds,
                  max_gap, move_deadzone, move_scale)
        if not all(_finite(v) for v in values):
            raise ValueError('gesture settings must be finite numbers')
        if not 0 <= threshold <= 1 or min(hold_seconds, release_seconds, cooldown_seconds) < 0:
            raise ValueError('invalid score threshold or duration')
        if max_gap <= 0 or move_scale <= move_deadzone or move_deadzone < 0:
            raise ValueError('require max_gap > 0 and move_scale > move_deadzone >= 0')
        self.threshold = threshold
        self.hold_seconds = hold_seconds
        self.release_seconds = release_seconds
        self.cooldown_seconds = cooldown_seconds
        self.max_gap = max_gap
        self.move_deadzone = move_deadzone
        self.move_scale = move_scale
        self._prefix = uuid4().hex
        self._event_count = 0
        self._hand_count = 0
        self.reset()

    def reset(self):
        """Explicitly forget tracking and latches, retaining unique event IDs."""
        self._last_stamp = None
        self._track = None
        self._candidate = ''
        self._candidate_since = None
        self._latched = ''
        self._release_since = None
        self._next_event_after = -math.inf
        self._neutral = None
        self._neutral_span = None
        self._palm_candidate_since = None
        self._last_hand_stamp = None
        self._wave = deque()
        self._wave_until = -math.inf
        self._wave_turn = -math.inf

    def _suspend(self, forget_track=False):
        self._candidate = ''
        self._candidate_since = None
        self._neutral = None
        self._neutral_span = None
        self._palm_candidate_since = None
        self._wave.clear()
        self._wave_until = -math.inf
        if forget_track:
            self._track = None
            self._last_hand_stamp = None

    def _release(self, stamp):
        if not self._latched:
            return
        if self._release_since is None:
            self._release_since = stamp
        if stamp - self._release_since + 1e-9 >= self.release_seconds:
            self._latched = ''
            self._release_since = None

    @staticmethod
    def _empty(stamp, reason):
        return {
            'valid': False, 'reason': reason,
            'stamp': float(stamp) if _finite(stamp) else 0.0,
            'hand_id': None, 'label': 'unknown', 'score': 0.0, 'source': 'none',
            'palm_open': False, 'palm_source': 'none',
            'center': None, 'center_px': None, 'pointing': None, 'events': [],
            'control': {'active': False, 'dx': 0.0, 'dy': 0.0, 'dz': 0.0,
                        'depth_source': 'none', 'reason': reason},
        }

    def _invalid(self, stamp, reason, forget_track=False):
        self._suspend(forget_track)
        if reason != 'no_hand':
            self._release_since = None
        return self._empty(stamp, reason)

    @staticmethod
    def _geometry(hand, frame_size):
        aspect = frame_size[0] / frame_size[1]
        # MediaPipe z uses approximately the x-coordinate scale.
        points = [(p[0] * aspect, p[1], p[2] * aspect) for p in hand.landmarks]
        center = [sum(hand.landmarks[i][axis] for i in (0, 5, 9, 13, 17)) / 5
                  for axis in (0, 1)]
        span = max(_distance(points[0][:2], points[9][:2]),
                   _distance(points[5][:2], points[17][:2]))
        return points, center, span

    @staticmethod
    def _extended_fingers(points):
        def extended(start):
            a, b, c, d = points[start:start+4]
            return (_angle(a, b, c) >= 155 and _angle(b, c, d) >= 145
                    and _distance(d, points[0]) > 1.12 * _distance(b, points[0]))
        return [extended(start) for start in (5, 9, 13, 17)]

    @classmethod
    def _is_pointing(cls, points):
        extended = cls._extended_fingers(points)
        return extended[0] and not any(extended[1:])

    @staticmethod
    def _is_open_palm_geometry(points):
        """Accept a broad open hand even when the model category flickers.

        MediaPipe commonly reports ``None`` while a palm rotates or moves in
        depth.  Three reasonably straight non-thumb fingers are sufficient
        for continuous palm control; semantic gesture events still use the
        recognizer's stricter label path below.
        """
        extended = 0
        for start in (5, 9, 13, 17):
            a, b, c, d = points[start:start + 4]
            if (_angle(a, b, c) >= 135 and _angle(b, c, d) >= 120 and
                    _distance(d, points[0]) >
                    1.02 * _distance(b, points[0])):
                extended += 1
        return extended >= 3

    def _classify(self, hand, points):
        label = _LABELS.get(hand.label, 'unknown')
        if label == 'unknown' and self._is_pointing(points):
            return 'pointing', 'heuristic', .70
        extended_count = sum(self._extended_fingers(points))
        toward_camera = False
        if hand.world_landmarks is not None:
            w = hand.world_landmarks
            axis = [w[9][i] - w[0][i] for i in range(3)]
            # Calibrated from the local end-effector-camera view. A negative
            # world-z longitudinal axis means the offered hand points toward
            # the camera. Keep a transverse margin to reject frontal palms.
            toward_camera = (axis[2] < -.025
                             and -axis[2] > .7 * math.hypot(axis[0], axis[1]))
        if toward_camera:
            # Edge-on offered hands are commonly assigned MediaPipe's None
            # category; finger geometry separates them from an offered fist.
            if label in ('open_palm', 'unknown') and extended_count >= 3:
                return 'handshake_offer', 'heuristic', .70
            if label == 'closed_fist' or (label == 'unknown' and extended_count == 0):
                return 'fist_bump_offer', 'heuristic', .70
        if hand.score < self.threshold:
            return 'unknown', 'low_confidence', 0.0
        if label == 'open_palm':
            axis = [points[9][i] - points[0][i] for i in (0, 1)]
            length = math.hypot(*axis)
            width = _distance(points[5][:2], points[17][:2])
            # An edge-on open hand extended horizontally is a handshake offer
            # candidate. RGB cannot establish actual handshake intent/contact.
            if length > 1e-4 and width < .45 * length and abs(axis[0]) > 1.5 * abs(axis[1]):
                return 'handshake_offer', 'heuristic', .70
        return (label, 'classifier' if label != 'unknown' else 'unsupported',
                float(hand.score) if label != 'unknown' else 0.0)

    def _is_wave(self, center, stamp, span):
        self._wave.append((stamp, center[0], center[1]))
        while self._wave and stamp - self._wave[0][0] > 1.2:
            self._wave.popleft()
        points = list(self._wave)
        amplitude = max(.10, .55 * span)
        if len(points) >= 5 and points[-1][0] - points[0][0] >= .25:
            vertical = max(p[2] for p in points) - min(p[2] for p in points)
            if vertical < max(.08, .75 * span):
                direction = 0
                extreme = points[0][1]
                reversals = 0
                last_turn = -math.inf
                for tick, x, _ in points[1:]:
                    if direction == 0:
                        if abs(x - extreme) >= amplitude:
                            direction = 1 if x > extreme else -1
                            extreme = x
                    elif (x - extreme) * direction >= 0:
                        extreme = x
                    elif (extreme - x) * direction >= amplitude:
                        reversals += 1
                        direction *= -1
                        extreme = x
                        last_turn = tick
                if reversals >= 2 and last_turn > self._wave_turn:
                    self._wave_turn = last_turn
                    self._wave_until = stamp + max(.45, self.hold_seconds + .1)
        return stamp <= self._wave_until

    def process(self, hands, stamp, now, frame_size=(640, 480), max_age=.20):
        """Return a JSON-safe snapshot. Only fresh, ordered observations act."""
        if not all(_finite(v) for v in (stamp, now, max_age)) or min(stamp, now) < 0 or max_age <= 0:
            return self._invalid(stamp, 'invalid_time')
        if stamp > now:
            return self._invalid(stamp, 'future_frame')
        # Age precedes duplicate detection so watchdog calls report expiration.
        if now - stamp > max_age:
            return self._invalid(stamp, 'stale_frame')
        if self._last_stamp is not None and stamp <= self._last_stamp:
            return self._invalid(stamp, 'out_of_order')
        if (not isinstance(frame_size, (list, tuple)) or len(frame_size) != 2
                or not all(_finite(v) and v > 0 for v in frame_size)):
            return self._invalid(stamp, 'invalid_frame_size')
        if self._last_stamp is not None and stamp - self._last_stamp > self.max_gap:
            self._suspend(forget_track=True)
            self._release_since = None
        self._last_stamp = stamp
        if not isinstance(hands, (list, tuple)):
            return self._invalid(stamp, 'invalid_observation', True)
        if len(hands) != 1:
            if not hands:
                self._release(stamp)
                # Semantic events still require uninterrupted dwell. Palm
                # control has its own short continuity window below.
                self._candidate = ''
                self._candidate_since = None
                # Preserve identity and the palm dwell timer across an
                # isolated detector miss. A sustained loss still clears all
                # state after max_gap, and this frame itself remains inactive.
                if (self._last_hand_stamp is not None and
                        stamp - self._last_hand_stamp <= self.max_gap):
                    return self._empty(stamp, 'no_hand')
            return self._invalid(
                stamp, 'no_hand' if not hands else 'ambiguous_hands', True)
        hand = hands[0]
        if (not isinstance(hand, HandObservation) or not _points_valid(hand.landmarks)
                or not _finite(hand.score) or not 0 <= hand.score <= 1
                or not isinstance(hand.label, str) or not isinstance(hand.handedness, str)
                or (hand.world_landmarks is not None and not _points_valid(hand.world_landmarks))):
            return self._invalid(stamp, 'invalid_observation', True)
        if any(not -.25 <= p[0] <= 1.25 or not -.25 <= p[1] <= 1.25 or abs(p[2]) > 5
               for p in hand.landmarks):
            return self._invalid(stamp, 'invalid_observation', True)
        points, center, span = self._geometry(hand, frame_size)
        if span < 1e-4:
            return self._invalid(stamp, 'degenerate_hand', True)
        if (self._last_hand_stamp is not None and
                stamp - self._last_hand_stamp > self.max_gap):
            self._suspend(forget_track=True)
        if self._track is not None:
            previous, previous_span, handedness, hand_id = self._track
            jump = math.hypot((center[0] - previous[0]) * frame_size[0] / frame_size[1],
                              center[1] - previous[1])
            if ((handedness and hand.handedness and handedness != hand.handedness)
                    or jump > max(.12, min(.35, previous_span * 2.0))
                    or not .4 <= span / previous_span <= 2.5):
                return self._invalid(stamp, 'identity_lost', True)
        else:
            self._hand_count += 1
            hand_id = self._hand_count
        self._track = (center, span, hand.handedness, hand_id)
        self._last_hand_stamp = stamp
        label, source, score = self._classify(hand, points)
        palm_open = (
            label in ('open_palm', 'wave') or
            self._is_open_palm_geometry(points))
        palm_source = (
            'classifier' if label == 'open_palm'
            else 'landmarks' if palm_open
            else 'none')
        if label == 'open_palm':
            if self._is_wave(center, stamp, span):
                label, source = 'wave', 'temporal_heuristic'
        else:
            self._wave.clear()
            self._wave_until = -math.inf
        if label == self._latched:
            self._release_since = None
        else:
            self._release(stamp)
        if label != self._candidate:
            self._candidate = label
            self._candidate_since = stamp
        # A suspended stream can resume the same unknown label with no candidate.
        if self._candidate_since is None:
            self._candidate_since = stamp
        stable = label != 'unknown' and stamp - self._candidate_since + 1e-9 >= self.hold_seconds
        if palm_open:
            if self._palm_candidate_since is None:
                self._palm_candidate_since = stamp
                self._neutral = None
                self._neutral_span = None
        else:
            self._palm_candidate_since = None
            self._neutral = None
            self._neutral_span = None
        palm_stable = (
            palm_open and self._palm_candidate_since is not None and
            stamp - self._palm_candidate_since + 1e-9 >= self.hold_seconds)
        events = []
        if stable and not self._latched and stamp + 1e-9 >= self._next_event_after:
            self._event_count += 1
            events.append({'event_id': f'{self._prefix}:{self._event_count}', 'label': label,
                           'stamp': float(stamp), 'hand_id': hand_id, 'score': score,
                           'source': source})
            self._latched = label
            self._release_since = None
            self._next_event_after = stamp + self.cooldown_seconds
        control = {'active': False, 'dx': 0.0, 'dy': 0.0, 'dz': 0.0,
                   'depth_source': 'none', 'reason': 'not_open_palm'}
        if palm_open:
            control['reason'] = 'arming'
            if palm_stable:
                if self._neutral is None:
                    self._neutral = list(center)
                    self._neutral_span = span
                def displacement(value):
                    magnitude = max(0.0, abs(value) - self.move_deadzone)
                    return math.copysign(min(1.0, magnitude / (self.move_scale - self.move_deadzone)), value)
                control = {'active': True,
                           'dx': displacement(center[0] - self._neutral[0]),
                           'dy': displacement(center[1] - self._neutral[1]),
                           'dz': 0.0, 'depth_source': 'apparent_size', 'reason': 'open_palm'}
                # A log ratio treats 1.25x and 0.8x apparent size symmetrically.
                size_change = math.log(span / self._neutral_span)
                deadband, saturation = math.log(1.08), math.log(1.5)
                magnitude = max(0.0, abs(size_change) - deadband)
                control['dz'] = math.copysign(min(1.0, magnitude / (saturation - deadband)), size_change)
        pointing = None
        if label == 'pointing':
            dx = points[8][0] - points[5][0]
            dy = points[8][1] - points[5][1]
            norm = math.hypot(dx, dy)
            if norm > 1e-6:
                pointing = [dx / norm, dy / norm]
        return {'valid': True, 'reason': 'ok', 'stamp': float(stamp), 'hand_id': hand_id,
                'label': label, 'score': score, 'source': source,
                'palm_open': palm_open, 'palm_source': palm_source,
                'raw_label': hand.label, 'raw_score': float(hand.score),
                'center': center, 'center_px': [center[0] * frame_size[0], center[1] * frame_size[1]],
                'pointing': pointing, 'events': events, 'control': control}
