"""Validate bounded, fresh gesture events before requesting an interaction."""
from collections import OrderedDict
import math


DEFAULT_ROUTES = {
    'open_palm': 'palm_follow', 'handshake_offer': 'handshake',
    'closed_fist': 'fist', 'fist_bump_offer': 'fist_bump', 'pointing': 'point_at',
    'wave': 'wave', 'thumb_up': 'acknowledge',
}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def string_map(value):
    if not isinstance(value, dict) or any(
        not isinstance(k, str) or not k or not isinstance(v, str) or not v
        for k, v in value.items()
    ):
        raise ValueError('mapping must be a JSON object of nonempty strings')
    return dict(value)


class EventGate:
    """No queued commands. Remember even rejected event IDs to prevent replay."""
    def __init__(self, routes=None, max_age=.25, min_score=.65,
                 cooldown=1.5, remembered=512):
        if (not all(_finite(v) for v in (max_age, min_score, cooldown)) or
                max_age <= 0 or not 0 < min_score <= 1 or cooldown < 0 or
                not isinstance(remembered, int) or remembered < 1):
            raise ValueError('invalid freshness, score, cooldown or history limits')
        self.routes = string_map(DEFAULT_ROUTES if routes is None else routes)
        self.max_age, self.min_score, self.cooldown = max_age, min_score, cooldown
        self.remembered = remembered
        self.seen = OrderedDict()
        self.last_stamp = None
        self.last_dispatch = -math.inf

    def evaluate(self, event, now, steady_now, busy=False):
        if not isinstance(event, dict) or not _finite(now) or not _finite(steady_now):
            return {'accepted': False, 'reason': 'invalid_event'}
        event_id, label = event.get('event_id'), event.get('label')
        stamp, score = event.get('stamp'), event.get('score')
        hand_id = event.get('hand_id')
        if (not isinstance(event_id, str) or not 0 < len(event_id) <= 160 or
                not isinstance(label, str) or not 0 < len(label) <= 80 or
                not isinstance(hand_id, (str, int)) or isinstance(hand_id, bool) or
                not str(hand_id) or not _finite(stamp) or stamp < 0 or
                not _finite(score) or not 0 <= score <= 1):
            return {'accepted': False, 'reason': 'invalid_event'}
        result = {'accepted': False, 'event_id': event_id, 'label': label,
                  'stamp': stamp, 'hand_id': hand_id, 'score': score,
                  'action': self.routes.get(label), 'source': event.get('source', 'unspecified')}
        if event_id in self.seen:
            return dict(result, reason='duplicate')
        self.seen[event_id] = None
        if len(self.seen) > self.remembered:
            self.seen.popitem(last=False)
        age = now-stamp
        if not -.03 <= age <= self.max_age:
            return dict(result, reason='stale_or_future')
        if self.last_stamp is not None and stamp <= self.last_stamp:
            return dict(result, reason='out_of_order')
        self.last_stamp = stamp
        if score < self.min_score:
            return dict(result, reason='low_score')
        if label not in self.routes:
            return dict(result, reason='unmapped_label')
        if busy:
            return dict(result, reason='busy')
        if steady_now-self.last_dispatch < self.cooldown:
            return dict(result, reason='cooldown')
        # Cooldown uses monotonic time; changing the ROS clock cannot bypass it.
        self.last_dispatch = steady_now
        return dict(result, accepted=True, reason='ready')
