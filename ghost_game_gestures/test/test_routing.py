import math

from ghost_game_gestures.routing import EventGate


def event(n=1, stamp=10., **changes):
    return dict(dict(event_id=f'event-{n}', label='handshake_offer', score=.9,
                     stamp=stamp, hand_id='hand-1', source='heuristic'), **changes)


def test_repeated_event_cannot_replay_after_cooldown():
    gate = EventGate()
    assert gate.evaluate(event(), 10.1, 20.)['accepted']
    assert gate.evaluate(event(), 10.1, 30.)['reason'] == 'duplicate'


def test_stale_future_bad_score_and_unknown_labels_never_start():
    for changes, now in [({}, 10.4), ({}, 9.), ({'score': math.nan}, 10.),
                         ({'score': .1}, 10.), ({'label': 'anything'}, 10.)]:
        assert not EventGate().evaluate(event(**changes), now, 20.)['accepted']


def test_busy_event_is_dropped_not_queued_or_replayed():
    gate = EventGate()
    assert gate.evaluate(event(), 10., 20., busy=True)['reason'] == 'busy'
    assert gate.evaluate(event(), 10., 20.1)['reason'] == 'duplicate'
    assert gate.evaluate(event(2, 10.2), 10.2, 20.2)['accepted']


def test_global_cooldown_prevents_gesture_alternation_spam():
    gate = EventGate()
    assert gate.evaluate(event(), 10., 20.)['accepted']
    assert gate.evaluate(event(2, 10.1, label='pointing'), 10.1, 20.1)['reason'] == 'cooldown'
    assert gate.evaluate(event(3, 11.6), 11.6, 21.6)['accepted']


def test_closed_fist_has_a_default_action_route():
    result = EventGate().evaluate(event(label='closed_fist'), 10., 20.)
    assert result['accepted']
    assert result['action'] == 'fist'


def test_out_of_order_and_bounded_history():
    gate = EventGate(cooldown=0, remembered=2)
    for n in range(4):
        assert gate.evaluate(event(n, 10.+n), 10.+n, 20.+n)['accepted']
    assert len(gate.seen) == 2
    # Evicted IDs still cannot replay a source timestamp already dispatched.
    assert gate.evaluate(event(0, 10.), 10., 40.)['reason'] == 'out_of_order'


def test_invalid_payloads_do_not_throw():
    for payload in [[], None, {}, {'event_id': []}, event(score=True), event(stamp=-1)]:
        assert not EventGate().evaluate(payload, 10., 20.)['accepted']
