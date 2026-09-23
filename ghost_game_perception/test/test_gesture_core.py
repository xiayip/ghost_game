"""Synthetic geometry and event-safety tests; these do not measure model accuracy."""

import json
import math
import unittest

from ghost_game_perception.gesture_core import GestureEngine, HandObservation


def hand(label='Open_Palm', dx=0, dy=0, score=.95, handedness='Right'):
    points = [[.5, .65, 0], [.45, .6, 0], [.4, .55, 0], [.36, .5, 0], [.32, .45, 0]]
    for x in (.44, .48, .52, .56):
        points.extend([[x, .50, 0], [x, .42, 0], [x, .34, 0], [x, .26, 0]])
    return HandObservation([(x+dx, y+dy, z) for x, y, z in points], label, score, handedness)


def pointing(angle=0):
    result = hand(label='None')
    points = [list(p) for p in result.landmarks]
    for start in (9, 13, 17):
        x = points[start][0]
        points[start+1] = [x, .43, 0]
        points[start+2] = [x+.02, .48, 0]
        points[start+3] = [x+.01, .54, 0]
    result.landmarks = [(.5 + (x-.5)*math.cos(angle) - (y-.5)*math.sin(angle),
                         .5 + (x-.5)*math.sin(angle) + (y-.5)*math.cos(angle), z)
                        for x, y, z in points]
    return result


def scaled_palm(scale):
    result = hand()
    center = [sum(result.landmarks[i][axis] for i in (0, 5, 9, 13, 17)) / 5
              for axis in (0, 1)]
    result.landmarks = [(center[0]+(x-center[0])*scale,
                         center[1]+(y-center[1])*scale, z*scale)
                        for x, y, z in result.landmarks]
    return result


def with_world_axis(result, x=.01, y=.01, z=-.04):
    result.world_landmarks = [(0, 0, 0)] * 21
    result.world_landmarks[9] = (x, y, z)
    return result


def curled_unknown():
    result = hand('None')
    points = [list(p) for p in result.landmarks]
    for start in (5, 9, 13, 17):
        x, y, z = points[start]
        points[start + 1] = [x, y - .04, z]
        points[start + 2] = [x + .05, y, z]
        points[start + 3] = [x + .02, y + .04, z]
    result.landmarks = points
    return result


class GestureCoreTests(unittest.TestCase):
    def process(self, engine, obs=None, stamp=0, **kwargs):
        return engine.process([obs or hand()], stamp, stamp, **kwargs)

    def hold(self, engine, obs=None, start=0, count=5):
        return [self.process(engine, obs, start+i*.1) for i in range(count)]

    def test_held_label_emits_once_even_after_cooldown(self):
        results = self.hold(GestureEngine(), count=60)
        self.assertEqual(sum(len(r['events']) for r in results), 1)
        self.assertEqual(results[4]['events'][0]['label'], 'open_palm')

    def test_rearm_requires_release_and_cooldown(self):
        engine = GestureEngine()
        results = self.hold(engine)
        for tick in (.5, .6, .7, .8):
            results.append(engine.process([], tick, tick))
        results += self.hold(engine, start=.9, count=20)
        events = [e for r in results for e in r['events']]
        self.assertEqual(len(events), 2)
        self.assertGreaterEqual(events[1]['stamp'] - events[0]['stamp'], 1.5)
        self.assertNotEqual(events[0]['event_id'], events[1]['event_id'])

    def test_brief_misclassification_does_not_release_latch(self):
        engine = GestureEngine(cooldown_seconds=0)
        results = self.hold(engine)
        results += self.hold(engine, hand('None'), start=.5, count=2)
        results += self.hold(engine, start=.7, count=10)
        self.assertEqual(sum(len(r['events']) for r in results), 1)

    def test_none_category_open_hand_still_drives_palm_control(self):
        engine = GestureEngine()
        results = self.hold(engine, hand('None'), count=5)

        self.assertTrue(all(result['label'] == 'unknown' for result in results))
        self.assertTrue(all(result['palm_open'] for result in results))
        self.assertEqual(results[-1]['palm_source'], 'landmarks')
        self.assertTrue(results[-1]['control']['active'])

    def test_semantic_label_flicker_does_not_rearm_palm_control(self):
        engine = GestureEngine()
        self.hold(engine)

        none_frame = self.process(engine, hand('None'), .5)
        recovered = self.process(engine, hand(), .6)

        self.assertTrue(none_frame['control']['active'])
        self.assertTrue(recovered['control']['active'])
        self.assertNotEqual(none_frame['label'], recovered['label'])

    def test_one_missing_frame_preserves_palm_dwell_and_identity(self):
        engine = GestureEngine()
        before = self.hold(engine)[-1]

        missing = engine.process([], .5, .5)
        recovered = self.process(engine, stamp=.6)

        self.assertFalse(missing['control']['active'])
        self.assertEqual(before['hand_id'], recovered['hand_id'])
        self.assertTrue(recovered['control']['active'])

    def test_hold_restarts_after_missing_observation(self):
        engine = GestureEngine()
        self.hold(engine, count=3)
        engine.process([], .3, .3)
        results = self.hold(engine, start=.4, count=4)
        self.assertFalse(any(r['events'] for r in results))
        self.assertEqual(len(self.process(engine, stamp=.8)['events']), 1)

    def test_palm_neutral_deadzone_motion_and_watchdog(self):
        engine = GestureEngine()
        armed = self.hold(engine)[-1]
        self.assertTrue(armed['control']['active'])
        self.assertEqual(armed['control']['dx'], 0)
        deadzone = self.process(engine, hand(dx=.03), .5)
        self.assertEqual(deadzone['control']['dx'], 0)
        moved = self.process(engine, hand(dx=.12, dy=-.10), .6)
        self.assertGreater(moved['control']['dx'], 0)
        self.assertLess(moved['control']['dy'], 0)
        timeout = engine.process([], .6, 1.0)
        self.assertEqual(timeout['reason'], 'stale_frame')
        self.assertEqual(timeout['control'], {'active': False, 'dx': 0.0, 'dy': 0.0, 'dz': 0.0,
                                             'depth_source': 'none', 'reason': 'stale_frame'})
        self.assertFalse(self.process(engine, stamp=1.1)['control']['active'])

    def test_palm_forward_back_is_apparent_size_and_has_deadband(self):
        engine = GestureEngine()
        armed = self.hold(engine)[-1]
        self.assertEqual(armed['control']['dz'], 0)
        self.assertEqual(armed['control']['depth_source'], 'apparent_size')
        small_change = self.process(engine, scaled_palm(1.04), .5)
        self.assertEqual(small_change['control']['dz'], 0)
        closer = self.process(engine, scaled_palm(1.25), .6)
        farther = self.process(engine, scaled_palm(.8), .7)
        self.assertGreater(closer['control']['dz'], 0)
        self.assertLess(farther['control']['dz'], 0)
        self.assertAlmostEqual(closer['control']['dz'], -farther['control']['dz'])
        self.assertAlmostEqual(closer['control']['dx'], 0)
        self.assertAlmostEqual(closer['control']['dy'], 0)
        saturated = self.process(engine, scaled_palm(1.6), .8)
        self.assertEqual(saturated['control']['dz'], 1)

    def test_palm_depth_neutral_recaptured_after_timeout(self):
        engine = GestureEngine()
        self.hold(engine)
        self.process(engine, scaled_palm(1.25), .5)
        expired = engine.process([], .5, 1.0)
        self.assertEqual(expired['control']['dz'], 0)
        self.assertFalse(expired['control']['active'])
        renewed = self.hold(engine, scaled_palm(1.25), start=1.1)[-1]
        self.assertTrue(renewed['control']['active'])
        self.assertEqual(renewed['control']['dz'], 0)

    def test_no_hand_immediately_disables_control(self):
        engine = GestureEngine()
        self.hold(engine)
        result = engine.process([], .5, .5)
        self.assertEqual(result['reason'], 'no_hand')
        self.assertFalse(result['control']['active'])
        self.assertEqual(result['control']['dx'], 0)

    def test_stale_frames_do_not_rearm_latched_event(self):
        engine = GestureEngine(cooldown_seconds=0)
        self.hold(engine)
        engine.process([], .4, 2)
        results = self.hold(engine, start=2.1, count=6)
        self.assertFalse(any(r['events'] for r in results))

    def test_ambiguous_hands_fail_closed(self):
        engine = GestureEngine()
        self.hold(engine)
        result = engine.process([hand(), hand(dx=.1)], .5, .5)
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'ambiguous_hands')
        self.assertFalse(result['events'])
        self.assertFalse(result['control']['active'])
        self.assertIsNone(result['hand_id'])

    def test_identity_jump_has_explicit_invalid_frame(self):
        engine = GestureEngine()
        previous = self.process(engine)
        jumped = self.process(engine, hand(dx=.4), .1)
        self.assertEqual(jumped['reason'], 'identity_lost')
        acquired = self.process(engine, hand(dx=.4), .2)
        self.assertNotEqual(previous['hand_id'], acquired['hand_id'])
        self.assertFalse(acquired['control']['active'])

    def test_handedness_switch_loses_identity(self):
        engine = GestureEngine()
        self.process(engine)
        self.assertEqual(self.process(engine, hand(handedness='Left'), .1)['reason'], 'identity_lost')

    def test_duplicate_backward_future_and_nan_disable_outputs(self):
        for stamp, now, reason in ((.4, .4, 'out_of_order'), (.3, .4, 'out_of_order'),
                                   (.6, .5, 'future_frame'), (float('nan'), .5, 'invalid_time')):
            engine = GestureEngine()
            self.hold(engine)
            result = engine.process([hand()], stamp, now)
            self.assertEqual(result['reason'], reason)
            self.assertFalse(result['control']['active'])
            self.assertFalse(result['events'])
            json.dumps(result, allow_nan=False)

    def test_zero_timestamp_is_valid_only_once(self):
        engine = GestureEngine()
        self.assertTrue(self.process(engine)['valid'])
        self.assertEqual(self.process(engine)['reason'], 'out_of_order')

    def test_invalid_numeric_landmarks_scores_and_degenerate_hands(self):
        cases = []
        bad = hand()
        bad.landmarks[8] = (float('inf'), .3, 0)
        cases.append((bad, 'invalid_observation'))
        cases.append((hand(score=float('nan')), 'invalid_observation'))
        cases.append((HandObservation([(0, 0, 0)]*21, 'Open_Palm', .9), 'degenerate_hand'))
        for obs, reason in cases:
            result = self.process(GestureEngine(), obs)
            self.assertEqual(result['reason'], reason)
            self.assertFalse(result['events'])

    def test_builtin_and_custom_labels(self):
        mappings = {'Open_Palm': 'open_palm', 'Closed_Fist': 'closed_fist',
                    'Pointing_Up': 'pointing', 'Thumb_Up': 'thumb_up',
                    'Thumb_Down': 'thumb_down', 'Victory': 'victory', 'ILoveYou': 'ilove_you',
                    'handshake_offer': 'handshake_offer', 'fist_bump_offer': 'fist_bump_offer'}
        for category, expected in mappings.items():
            result = self.process(GestureEngine(), hand(category))
            self.assertEqual(result['label'], expected)
            self.assertEqual(result['source'], 'classifier')

    def test_unknown_and_low_confidence_never_trigger(self):
        low_confidence_fist = curled_unknown()
        low_confidence_fist.label = 'Closed_Fist'
        low_confidence_fist.score = .3
        for obs in (curled_unknown(), low_confidence_fist):
            results = self.hold(GestureEngine(), obs, count=30)
            self.assertTrue(all(r['label'] == 'unknown' for r in results))
            self.assertFalse(any(r['events'] or r['control']['active'] for r in results))

    def test_pointing_geometry_is_rotation_independent(self):
        for angle in (0, math.pi/2, math.pi, 3*math.pi/2):
            result = self.process(GestureEngine(), pointing(angle), frame_size=(640, 640))
            self.assertEqual(result['label'], 'pointing')
            self.assertEqual(result['source'], 'heuristic')
            self.assertAlmostEqual(math.hypot(*result['pointing']), 1)

    def test_handshake_heuristic_is_marked(self):
        obs = hand()
        # Compress palm width then rotate upright hand to horizontal.
        obs.landmarks = [(.5-(y-.5), .5+(x-.5)*.2, z) for x, y, z in obs.landmarks]
        result = self.process(GestureEngine(), obs, frame_size=(640, 640))
        self.assertEqual(result['label'], 'handshake_offer')
        self.assertEqual(result['source'], 'heuristic')

    def test_fist_bump_requires_world_orientation_evidence(self):
        obs = hand('Closed_Fist')
        ordinary = self.process(GestureEngine(), obs)
        self.assertEqual(ordinary['label'], 'closed_fist')
        with_world_axis(obs, .001, .001, -.04)
        result = self.process(GestureEngine(), obs)
        self.assertEqual(result['label'], 'fist_bump_offer')
        self.assertEqual(result['source'], 'heuristic')

    def test_perspective_none_category_can_be_handshake_or_fist_bump(self):
        handshake_obs = with_world_axis(hand('None', score=.2))
        handshake = self.process(GestureEngine(), handshake_obs)
        self.assertEqual(handshake['label'], 'handshake_offer')
        self.assertEqual(handshake['source'], 'heuristic')
        self.assertEqual(handshake['score'], .70)
        self.assertEqual(handshake['raw_score'], .2)
        fist_obs = curled_unknown()
        fist_obs.score = .2
        fist_bump = self.process(GestureEngine(), with_world_axis(fist_obs))
        self.assertEqual(fist_bump['label'], 'fist_bump_offer')
        self.assertEqual(fist_bump['source'], 'heuristic')
        self.assertEqual(fist_bump['score'], .70)

    def test_perspective_candidates_require_camera_facing_margin(self):
        side_on = self.process(GestureEngine(), with_world_axis(hand('None'), .04, 0, -.01))
        self.assertEqual(side_on['label'], 'unknown')

    def test_wave_requires_reversals_and_not_small_jitter(self):
        engine = GestureEngine(hold_seconds=.2, cooldown_seconds=0)
        offsets = [0, .06, .12, .06, 0, -.06, -.12, -.06, 0, .06, .12] + [.12]*6
        results = [self.process(engine, hand(dx=dx), i*.08) for i, dx in enumerate(offsets)]
        self.assertTrue(any(r['label'] == 'wave' for r in results))
        self.assertTrue(any(e['label'] == 'wave' for r in results for e in r['events']))
        self.assertTrue(all(r['palm_open'] for r in results if r['label'] == 'wave'))
        self.assertTrue(any(r['control']['active'] for r in results if r['label'] == 'wave'))
        jitter = GestureEngine()
        results = [self.process(jitter, hand(dx=.015*math.sin(i)), i*.05) for i in range(40)]
        self.assertTrue(all(r['label'] != 'wave' for r in results))

    def test_monotonic_palm_motion_is_not_wave(self):
        engine = GestureEngine()
        results = [self.process(engine, hand(dx=i*.006), i*.05) for i in range(30)]
        self.assertTrue(all(r['label'] == 'open_palm' for r in results))

    def test_center_is_normalized_and_pixels_separate(self):
        result = self.process(GestureEngine())
        self.assertTrue(all(0 <= v <= 1 for v in result['center']))
        self.assertEqual(result['center_px'], [result['center'][0]*640, result['center'][1]*480])

    def test_reset_and_new_engine_keep_event_ids_distinct(self):
        engine = GestureEngine()
        first = self.hold(engine)[-1]['events'][0]['event_id']
        engine.reset()
        second = self.hold(engine)[-1]['events'][0]['event_id']
        third = self.hold(GestureEngine())[-1]['events'][0]['event_id']
        self.assertEqual(len({first, second, third}), 3)


if __name__ == '__main__':
    unittest.main()
