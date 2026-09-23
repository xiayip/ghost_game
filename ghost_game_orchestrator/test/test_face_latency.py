import math
import pytest

from ghost_game_orchestrator.face_tracking import (
    FaceStabilityGate, SmoothFaceServo, face_is_acceptable,
    face_matches_track, make_face_sample,
)


def sample(stamp=10., error=.6, track_id='a', age=0.):
    return make_face_sample(stamp,320*(1+error),240,100,100,640,480,.9,
                            source_age_at_receive=age,source_stamp=stamp-age,track_id=track_id)


def options():
    return dict(reference=[0.]*6,lower_limits=[-1.]*6,upper_limits=[1.]*6,
                yaw_index=4,pitch_index=3,yaw_gain=.8,pitch_gain=.8,max_speed=.25,
                yaw_direction=-1.,pitch_direction=-1.,yaw_max_offset=.5,
                pitch_max_offset=.3,deadband=.08,joint_margin=.05)


def test_old_sensor_frame_does_not_become_fresh_on_arrival():
    old = sample(age=.14)
    assert face_is_acceptable(old,10.005,.15,.005,.65)
    assert not face_is_acceptable(old,10.02,.15,.005,.65)


def test_track_id_switch_rejected_even_if_same_coordinates():
    assert not face_matches_track(sample(track_id='a'),sample(track_id='b'),.25,.75)
    gate = FaceStabilityGate(.2,.08,.35)
    assert not gate.update(sample(track_id='a'),10.)
    assert not gate.update(sample(track_id='b'),10.3)


def test_smoothing_limits_acceleration_and_does_not_move_other_joints():
    smooth = SmoothFaceServo(max_acceleration=1.,max_lead=.06)
    command = [0.]*6
    previous_v = 0.
    for i in range(120):
        observation = sample(stamp=10.+i/60.,error=.8 if i<60 else -.8)
        result,v = smooth.step(observation,command,list(command),1/60.,**options())
        assert abs(v[4]) <= .25+1e-9
        assert abs(v[4]-previous_v) <= 1/60.+1e-9
        assert [result[i] for i in [0,1,2,5]] == [0.]*4
        previous_v,command = v[4],result


def test_reference_does_not_wind_up_ahead_of_stationary_feedback():
    smooth = SmoothFaceServo(max_lead=.06)
    command = [0.]*6
    for i in range(300):
        command,v = smooth.step(sample(stamp=10.+i/60.),command,[0.]*6,1/60.,**options())
    assert command[4] == pytest.approx(-.06)
    assert v[4] == pytest.approx(0.)


def test_filter_is_applied_only_once_per_camera_observation():
    smooth = SmoothFaceServo()
    command = [0.]*6
    smooth.step(sample(error=.5),command,command,1/60.,**options())
    new = sample(stamp=10.04,error=-.5)
    smooth.step(new,command,command,1/60.,**options())
    filtered = list(smooth.filtered)
    for _ in range(3):
        smooth.step(new,command,command,1/60.,**options())
    assert smooth.filtered == filtered


def test_limit_stop_and_invalid_feedback():
    smooth = SmoothFaceServo()
    command = [0.]*6
    command[4] = -.5
    result,v = smooth.step(sample(),command,list(command),1/60.,**options())
    assert result[4] == -.5 and v[4] == 0.
    with pytest.raises(ValueError):
        smooth.step(sample(),command,[math.nan]*6,1/60.,**options())


def test_no_oscillation_for_small_pixel_noise_inside_deadband():
    smooth = SmoothFaceServo()
    command = [0.]*6
    for i in range(120):
        command,v = smooth.step(sample(stamp=10.+i/30.,error=.02*(-1)**i),
                                command,list(command),1/60.,**options())
    assert command == [0.]*6
