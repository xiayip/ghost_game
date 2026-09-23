"""Execute actual callback body without ROS, exercising source-time rejection."""
import ast
from pathlib import Path
from types import SimpleNamespace
import math
import threading
import time

from ghost_game_orchestrator.face_tracking import make_face_sample
from ghost_game_orchestrator.face_tracking import SmoothFaceServo


def node():
    source = Path(__file__).parents[1]/'ghost_game_orchestrator/ghost_game_node.py'
    module = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='GhostGameNode')
    method = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_face_detection_cb')
    method.args.args[1].annotation = None
    namespace = dict(time=time,math=math,make_face_sample=make_face_sample)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
    class Adapter:
        _face_detection_cb = namespace['_face_detection_cb']
    result=Adapter()
    result._face_lock=threading.Lock()
    result._face_last_clock=result._face_last_stamp=0
    result._camera_size=(640,480)
    result._camera_frame='camera'
    result._latest_face=None
    result.face_detection_max_age=.15
    result.get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=100_000_000_000))
    return result


def message(stamp, empty=False, frame='camera'):
    header=SimpleNamespace(stamp=SimpleNamespace(sec=int(stamp),nanosec=round((stamp-int(stamp))*1e9)),frame_id=frame)
    detection=SimpleNamespace(id='track_1',results=[],bbox=SimpleNamespace(size_x=100.,size_y=100.,
        center=SimpleNamespace(position=SimpleNamespace(x=320.,y=240.))))
    return SimpleNamespace(header=header,detections=[] if empty else [detection])


def test_actual_callback_preserves_age_and_rejects_stale_and_wrong_frame():
    adapter=node()
    adapter._face_detection_cb(message(99.))
    assert adapter._latest_face is None
    adapter._face_detection_cb(message(99.90))
    assert abs(adapter._latest_face.source_age_at_receive-.1)<1e-6
    adapter._face_detection_cb(message(99.91,frame='other_camera'))
    assert adapter._latest_face is None


def test_out_of_order_message_cannot_replace_newer_face():
    adapter=node()
    adapter._face_detection_cb(message(99.95))
    newest=adapter._latest_face
    adapter._face_detection_cb(message(99.90,empty=True))
    assert adapter._latest_face is newest
    adapter._face_detection_cb(message(99.98,empty=True))
    assert adapter._latest_face is None


def test_actual_joint_callback_does_not_refresh_stale_or_out_of_order_feedback():
    path=Path(__file__).parents[1]/'ghost_game_orchestrator/ghost_game_node.py'
    cls=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef))
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_joint_state_cb','_face_positions_snapshot')]
    for method in methods:
        for arg in method.args.args: arg.annotation=None
    namespace=dict(time=time,math=math)
    exec(compile(ast.Module(body=methods,type_ignores=[]),str(path),'exec'),namespace)
    adapter=SimpleNamespace(joints=['joint4'],_joint_positions_lock=threading.Lock(),
        _joint_positions={},_joint_received={},_joint_stamp={},_joint_last_clock=0,
        face_joint_feedback_timeout=.2,
        get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=100_000_000_000)))
    old=message(99.)
    old.name,old.position=['joint4'],[.3]
    namespace['_joint_state_cb'](adapter,old)
    assert namespace['_face_positions_snapshot'](adapter)==[None]
    fresh=message(99.98)
    fresh.name,fresh.position=['joint4'],[.4]
    namespace['_joint_state_cb'](adapter,fresh)
    namespace['_joint_state_cb'](adapter,old)
    assert namespace['_face_positions_snapshot'](adapter)==[.4]


def test_centering_abort_clears_velocity_feedforward():
    path=Path(__file__).parents[1]/'ghost_game_orchestrator/ghost_game_node.py'
    cls=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef))
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_center_face')
    namespace=dict(time=time,SmoothFaceServo=SmoothFaceServo,rclpy=SimpleNamespace(ok=lambda:True))
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
    recorded=[]
    adapter=SimpleNamespace(face_center_timeout=20.,_face_positions_snapshot=lambda:[0.]*6,
        face_servo_rate_hz=60.,face_filter_cutoff_hz=6.,face_servo_acceleration=1.,face_max_command_lead=.06,
        _state_lock=threading.Lock(),face_continuous_follow=True,_face_control_request=lambda:'abort',
        _publish_impedance_command=lambda positions,velocities=None:recorded.append((positions,velocities)))
    result=namespace['_center_face'](adapter,[0.]*6,SimpleNamespace(track_id='a'))
    assert result=='abort'
    assert recorded==[([0.]*6,None)]  # Existing publisher maps None to all-zero velocities.
