import json
from pathlib import Path
import queue
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import String,Bool
from std_srvs.srv import Trigger

from .engine import PiperEngine,Cancelled,play_audio,device_argument
from .effects import cyber_effect,PRESETS
from .worker import SpeechWorker


class GhostTTS(Node):
    def __init__(self):
        super().__init__('ghost_tts')
        defaults = {
            'model_path':str(Path.home()/'.local/share/ghost_tts/zh_CN-huayan-medium.onnx'),
            'preset':'ghost','effect_strength':0.7,'volume':0.7,'length_scale':1.08,
            'audio_device':'pulse','queue_size':16,'max_text_chars':500,
            'max_queue_wait_seconds':30.0,'max_audio_seconds':90.0,
        }
        for key,value in defaults.items():
            self.declare_parameter(key,value)
        self.cfg = {key:self.get_parameter(key).value for key in defaults}
        if self.cfg['preset'] not in PRESETS:
            raise ValueError('Unknown preset')
        if not 0<=self.cfg['volume']<=1 or not 0<=self.cfg['effect_strength']<=1:
            raise ValueError('volume/effect_strength must be between 0 and 1')
        if not 0.5<=self.cfg['length_scale']<=2 or self.cfg['max_audio_seconds']<=0:
            raise ValueError('Invalid synthesis limits')
        # Settings are startup-only, so ros2 param set cannot silently do nothing.
        self.add_on_set_parameters_callback(lambda changes: SetParametersResult(
            successful=not any(p.name in self.cfg for p in changes),
            reason='TTS settings are startup-only; restart with updated YAML'))
        self.events = queue.Queue()
        self.status_pub = self.create_publisher(String,'/ghost/tts/status',20)
        self.busy_pub = self.create_publisher(Bool,'/ghost/tts/busy',QoSProfile(
            depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.speaking_pub = self.create_publisher(Bool,'/ghost/tts/speaking',QoSProfile(
            depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.busy_pub.publish(Bool(data=False))
        self.speaking_pub.publish(Bool(data=False))
        self.worker = SpeechWorker(self.prepare,self.perform,self.emit,
            queue_size=self.cfg['queue_size'],max_chars=self.cfg['max_text_chars'],
            max_wait=self.cfg['max_queue_wait_seconds'])
        self.text_sub = self.create_subscription(String,'/ghost/tts/text',self.on_text,10)
        self.stop_service = self.create_service(Trigger,'/ghost/tts/stop',self.on_stop)
        self.event_timer = self.create_timer(0.05,self.flush_events)
        self.worker.start()
        self.get_logger().info(
            'Offline TTS ready: '
            f'output={self.cfg["audio_device"]}, '
            'publish std_msgs/msg/String on /ghost/tts/text; '
            'stop: /ghost/tts/stop')

    def emit(self,state,job_id,detail):
        self.events.put((state,job_id,detail))

    def prepare(self):
        return PiperEngine(self.cfg['model_path'],self.cfg['length_scale'],self.cfg['max_audio_seconds'])

    def perform(self,engine,job,emit):
        emit('synthesizing',job.job_id,'')
        audio,rate = engine.synthesize(job.text,job.cancel)
        if job.cancel.is_set():
            raise Cancelled()
        emit('processing',job.job_id,'')
        audio = cyber_effect(audio,rate,self.cfg['preset'],self.cfg['effect_strength'],self.cfg['volume'])
        if job.cancel.is_set():
            raise Cancelled()
        emit('playing',job.job_id,'')
        play_audio(audio,rate,job.cancel,device_argument(self.cfg['audio_device']))

    def on_text(self,message):
        self.worker.submit(message.data)

    def on_stop(self,request,response):
        count = self.worker.stop()
        response.success = True
        response.message = f'Cancellation requested for {count} job(s). Active synthesis cancels after current inference chunk.'
        return response

    def flush_events(self):
        for _ in range(100):
            try:
                state,job_id,detail = self.events.get_nowait()
            except queue.Empty:
                break
            event = {'state':state,'job_id':job_id,'detail':detail,
                     'stamp_ns':self.get_clock().now().nanoseconds}
            self.status_pub.publish(String(data=json.dumps(event,ensure_ascii=False)))
            if state in ('synthesizing','processing','playing'):
                self.busy_pub.publish(Bool(data=True))
                self.speaking_pub.publish(Bool(data=state=='playing'))
            elif state in ('done','cancelled','error','ready'):
                # A queued job cancellation must not reset an active playback flag.
                if detail != 'cleared_from_queue':
                    self.busy_pub.publish(Bool(data=False))
                    self.speaking_pub.publish(Bool(data=False))
            if state in ('error','rejected','expired'):
                self.get_logger().warning(f'{state} job={job_id}: {detail}')

    def destroy_node(self):
        self.worker.close()
        if self.context.ok():
            self.busy_pub.publish(Bool(data=False))
            self.speaking_pub.publish(Bool(data=False))
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GhostTTS()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
