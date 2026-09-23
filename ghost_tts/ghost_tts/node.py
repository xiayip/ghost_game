import json
from pathlib import Path
import queue
import threading
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import String,Bool
from std_srvs.srv import Trigger
from ghost_game_interfaces.action import Speak

from .engine import (
    PiperEngine,DoubaoEngine,Cancelled,play_audio,device_argument,
    resolve_tts_backend)
from .effects import cyber_effect,PRESETS
from .worker import SpeechWorker,clean_text


class GhostTTS(Node):
    def __init__(self):
        super().__init__('ghost_tts')
        defaults = {
            'backend':'auto',
            'model_path':str(Path.home()/'.local/share/ghost_tts/zh_CN-huayan-medium.onnx'),
            'preset':'ghost','effect_strength':0.7,'volume':0.7,'length_scale':1.08,
            'apply_effects':True,
            'audio_device':'pulse','queue_size':16,'max_text_chars':500,
            'max_queue_wait_seconds':30.0,'max_audio_seconds':90.0,
            'doubao_endpoint':DoubaoEngine.DEFAULT_ENDPOINT,
            'doubao_model':'seed-tts-2.0-standard',
            'doubao_sample_rate':24000,
            'doubao_speech_rate':0,
            'doubao_loudness_rate':0,
            'doubao_timeout_seconds':60.0,
            'action_name':'/ghost/tts/speak',
            'caption_topic':'/ghost/tts/caption',
        }
        for key,value in defaults.items():
            self.declare_parameter(key,value)
        self.cfg = {key:self.get_parameter(key).value for key in defaults}
        self.requested_backend = str(self.cfg['backend']).strip().lower()
        self.cfg['backend'] = resolve_tts_backend(self.requested_backend)
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
        self.action_condition = threading.Condition()
        self.action_states = {}
        self.action_goal_to_job = {}
        self.action_job_to_goal = {}
        self.status_pub = self.create_publisher(String,'/ghost/tts/status',20)
        self.caption_pub = self.create_publisher(
            String,self.cfg['caption_topic'],QoSProfile(
                depth=1,reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL))
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
        self.action_server = ActionServer(
            self,Speak,self.cfg['action_name'],self.execute_speech,
            goal_callback=self.on_action_goal,
            cancel_callback=self.on_action_cancel,
            callback_group=ReentrantCallbackGroup())
        self.event_timer = self.create_timer(0.05,self.flush_events)
        self.worker.start()
        backend_text = self.cfg['backend']
        if self.requested_backend == 'auto':
            backend_text += ' (auto)'
        self.get_logger().info(
            f'TTS started: backend={backend_text}, '
            f'output={self.cfg["audio_device"]}, '
            'publish std_msgs/msg/String on /ghost/tts/text; '
            f'action: {self.cfg["action_name"]}; stop: /ghost/tts/stop')

    def emit(self,state,job_id,detail):
        self.events.put((state,job_id,detail))
        with self.action_condition:
            if job_id in self.action_job_to_goal:
                self.action_states[job_id] = (state,detail)
                self.action_condition.notify_all()

    def prepare(self):
        if self.cfg['backend'] == 'piper':
            return PiperEngine(
                self.cfg['model_path'],self.cfg['length_scale'],
                self.cfg['max_audio_seconds'])
        return DoubaoEngine.from_environment(
            model=self.cfg['doubao_model'],
            sample_rate=self.cfg['doubao_sample_rate'],
            speech_rate=self.cfg['doubao_speech_rate'],
            loudness_rate=self.cfg['doubao_loudness_rate'],
            endpoint=self.cfg['doubao_endpoint'],
            timeout_seconds=self.cfg['doubao_timeout_seconds'],
            max_audio_seconds=self.cfg['max_audio_seconds'])

    def perform(self,engine,job,emit):
        emit('synthesizing',job.job_id,'')
        audio,rate = engine.synthesize(job.text,job.cancel)
        if job.cancel.is_set():
            raise Cancelled()
        emit('processing',job.job_id,'')
        if self.cfg['apply_effects']:
            audio = cyber_effect(
                audio,rate,self.cfg['preset'],self.cfg['effect_strength'],
                self.cfg['volume'])
        else:
            audio = audio*self.cfg['volume']
        if job.cancel.is_set():
            raise Cancelled()
        emit('playing',job.job_id,'')
        play_audio(audio,rate,job.cancel,device_argument(self.cfg['audio_device']))

    def on_text(self,message):
        job_id = self.worker.submit(message.data)
        if job_id is not None:
            caption = clean_text(message.data,self.cfg['max_text_chars'])
            self.caption_pub.publish(String(data=caption))

    @staticmethod
    def _goal_key(goal_handle):
        return bytes(goal_handle.goal_id.uuid)

    def on_action_goal(self,goal_request):
        try:
            clean_text(goal_request.text,self.cfg['max_text_chars'])
        except ValueError:
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def on_action_cancel(self,goal_handle):
        key = self._goal_key(goal_handle)
        with self.action_condition:
            job_id = self.action_goal_to_job.get(key)
        if job_id is not None:
            self.worker.cancel(job_id,reason='cancel_requested')
        return CancelResponse.ACCEPT

    def execute_speech(self,goal_handle):
        """Run one cancellable FIFO goal; interrupt goals preempt the queue."""
        result = Speak.Result()
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.success = False
            result.job_id = 0
            result.message = 'cancelled_before_queue'
            return result

        if goal_handle.request.interrupt:
            self.worker.stop(reason='preempted')

        key = self._goal_key(goal_handle)

        def register(job):
            with self.action_condition:
                self.action_goal_to_job[key] = job.job_id
                self.action_job_to_goal[job.job_id] = key
                self.action_states[job.job_id] = ('queued','')

        text = clean_text(
            goal_handle.request.text,self.cfg['max_text_chars'])
        job_id = self.worker.submit(text,on_created=register)
        if job_id is None:
            goal_handle.abort()
            result.success = False
            result.job_id = 0
            result.message = 'speech_queue_rejected'
            return result
        self.caption_pub.publish(String(data=text))

        terminal_states = {'done','cancelled','error','expired','rejected'}
        last_feedback = None
        state,detail = 'queued',''
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self.worker.cancel(job_id)
            with self.action_condition:
                state,detail = self.action_states.get(
                    job_id,(state,detail))
                if state not in terminal_states:
                    self.action_condition.wait(timeout=0.05)
                    state,detail = self.action_states.get(
                        job_id,(state,detail))
            feedback_key = (state,detail)
            if feedback_key != last_feedback:
                feedback = Speak.Feedback()
                feedback.job_id = job_id
                feedback.state = state
                feedback.detail = detail
                goal_handle.publish_feedback(feedback)
                last_feedback = feedback_key
            if state in terminal_states:
                break

        with self.action_condition:
            self.action_goal_to_job.pop(key,None)
            self.action_job_to_goal.pop(job_id,None)
            self.action_states.pop(job_id,None)

        result.job_id = job_id
        result.success = state == 'done'
        result.message = detail or state
        if state == 'done':
            goal_handle.succeed()
        elif state == 'cancelled':
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
            else:
                goal_handle.abort()
        else:
            goal_handle.abort()
        return result

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
        self.action_server.destroy()
        if self.context.ok():
            self.busy_pub.publish(Bool(data=False))
            self.speaking_pub.publish(Bool(data=False))
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GhostTTS()
        # Each accepted Action waits for its worker job to reach a terminal
        # state. Keep enough executor threads for the entire bounded queue plus
        # cancel/interrupt callbacks, so a burst of stage cues can never starve
        # an urgent preemption request.
        executor = rclpy.executors.MultiThreadedExecutor(
            num_threads=min(32,int(node.cfg['queue_size'])+4))
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
