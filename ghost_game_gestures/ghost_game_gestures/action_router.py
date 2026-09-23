"""Opt-in async adapter for gesture labels and completed-motion Trigger services.

Default preview output cannot command the arm. Trigger backends must reply only
when their interaction completes; enqueue-only services need a separate executor.
"""
import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .routing import DEFAULT_ROUTES, EventGate, string_map


class GestureActionRouter(Node):
    def __init__(self):
        super().__init__('gesture_action_router')
        defaults = {
            'dry_run': True, 'events_topic': '/gestures/events',
            'requests_topic': '/interaction/requests', 'status_topic': '/interaction/status',
            'routes_json': json.dumps(DEFAULT_ROUTES), 'service_map_json': '{}',
            'event_max_age': .25, 'min_score': .65, 'cooldown_seconds': 1.5,
            'service_timeout': 15.,
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        p = lambda name: self.get_parameter(name).value
        self.dry_run = bool(p('dry_run'))
        self.timeout = float(p('service_timeout'))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError('service_timeout must be finite and positive')
        routes = string_map(json.loads(p('routes_json')))
        services = string_map(json.loads(p('service_map_json')))
        if any(not name.startswith('/') for name in services.values()):
            raise ValueError('service_map_json must contain absolute ROS service names')
        if any(action not in routes.values() for action in services):
            raise ValueError('every service_map_json action must appear in routes_json')
        self.gate = EventGate(routes, float(p('event_max_age')),
                              float(p('min_score')), float(p('cooldown_seconds')))
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self.requests = self.create_publisher(String, p('requests_topic'), qos)
        self.status = self.create_publisher(String, p('status_topic'), qos)
        self.clients = {action: self.create_client(Trigger, endpoint)
                        for action, endpoint in services.items()}
        self.pending = None
        self.create_subscription(String, p('events_topic'), self.on_event, qos)
        self.create_timer(.05, self.poll)
        self.get_logger().info('Gesture router: ' + ('preview only' if self.dry_run else
                                                   'configured Trigger backends enabled'))

    @staticmethod
    def publish(publisher, data):
        message = String()
        message.data = json.dumps(data, allow_nan=False, separators=(',', ':'))
        publisher.publish(message)

    def on_event(self, message):
        # Bound parsing cost; detailed perception arrays do not belong on events.
        if len(message.data) > 8192:
            self.publish(self.status, {'accepted': False, 'reason': 'oversized_event'})
            return
        try:
            event = json.loads(message.data)
        except (ValueError, TypeError):
            self.publish(self.status, {'accepted': False, 'reason': 'invalid_json'})
            return
        decision = self.gate.evaluate(event, self.get_clock().now().nanoseconds/1e9,
                                      time.monotonic(), busy=self.pending is not None)
        decision['dry_run'] = self.dry_run
        if not decision['accepted']:
            self.publish(self.status, decision)
            return
        # Reviewable requests contain intent, not joint positions or velocities.
        self.publish(self.requests, dict(decision, executed=False))
        if self.dry_run:
            self.publish(self.status, dict(decision, reason='preview', executed=False))
            return
        client = self.clients.get(decision['action'])
        if client is None or not client.service_is_ready():
            self.publish(self.status, dict(decision, accepted=False, executed=False,
                         reason='unmapped_service' if client is None else 'service_unavailable'))
            return
        try:
            future = client.call_async(Trigger.Request())
        except Exception as error:
            self.publish(self.status, dict(decision, accepted=False, executed=False,
                                          reason='dispatch_error', detail=str(error)))
            return
        self.pending = [future, time.monotonic(), decision, False]
        self.publish(self.status, dict(decision, reason='dispatched'))

    def poll(self):
        if self.pending is None:
            return
        future, started, decision, timed_out = self.pending
        if future.done():
            self.pending = None
            try:
                result = future.result()
                self.publish(self.status, dict(decision, reason='completed' if result.success else
                                               'backend_rejected', success=bool(result.success),
                                               detail=result.message, after_timeout=timed_out))
            except Exception as error:
                self.publish(self.status, dict(decision, reason='backend_error', detail=str(error)))
        elif not timed_out and time.monotonic()-started > self.timeout:
            self.pending[3] = True
            # A Trigger request has no motion cancellation protocol. Keep busy;
            # do not stack a new command on a possibly still moving backend.
            self.publish(self.status, dict(decision, reason='timeout_busy',
                                          detail='Awaiting backend completion; new events are blocked'))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GestureActionRouter()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
