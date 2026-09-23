#!/usr/bin/env python3
"""
Ghost Game Monitor - live terminal dashboard for tuning/testing.

Subscribes to ghost_game_node's ~/state topic (published with
publish_debug_distances:=true) and redraws a table of per-joint
current position / distance-to-target / locked status a few times
per second. Purely a debug/dev tool - not used in the final demo
(disable publish_debug_distances / debug_reveal_targets for the
real show so the secret targets aren't visible anywhere).
"""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

CLEAR_SCREEN = '\x1b[2J\x1b[H'


class GhostGameMonitor(Node):

    def __init__(self):
        super().__init__('ghost_game_monitor')
        self.declare_parameter('state_topic', '/ghost_game_node/state')
        topic = self.get_parameter('state_topic').value
        self.create_subscription(String, topic, self._on_state, 10)
        self.get_logger().info(f'Watching {topic} ... (Ctrl+C to quit)')

    def _on_state(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        joints = payload.get('joints', [])
        positions = payload.get('positions', [])
        distances = payload.get('distances', [])
        targets = payload.get('targets')
        locked = payload.get('locked', [])

        lines = [
            CLEAR_SCREEN + 'Ghost Game Monitor',
            f"phase: {payload.get('phase')}   found: {payload.get('found_count')}/{payload.get('total')}",
            '',
            f"{'joint':<10}{'pos':>10}{'target':>10}{'dist':>10}{'locked':>8}",
        ]
        for i, name in enumerate(joints):
            pos = positions[i] if i < len(positions) else None
            dist = distances[i] if i < len(distances) else None
            tgt = targets[i] if targets and i < len(targets) else None
            is_locked = locked[i] if i < len(locked) else False
            lines.append(
                f"{name:<10}"
                f"{'--' if pos is None else f'{pos:.3f}':>10}"
                f"{'--' if tgt is None else f'{tgt:.3f}':>10}"
                f"{'--' if dist is None else f'{dist:.3f}':>10}"
                f"{'YES' if is_locked else '':>8}"
            )
        print('\n'.join(lines))


def main(args=None):
    rclpy.init(args=args)
    node = GhostGameMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
