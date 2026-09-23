#!/usr/bin/env python3
"""Start a Ghost round and request its built-in development auto-solver."""

import argparse
import json
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from std_msgs.msg import String
from std_srvs.srv import Trigger


class MockSolveClient(Node):

    def __init__(self):
        super().__init__('ghost_game_mock_solve')
        self.state = None
        self.create_subscription(
            String, '/ghost_game_node/state', self._state_cb, 10)
        self.start_client = self.create_client(
            Trigger, '/ghost_game_node/start')
        self.solve_client = self.create_client(
            Trigger, '/ghost_game_node/mock_solve')

    def _state_cb(self, message):
        try:
            self.state = json.loads(message.data)
        except (TypeError, ValueError):
            self.get_logger().warn('Ignoring malformed Ghost state JSON')

    def wait_for_state(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.state is not None and predicate(self.state):
                return True
        return False

    def call(self, client, timeout):
        if not client.wait_for_service(timeout_sec=timeout):
            return None, 'service unavailable (restart the Ghost Game launch)'
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not future.done():
            if time.monotonic() >= deadline:
                return None, 'service response timeout'
            rclpy.spin_once(self, timeout_sec=0.1)
        return future.result(), None


def _parse_args(args):
    parser = argparse.ArgumentParser(
        description=(
            'Start a Ghost round, wait for searching, then run the soft '
            'quintic mock trajectory to the secret pose.'))
    parser.add_argument(
        '--no-start', action='store_true',
        help='require an already-running round instead of calling ~/start')
    parser.add_argument(
        '--setup-timeout', type=float, default=15.0,
        help='seconds to wait for the searching phase (default: 15)')
    parser.add_argument(
        '--completion-timeout', type=float, default=90.0,
        help='seconds to wait for all joints to lock (default: 90)')
    return parser.parse_args(remove_ros_args(args=args)[1:])


def main(args=None):
    cli_args = _parse_args(sys.argv if args is None else args)
    rclpy.init(args=args)
    node = MockSolveClient()
    exit_code = 1
    try:
        if not node.wait_for_state(lambda state: 'phase' in state, 5.0):
            node.get_logger().error('No /ghost_game_node/state received')
            return exit_code
        if 'mock_solve_active' not in node.state:
            node.get_logger().error(
                'Running ghost_game_node predates mock-solve support; '
                'restart the Ghost Game launch')
            return exit_code

        phase = node.state.get('phase')
        if not cli_args.no_start and phase == 'idle':
            response, call_error = node.call(node.start_client, 5.0)
            if response is None or not response.success:
                reason = call_error if response is None else response.message
                node.get_logger().error(f'Could not start Ghost round: {reason}')
                return exit_code
            node.get_logger().info('Ghost round started; waiting for search mode')
        elif phase not in ('setup', 'searching'):
            node.get_logger().error(
                f'Ghost phase is {phase!r}; return home before mock solving')
            return exit_code

        if not node.wait_for_state(
                lambda state: state.get('phase') == 'searching',
                cli_args.setup_timeout):
            phase = None if node.state is None else node.state.get('phase')
            node.get_logger().error(
                f'Ghost did not enter searching phase (phase={phase!r})')
            return exit_code

        response, call_error = node.call(node.solve_client, 5.0)
        if response is None or not response.success:
            reason = call_error if response is None else response.message
            node.get_logger().error(f'Mock solve was rejected: {reason}')
            return exit_code
        node.get_logger().info(response.message)

        def all_found(state):
            total = int(state.get('total', 0))
            return total > 0 and int(state.get('found_count', 0)) >= total

        if not node.wait_for_state(all_found, cli_args.completion_timeout):
            phase = None if node.state is None else node.state.get('phase')
            count = 0 if node.state is None else node.state.get('found_count', 0)
            node.get_logger().error(
                f'Mock solve did not finish (phase={phase!r}, found={count})')
            return exit_code

        node.get_logger().info(
            'All secret joints found; normal success flow has taken over')
        exit_code = 0
        return exit_code
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
