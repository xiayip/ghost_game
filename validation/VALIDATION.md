# Local verification

2026-09-23: 38 offline tests passed. All Python files parsed and package.xml files parsed.
Tests: detector test_core/test_runtime; orchestrator test_face_tracking/test_face_latency/test_face_callback/test_mock_trajectory.
Actual YuNet 2023/OpenCV 4.10 CPU model exercised by the benchmark; source methods from baseline and updated node.
Benchmark clocks use high-resolution perf_counter; fake ROS messages, emulated backlog, public static fixture.
Not run: colcon, actual ROS adapters/DDS, existing ROS-dependent TTS cue tests, Orbbec capture, RS hardware, DGX Spark.
Source basis: xiayip/ghost_game 961fab056aae235cb5d26f6a13bb6c2e2ed30b63.
See ../LIGHTWEIGHT_FOLLOW.md for commands and limitations. No physical latency guarantee.
