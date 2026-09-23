"""Webcam/video gesture preview. No ROS publishers or motor commands are created."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import threading
import time

from .download_model import DEFAULT_MODEL_PATH


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


class LatestCapture:
    """Drain capture continuously into one slot; video files play in real time."""

    def __init__(self, cv2, source, *, video: bool = False, mirror: bool = False):
        self.cv2 = cv2
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Cannot open {'video' if video else 'camera'}: {source}")
        if not video:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # Some backends ignore this hint. The one-slot worker still avoids
            # adding an application FIFO; hardware latency is not measurable here.
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.video_period = 1.0 / fps if video and math.isfinite(fps) and fps > 0 else (1 / 30 if video else 0)
        self.video = video
        self.mirror = mirror
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.finished = threading.Event()
        self.latest = None
        self.error = None
        self.thread = threading.Thread(target=self._run, name="gesture-preview-camera", daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        sequence = 0
        next_capture = time.monotonic()
        try:
            while not self.stop.is_set():
                if self.video_period and self.stop.wait(max(0, next_capture - time.monotonic())):
                    break
                success, frame = self.capture.read()
                if not success:
                    if not self.video:
                        self.error = "Camera stopped delivering frames"
                    break
                stamp = time.monotonic()
                if self.mirror:
                    frame = self.cv2.flip(frame, 1)
                sequence += 1
                with self.lock:
                    self.latest = (sequence, stamp, frame)
                next_capture = max(next_capture + self.video_period, stamp)
        except Exception as exc:
            self.error = f"Capture failed: {exc}"
        finally:
            self.capture.release()
            self.finished.set()

    def snapshot(self):
        with self.lock:
            return self.latest

    def close(self):
        self.stop.set()
        if self.thread.ident is None:
            self.capture.release()
            return
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            # A backend can be stuck in read() after USB disconnect. Interrupt
            # capture once, then bound shutdown instead of hanging on exit.
            self.capture.release()
            self.thread.join(timeout=2)


def draw_result(cv2, frame, hands, result, *, inference_ms, age_ms, event_text):
    height, width = frame.shape[:2]
    canvas = frame.copy()
    for hand in hands:
        points = [(int(p[0] * width), int(p[1] * height)) for p in hand.landmarks]
        for left, right in HAND_CONNECTIONS:
            if max(left, right) < len(points):
                cv2.line(canvas, points[left], points[right], (220, 190, 60), 2)
        for point in points:
            cv2.circle(canvas, point, 3, (80, 240, 120), -1)
        if points:
            cv2.putText(canvas, f"{hand.label} {hand.score:.2f}", points[0],
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    control = result.get("control", {})
    status = str(result.get("label", "none")) if result.get("valid") else str(result.get("reason", "invalid"))
    lines = [
        f"Label: {status} | hand ID: {result.get('hand_id', '-')} | score: {result.get('score', 0):.2f}",
        f"Control: {'ACTIVE' if control.get('active') else 'off'}  dx={control.get('dx', 0):+.2f}  dy={control.get('dy', 0):+.2f}  dz={control.get('dz', 0):+.2f}",
        f"Inference: {inference_ms:.1f} ms | local frame age: {age_ms:.1f} ms",
        f"Event: {event_text or '-'}",
        "PREVIEW ONLY | q / Esc: exit",
    ]
    for index, line in enumerate(lines):
        y = 22 + index * 23
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .47, (0, 0, 0), 3)
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .47, (240, 240, 240), 1)
    return canvas


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    source.add_argument("--input-video", type=Path, help="Replay a local video at its nominal frame rate")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--fps", type=float, default=15, help="Maximum inference rate")
    parser.add_argument("--max-width", type=int, default=640)
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--max-age", type=float, default=.20, help="Maximum local frame age in seconds")
    parser.add_argument("--mirror", action="store_true", help="Mirror BEFORE inference; axes follow the displayed image")
    parser.add_argument("--headless", action="store_true", help="No OpenCV window")
    parser.add_argument("--json-every-frame", action="store_true", help="Print every result instead of events only")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N processed frames; 0 runs until quit/EOF")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    if not math.isfinite(args.max_age) or args.max_age <= 0:
        parser.error("--max-age must be finite and positive")
    if args.num_hands < 1 or args.max_width < 64 or args.max_frames < 0:
        parser.error("Use num-hands >= 1, max-width >= 64, max-frames >= 0")
    if args.input_video and not args.input_video.is_file():
        parser.error(f"Video does not exist: {args.input_video}")
    if not args.model_path.expanduser().is_file():
        parser.error("Model missing. Run python -m ghost_game_gestures.download_model first.")

    # Keep --help usable without MediaPipe, NumPy or an installed ROS runtime.
    from .core import GestureEngine
    from .model import GestureModel
    try:
        import cv2
    except ImportError:
        parser.exit(1, "OpenCV missing. Install ghost_game_gestures/requirements.txt.\n")
    capture = None
    model = None
    try:
        model = GestureModel(model_path=str(args.model_path.expanduser()),
                             num_hands=args.num_hands, max_width=args.max_width)
        engine = GestureEngine()
        capture = LatestCapture(cv2, str(args.input_video) if args.input_video else args.camera,
                                video=args.input_video is not None, mirror=args.mirror).start()
        last_sequence = processed = 0
        last_frame_at = time.monotonic()
        next_inference = 0.0
        event_text = ""
        print("Gesture preview: outputs labels/events only; no robot control.", file=sys.stderr)
        while True:
            if not args.headless:
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                if processed and cv2.getWindowProperty("Ghost Game Gestures", cv2.WND_PROP_VISIBLE) < 1:
                    break
            now = time.monotonic()
            slot = capture.snapshot()
            if slot is None or slot[0] == last_sequence:
                if capture.finished.is_set():
                    if capture.error:
                        raise RuntimeError(capture.error)
                    break
                if now - last_frame_at > 3:
                    raise RuntimeError("No new camera frame for 3 seconds")
                time.sleep(.005)
                continue
            if now < next_inference:
                time.sleep(min(.01, next_inference - now))
                continue
            last_sequence, stamp, frame = slot
            last_frame_at = now
            next_inference = now + 1.0 / args.fps
            started = time.monotonic()
            hands = model.recognize(frame) if started - stamp <= args.max_age else []
            finished = time.monotonic()
            result = engine.process(hands, stamp=stamp, now=finished,
                                    frame_size=(frame.shape[1], frame.shape[0]), max_age=args.max_age)
            if result.get("events"):
                event_text = ", ".join(str(e.get("label", e.get("action", "event"))) for e in result["events"])
            if args.json_every_frame or result.get("events"):
                print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
            if not args.headless:
                canvas = draw_result(cv2, frame, hands, result, inference_ms=(finished - started) * 1000,
                                     age_ms=(finished - stamp) * 1000, event_text=event_text)
                cv2.imshow("Ghost Game Gestures", canvas)
            processed += 1
            if args.max_frames and processed >= args.max_frames:
                break
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Gesture preview failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if capture is not None:
            capture.close()
        if model is not None:
            model.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
