"""Bounded latest-frame scheduling and geometric face association (no ROS)."""
from threading import Condition, Event, Thread
import math
import time


class LatestSlot:
    def __init__(self):
        self.condition = Condition()
        self.item = None
        self.overwritten = 0

    def put(self, item):
        with self.condition:
            if self.item is not None:
                self.overwritten += 1
            self.item = item
            self.condition.notify()

    def take(self, timeout=0):
        with self.condition:
            if self.item is None and timeout:
                self.condition.wait_for(lambda: self.item is not None, timeout)
            item, self.item = self.item, None
            return item


class LatestWorker:
    """One active, one pending and one completed item; never FIFO backlog."""
    def __init__(self, process, on_error, max_fps):
        if not math.isfinite(max_fps) or max_fps <= 0:
            raise ValueError('max_fps must be positive and finite')
        self.process, self.on_error = process, on_error
        self.period = 1/max_fps
        self.pending, self.completed = LatestSlot(), LatestSlot()
        self.stop = Event()
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        deadline = 0.
        while not self.stop.is_set():
            if self.stop.wait(max(0., deadline-time.perf_counter())):
                break
            # Wait BEFORE taking a job, so rate limiting also selects newest.
            job = self.pending.take(timeout=.05)
            if job is None:
                continue
            started = time.perf_counter()
            # Keep the cadence phase instead of adding scheduling jitter to
            # every deadline; otherwise a 30 Hz source drifts into old frames.
            deadline = started+self.period if deadline == 0. else max(deadline+self.period,started)
            try:
                result = self.process(job)
            except Exception as error:
                result = self.on_error(job,error)
            if not self.stop.is_set():
                self.completed.put(result)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)


def source_age_seconds(stamp_ns, now_ns, max_age, future_tolerance=.03):
    """Reject missing, future and expired sensor stamps; never re-date images."""
    age = (now_ns-stamp_ns)/1e9
    if stamp_ns <= 0 or not -future_tolerance <= age <= max_age:
        return None
    return max(0.,age)


def _iou(a,b):
    left,top = max(a.x,b.x),max(a.y,b.y)
    right,bottom = min(a.x+a.width,b.x+b.width),min(a.y+a.height,b.y+b.height)
    intersection = max(0,right-left)*max(0,bottom-top)
    return intersection/max(1,a.area+b.area-intersection)


class FaceTracker:
    """Short-term box association, not biometric identity. Fail on ambiguity."""
    def __init__(self, lost_timeout=.4):
        self.lost_timeout = lost_timeout
        self.previous = None
        self.last_seen = None
        self.last_stamp = None
        self.counter = 0
        self.track_id = ''

    def update(self, candidates, stamp):
        if self.last_stamp is not None and stamp <= self.last_stamp:
            return None,self.track_id,'non_increasing_stamp'
        self.last_stamp = stamp
        if self.last_seen is not None and stamp-self.last_seen > self.lost_timeout:
            self.previous = None
        if not candidates:
            return None,self.track_id,'no_face'
        if self.previous is None:
            selected = max(candidates,key=lambda b:(b.area,b.score,-b.x,-b.y))
            self.counter += 1
            self.track_id = f'face_track_{self.counter}'
        else:
            previous = self.previous
            ranked = []
            for box in candidates:
                iou = _iou(previous,box)
                displacement = math.hypot((box.x+box.width/2)-(previous.x+previous.width/2),
                                          (box.y+box.height/2)-(previous.y+previous.height/2))
                relative = displacement/max(1.,math.hypot(previous.width,previous.height))
                if iou >= .15 and relative <= .65 and .4 <= box.area/previous.area <= 2.5:
                    ranked.append((iou-.2*relative,box))
            ranked.sort(key=lambda item:item[0],reverse=True)
            if not ranked:
                return None,self.track_id,'target_lost'
            if len(ranked)>1 and ranked[0][0]-ranked[1][0] < .12:
                return None,self.track_id,'ambiguous_target'
            selected = ranked[0][1]
        self.previous,self.last_seen = selected,stamp
        return selected,self.track_id,'tracking'
