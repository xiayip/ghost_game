"""Bounded FIFO with cancellation; independent of ROS for direct tests."""
from collections import deque
from dataclasses import dataclass,field
import threading
import time
import unicodedata
from .engine import Cancelled


@dataclass
class Job:
    job_id: int
    text: str
    created: float
    cancel: threading.Event = field(default_factory=threading.Event)
    cancel_detail: str = ''


def clean_text(text,max_chars):
    if not isinstance(text,str):
        raise ValueError('Text must be a string')
    # Treat control characters as separators; no shell / SSML interpretation.
    text = ''.join(' ' if unicodedata.category(c).startswith('C') else c for c in text)
    text = ' '.join(text.split())
    # Piper supports inline phonemes. Accept plain text only in this interface.
    text = text.replace('[[','［［').replace(']]','］］')
    if not text:
        raise ValueError('empty_text')
    if len(text)>max_chars:
        raise ValueError('text_too_long')
    return text


class SpeechWorker:
    def __init__(self,prepare,perform,emit,queue_size=8,max_chars=500,max_wait=60.0):
        if queue_size<1 or max_chars<1 or max_wait<=0:
            raise ValueError('Invalid queue limits')
        self.prepare,self.perform,self.emit = prepare,perform,emit
        self.queue_size,self.max_chars,self.max_wait = queue_size,max_chars,max_wait
        self.pending = deque()
        self.condition = threading.Condition()
        self.current = None
        self.closed = False
        self.failure = None
        self.sequence = 0
        self.thread = threading.Thread(target=self.run,name='ghost-tts-worker',daemon=True)

    def start(self):
        self.thread.start()

    def submit(self,text,on_created=None):
        try:
            text = clean_text(text,self.max_chars)
        except ValueError as error:
            self.emit('rejected',0,str(error))
            return None
        with self.condition:
            if self.closed or self.failure:
                self.emit('rejected',0,'not_available')
                return None
            if len(self.pending)>=self.queue_size:
                self.emit('rejected',0,'queue_full')
                return None
            self.sequence += 1
            job = Job(self.sequence,text,time.monotonic())
            if on_created is not None:
                on_created(job)
            self.pending.append(job)
            self.emit('queued',job.job_id,'')
            self.condition.notify()
            return job.job_id

    def cancel(self,job_id,reason='cancel_requested'):
        """Cancel one active or queued job without disturbing other speech."""
        with self.condition:
            if self.current is not None and self.current.job_id == job_id:
                self.current.cancel_detail = reason
                self.current.cancel.set()
                self.condition.notify_all()
                return True
            for job in list(self.pending):
                if job.job_id != job_id:
                    continue
                self.pending.remove(job)
                job.cancel.set()
                self.emit('cancelled',job.job_id,'cleared_from_queue')
                self.condition.notify_all()
                return True
            return False

    def stop(self,reason=''):
        with self.condition:
            count = len(self.pending)+(1 if self.current else 0)
            while self.pending:
                job = self.pending.popleft()
                job.cancel.set()
                self.emit(
                    'cancelled',job.job_id,reason or 'cleared_from_queue')
            if self.current:
                self.current.cancel_detail = reason
                self.current.cancel.set()
            self.condition.notify_all()
            return count

    def close(self):
        with self.condition:
            self.closed = True
            self.stop()
            self.condition.notify_all()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def run(self):
        self.emit('loading',0,'')
        try:
            resource = self.prepare()
        except Exception as error:
            with self.condition:
                self.failure = str(error)
                while self.pending:
                    self.emit('error',self.pending.popleft().job_id,'engine_initialization_failed')
            self.emit('error',0,str(error))
            return
        if not self.closed:
            self.emit('ready',0,'')
        while True:
            with self.condition:
                while not self.pending and not self.closed:
                    self.condition.wait()
                if self.closed:
                    return
                job = self.pending.popleft()
                if time.monotonic()-job.created>self.max_wait:
                    self.emit('expired',job.job_id,'max_queue_wait_exceeded')
                    continue
                self.current = job
            try:
                if job.cancel.is_set():
                    raise Cancelled()
                self.perform(resource,job,self.emit)
                self.emit('cancelled' if job.cancel.is_set() else 'done',job.job_id,'')
            except Cancelled:
                self.emit('cancelled',job.job_id,job.cancel_detail)
            except Exception as error:
                self.emit('error',job.job_id,str(error))
            finally:
                with self.condition:
                    self.current = None
