"""Core regression tests runnable without ROS, a model, or an audio device."""
import sys
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from ghost_tts.effects import cyber_effect, PRESETS
from ghost_tts.engine import Cancelled, play_audio
from ghost_tts.worker import SpeechWorker, clean_text


def eventually(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError('Timed out waiting for worker')


class EffectsTests(unittest.TestCase):
    def test_presets_finite_bounded_and_distinct(self):
        source = (2 * np.sin(np.arange(22050) * 2*np.pi*200/22050)).astype(np.float32)
        rendered = []
        for preset in PRESETS:
            result = cyber_effect(source, 22050, preset)
            self.assertEqual(result.dtype, np.float32)
            self.assertTrue(np.isfinite(result).all())
            self.assertLessEqual(float(np.max(np.abs(result))), 0.645)
            rendered.append(result)
        self.assertFalse(np.allclose(rendered[1], rendered[2]))
        self.assertFalse(np.allclose(rendered[2], rendered[3]))

    def test_silence_stays_silent(self):
        for preset in PRESETS:
            self.assertEqual(np.count_nonzero(cyber_effect(np.zeros(22050), 22050, preset)), 0)

    def test_zero_strength_matches_clean(self):
        source = np.linspace(-.5, .5, 2205)
        np.testing.assert_array_equal(cyber_effect(source, 22050, strength=0),
                                      cyber_effect(source, 22050, preset='clean'))

    def test_invalid_audio_and_parameters_rejected(self):
        for audio, kwargs in [(np.array([np.nan]), {}), (np.zeros((3, 2)), {}),
                              (np.zeros(20), {'strength': -1}),
                              (np.zeros(20), {'preset': 'missing'})]:
            with self.assertRaises(ValueError):
                cyber_effect(audio, 22050, **kwargs)


class WorkerTests(unittest.TestCase):
    def make_worker(self, perform, prepare=lambda: None, **kwargs):
        self.events = []
        self.worker = SpeechWorker(prepare, perform, lambda *event: self.events.append(event), **kwargs)
        self.addCleanup(self.worker.close)
        return self.worker

    def test_fifo_and_recovery_after_utterance_error(self):
        seen = []
        def perform(resource, job, emit):
            seen.append(job.text)
            if job.text == 'bad':
                raise RuntimeError('simulated output error')
        worker = self.make_worker(perform)
        for text in ['one', 'bad', 'three']:
            worker.submit(text)
        worker.start()
        eventually(lambda: ('done', 3, '') in self.events)
        self.assertEqual(seen, ['one', 'bad', 'three'])
        self.assertIn(('error', 2, 'simulated output error'), self.events)

    def test_stop_cancels_active_and_pending_then_accepts_new_job(self):
        started = threading.Event()
        seen = []
        def perform(resource, job, emit):
            seen.append(job.text)
            if job.text == 'active':
                started.set()
                if not job.cancel.wait(3):
                    raise RuntimeError('test failed to stop')
                raise Cancelled()
        worker = self.make_worker(perform)
        worker.start()
        worker.submit('active')
        self.assertTrue(started.wait(3))
        worker.submit('queued')
        self.assertEqual(worker.stop(), 2)
        eventually(lambda: ('cancelled', 1, '') in self.events)
        worker.submit('new')
        eventually(lambda: ('done', 3, '') in self.events)
        self.assertEqual(seen, ['active', 'new'])
        self.assertIn(('cancelled', 2, 'cleared_from_queue'), self.events)

    def test_queue_limit_and_invalid_text(self):
        worker = self.make_worker(lambda *args: None, queue_size=1, max_chars=5)
        self.assertEqual(worker.submit('first'), 1)
        self.assertIsNone(worker.submit('next'))
        self.assertIsNone(worker.submit('  '))
        self.assertIsNone(worker.submit('too long'))
        self.assertIn(('rejected', 0, 'queue_full'), self.events)
        self.assertIn(('rejected', 0, 'empty_text'), self.events)
        self.assertIn(('rejected', 0, 'text_too_long'), self.events)

    def test_initialization_failure_rejects_future_work(self):
        def prepare():
            raise RuntimeError('missing model')
        worker = self.make_worker(lambda *args: None, prepare=prepare)
        worker.submit('queued')
        worker.start()
        eventually(lambda: ('error', 0, 'missing model') in self.events)
        self.assertIsNone(worker.submit('new'))
        self.assertIn(('error', 1, 'engine_initialization_failed'), self.events)

    def test_expired_job_is_not_spoken(self):
        seen = []
        worker = self.make_worker(lambda *args: seen.append(True), max_wait=1)
        worker.submit('stale')
        worker.pending[0].created -= 2
        worker.start()
        eventually(lambda: ('expired', 1, 'max_queue_wait_exceeded') in self.events)
        self.assertEqual(seen, [])

    def test_plain_text_sanitizing(self):
        self.assertEqual(clean_text('hello\x00\n world', 30), 'hello world')
        self.assertEqual(clean_text('[[abc]]', 30), '［［abc］］')


class PlaybackTests(unittest.TestCase):
    def test_pulse_device_uses_pulse_backend(self):
        cancel = threading.Event()
        audio = np.zeros(100, dtype=np.float32)
        with patch('ghost_tts.engine._play_audio_pulse') as pulse:
            play_audio(audio, 22050, cancel, 'pulse:show-speaker')
        pulse.assert_called_once_with(audio, 22050, cancel, 'show-speaker')

    def test_abort_on_cancellation_and_close_device(self):
        cancel = threading.Event()
        actions = []
        class Stream:
            def start(self): actions.append('start')
            def write(self, block):
                actions.append('write')
                cancel.set()
            def stop(self): actions.append('stop')
            def abort(self): actions.append('abort')
            def close(self): actions.append('close')
        fake = SimpleNamespace(check_output_settings=lambda **kw: None,
                               OutputStream=lambda **kw: Stream(), PortAudioError=RuntimeError)
        with patch.dict(sys.modules, {'sounddevice': fake}):
            with self.assertRaises(Cancelled):
                play_audio(np.zeros(22050, dtype=np.float32), 22050, cancel)
        self.assertEqual(actions, ['start', 'write', 'abort', 'close'])

    def test_resample_to_device_native_rate(self):
        observed = {'samples': 0}
        def check(**kwargs):
            if kwargs['samplerate'] == 22050:
                raise RuntimeError('unsupported rate')
        class Stream:
            def start(self): pass
            def write(self, block): observed['samples'] += len(block)
            def stop(self): observed['stopped'] = True
            def close(self): observed['closed'] = True
        def output(**kwargs):
            observed['rate'] = kwargs['samplerate']
            return Stream()
        fake = SimpleNamespace(check_output_settings=check, OutputStream=output,
                               query_devices=lambda *a, **kw: {'default_samplerate': 48000},
                               PortAudioError=RuntimeError)
        with patch.dict(sys.modules, {'sounddevice': fake}):
            play_audio(np.zeros(22050, dtype=np.float32), 22050, threading.Event())
        self.assertEqual(observed, {'samples': 48000, 'rate': 48000, 'stopped': True, 'closed': True})


if __name__ == '__main__':
    unittest.main()
