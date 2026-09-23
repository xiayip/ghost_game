"""Core regression tests runnable without ROS, a model, or an audio device."""
import sys
import base64
import json
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from ghost_tts.effects import cyber_effect, PRESETS
from ghost_tts.engine import (
    Cancelled,DoubaoEngine,play_audio,resolve_tts_backend)
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

    def test_cancel_targets_one_queued_job(self):
        release = threading.Event()
        started = threading.Event()
        seen = []
        def perform(resource, job, emit):
            seen.append(job.text)
            if job.text == 'active':
                started.set()
                release.wait(3)
        worker = self.make_worker(perform)
        worker.start()
        worker.submit('active')
        self.assertTrue(started.wait(3))
        cancelled_id = worker.submit('cancel me')
        kept_id = worker.submit('keep me')

        self.assertTrue(worker.cancel(cancelled_id))
        self.assertFalse(worker.cancel(9999))
        release.set()
        eventually(lambda: ('done', kept_id, '') in self.events)

        self.assertEqual(seen, ['active', 'keep me'])
        self.assertIn(
            ('cancelled', cancelled_id, 'cleared_from_queue'), self.events)

    def test_interrupt_reason_reaches_active_and_queued_jobs(self):
        started = threading.Event()
        def perform(resource, job, emit):
            started.set()
            if job.cancel.wait(3):
                raise Cancelled()
        worker = self.make_worker(perform)
        worker.start()
        active_id = worker.submit('active')
        self.assertTrue(started.wait(3))
        queued_id = worker.submit('queued')

        self.assertEqual(worker.stop(reason='preempted'), 2)
        eventually(
            lambda: ('cancelled', active_id, 'preempted') in self.events)

        self.assertIn(('cancelled', queued_id, 'preempted'), self.events)

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


class DoubaoEngineTests(unittest.TestCase):
    class Response:
        status_code = 200
        headers = {}

        def __init__(self,lines):
            self.lines = lines
            self.closed = False

        def iter_lines(self):
            yield from self.lines

        def close(self):
            self.closed = True

    class Session:
        def __init__(self,response):
            self.response = response
            self.request = None

        def post(self,url,**kwargs):
            self.request = (url,kwargs)
            return self.response

    @staticmethod
    def event(value):
        return ('data: '+json.dumps(value)).encode()

    def make_engine(self,lines,**kwargs):
        response = self.Response(lines)
        session = self.Session(response)
        engine = DoubaoEngine(
            'test-secret','S_test','seed-icl-2.0',session=session,
            **kwargs)
        return engine,session,response

    def test_sse_pcm_synthesis_and_v3_request(self):
        pcm = np.array([-32768,0,32767],dtype='<i2').tobytes()
        lines = [
            b': keep-alive',
            self.event({'code':0,'data':base64.b64encode(pcm).decode()}),
            self.event({'code':20000000,'message':'ok','data':None}),
        ]
        engine,session,response = self.make_engine(lines)

        audio,rate = engine.synthesize('幽灵链路已建立',threading.Event())

        self.assertEqual(rate,24000)
        np.testing.assert_allclose(
            audio,np.array([-1,0,32767/32768],dtype=np.float32))
        url,request = session.request
        self.assertEqual(url,DoubaoEngine.DEFAULT_ENDPOINT)
        self.assertTrue(request['stream'])
        self.assertEqual(request['headers']['X-Api-Key'],'test-secret')
        self.assertEqual(
            request['headers']['X-Api-Resource-Id'],'seed-icl-2.0')
        self.assertEqual(request['json']['req_params']['speaker'],'S_test')
        self.assertEqual(
            request['json']['req_params']['audio_params']['format'],'pcm')
        self.assertTrue(response.closed)

    def test_cancellation_interrupts_response_collection(self):
        cancel = threading.Event()
        first = self.event({
            'code':0,
            'data':base64.b64encode(np.zeros(2,dtype='<i2')).decode(),
        })

        class CancelResponse(self.Response):
            def iter_lines(inner_self):
                yield first
                cancel.set()
                yield first

        response = CancelResponse([])
        engine = DoubaoEngine(
            'test-secret','S_test','seed-icl-2.0',
            session=self.Session(response))
        with self.assertRaises(Cancelled):
            engine.synthesize('cancel me',cancel)
        self.assertTrue(response.closed)

    def test_api_and_network_errors_never_expose_key(self):
        engine,_,_ = self.make_engine([
            self.event({'code':45000000,'message':'voice denied'})])
        with self.assertRaisesRegex(RuntimeError,'45000000') as caught:
            engine.synthesize('test',threading.Event())
        self.assertNotIn('test-secret',str(caught.exception))

        class FailingSession:
            def post(self,*args,**kwargs):
                raise RuntimeError('request included test-secret')

        engine = DoubaoEngine(
            'test-secret','S_test','seed-icl-2.0',session=FailingSession())
        with self.assertRaisesRegex(RuntimeError,'request failed') as caught:
            engine.synthesize('test',threading.Event())
        self.assertNotIn('test-secret',str(caught.exception))

    def test_environment_credentials_are_required(self):
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaisesRegex(ValueError,'DOUBAO_TTS_API_KEY'):
                DoubaoEngine.from_environment()

    def test_auto_backend_requires_all_credentials(self):
        complete = {
            'DOUBAO_TTS_API_KEY':'key',
            'DOUBAO_TTS_VOICE_TYPE':'S_test',
            'DOUBAO_TTS_RESOURCE_ID':'seed-icl-2.0',
        }
        self.assertEqual(resolve_tts_backend('auto',complete),'doubao')
        incomplete = dict(complete)
        incomplete.pop('DOUBAO_TTS_API_KEY')
        self.assertEqual(resolve_tts_backend('auto',incomplete),'piper')
        self.assertEqual(resolve_tts_backend('doubao',{}),'doubao')
        with self.assertRaisesRegex(ValueError,'auto, piper, or doubao'):
            resolve_tts_backend('unknown',complete)


if __name__ == '__main__':
    unittest.main()
