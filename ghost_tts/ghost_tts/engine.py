from pathlib import Path
import base64
import ctypes
import ctypes.util
import json
import os
import stat
import uuid
import wave
import numpy as np


class Cancelled(Exception):
    pass


DOUBAO_ENV_VARS = (
    'DOUBAO_TTS_API_KEY',
    'DOUBAO_TTS_VOICE_TYPE',
    'DOUBAO_TTS_RESOURCE_ID',
)


def resolve_tts_backend(requested,environ=None):
    """Resolve piper/doubao/auto without exposing credential values."""
    backend = str(requested).strip().lower()
    if backend not in ('auto','piper','doubao'):
        raise ValueError('backend must be auto, piper, or doubao')
    if backend != 'auto':
        return backend
    source = os.environ if environ is None else environ
    return ('doubao' if all(str(source.get(name,'')).strip()
                            for name in DOUBAO_ENV_VARS) else 'piper')


class PiperEngine:
    """One instance owned by one worker; voice model loaded once on CPU."""
    def __init__(self, model_path, length_scale=1.08, max_audio_seconds=90.0):
        from piper import PiperVoice, SynthesisConfig
        model = Path(model_path).expanduser()
        if not model.is_file() or not Path(str(model)+'.json').is_file():
            raise FileNotFoundError(f"Missing Piper .onnx / .onnx.json: {model}")
        if not 0.5 <= length_scale <= 2 or max_audio_seconds <= 0:
            raise ValueError("Invalid speech speed or audio limit")
        self.voice = PiperVoice.load(str(model),use_cuda=False)
        self.config = SynthesisConfig(length_scale=length_scale,noise_scale=0.5,noise_w_scale=0.65,normalize_audio=True)
        self.max_audio_seconds = max_audio_seconds

    def synthesize(self, text, cancel):
        pieces, rate, samples = [], None, 0
        for chunk in self.voice.synthesize(text,syn_config=self.config):
            if cancel.is_set():
                raise Cancelled()
            if chunk.sample_channels != 1 or chunk.sample_width != 2:
                raise ValueError("Expected mono 16-bit Piper output")
            if rate is not None and rate != chunk.sample_rate:
                raise ValueError("Sample rate changed within utterance")
            rate = chunk.sample_rate
            piece = np.frombuffer(chunk.audio_int16_bytes,dtype='<i2').astype(np.float32)/32768.0
            samples += len(piece)
            if samples/rate > self.max_audio_seconds:
                raise ValueError("Generated speech exceeds max_audio_seconds")
            pieces.append(piece)
        if cancel.is_set():
            raise Cancelled()
        if rate is None or samples == 0:
            raise ValueError("TTS generated no audio")
        return np.concatenate(pieces),rate


class DoubaoEngine:
    """Volcengine Doubao Voice V3 SSE client returning mono float32 PCM.

    Credentials are accepted by the constructor so callers can load them from
    the environment without ever placing them in ROS parameters.  A Session is
    kept for HTTP keep-alive; the service currently keeps connections alive for
    about one minute.
    """

    DEFAULT_ENDPOINT = (
        'https://openspeech.bytedance.com/api/v3/tts/unidirectional/sse')
    VALID_SAMPLE_RATES = {8000,16000,22050,24000,32000,44100,48000}

    def __init__(self,api_key,voice_type,resource_id,
                 model='seed-tts-2.0-standard',sample_rate=24000,
                 speech_rate=0,loudness_rate=0,endpoint=DEFAULT_ENDPOINT,
                 timeout_seconds=60.0,max_audio_seconds=90.0,session=None):
        if not str(api_key).strip():
            raise ValueError('DOUBAO_TTS_API_KEY is required for backend=doubao')
        if not str(voice_type).strip():
            raise ValueError(
                'DOUBAO_TTS_VOICE_TYPE is required for backend=doubao')
        if not str(resource_id).strip():
            raise ValueError(
                'DOUBAO_TTS_RESOURCE_ID is required for backend=doubao')
        if int(sample_rate) not in self.VALID_SAMPLE_RATES:
            raise ValueError('Unsupported Doubao sample rate')
        if not -50 <= int(speech_rate) <= 100:
            raise ValueError('doubao_speech_rate must be between -50 and 100')
        if not -50 <= int(loudness_rate) <= 100:
            raise ValueError(
                'doubao_loudness_rate must be between -50 and 100')
        if float(timeout_seconds) <= 0 or float(max_audio_seconds) <= 0:
            raise ValueError('Invalid Doubao timeout or audio limit')
        if not str(endpoint).startswith('https://'):
            raise ValueError('Doubao endpoint must use HTTPS')

        self.api_key = str(api_key).strip()
        self.voice_type = str(voice_type).strip()
        self.resource_id = str(resource_id).strip()
        self.model = str(model).strip()
        self.sample_rate = int(sample_rate)
        self.speech_rate = int(speech_rate)
        self.loudness_rate = int(loudness_rate)
        self.endpoint = str(endpoint).strip()
        self.timeout_seconds = float(timeout_seconds)
        self.max_audio_seconds = float(max_audio_seconds)
        if session is None:
            import requests
            session = requests.Session()
        self.session = session

    @classmethod
    def from_environment(cls,**kwargs):
        """Create an engine without copying the API key into ROS config."""
        return cls(
            api_key=os.environ.get('DOUBAO_TTS_API_KEY',''),
            voice_type=os.environ.get('DOUBAO_TTS_VOICE_TYPE',''),
            resource_id=os.environ.get('DOUBAO_TTS_RESOURCE_ID',''),
            **kwargs)

    def _payload(self,text):
        params = {
            'text':text,
            'speaker':self.voice_type,
            'audio_params':{
                'format':'pcm',
                'sample_rate':self.sample_rate,
                'speech_rate':self.speech_rate,
                'loudness_rate':self.loudness_rate,
            },
        }
        if self.model:
            params['model'] = self.model
        return {'user':{'uid':'ghost-game'},'req_params':params}

    def synthesize(self,text,cancel):
        if cancel.is_set():
            raise Cancelled()
        headers = {
            'Content-Type':'application/json',
            'Accept':'text/event-stream',
            'X-Api-Key':self.api_key,
            'X-Api-Resource-Id':self.resource_id,
            'X-Api-Request-Id':str(uuid.uuid4()),
        }
        try:
            response = self.session.post(
                self.endpoint,headers=headers,json=self._payload(text),
                stream=True,timeout=self.timeout_seconds)
        except Exception as exc:
            # Never interpolate headers or the credential into diagnostics.
            raise RuntimeError(
                f'Doubao TTS request failed: {type(exc).__name__}') from exc

        audio = bytearray()
        saw_success = False
        try:
            status = int(getattr(response,'status_code',0))
            if status < 200 or status >= 300:
                log_id = str(getattr(response,'headers',{}).get(
                    'X-Tt-Logid','')).strip()
                suffix = f' (logid={log_id})' if log_id else ''
                raise RuntimeError(f'Doubao TTS HTTP {status}{suffix}')
            for raw_line in response.iter_lines():
                if cancel.is_set():
                    raise Cancelled()
                if isinstance(raw_line,str):
                    line = raw_line.strip()
                else:
                    line = bytes(raw_line).decode(
                        'utf-8',errors='strict').strip()
                if not line or line.startswith(':'):
                    continue
                if not line.startswith('data:'):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except (json.JSONDecodeError,UnicodeDecodeError) as exc:
                    raise RuntimeError('Invalid Doubao TTS SSE response') from exc
                code = int(event.get('code',0))
                if code == 20000000:
                    saw_success = True
                    continue
                if code != 0:
                    message = str(event.get('message','')).replace('\n',' ')[:200]
                    raise RuntimeError(
                        f'Doubao TTS API error {code}: {message}')
                encoded = event.get('data')
                if not encoded:
                    continue
                try:
                    piece = base64.b64decode(encoded,validate=True)
                except (ValueError,TypeError) as exc:
                    raise RuntimeError('Invalid Doubao TTS audio data') from exc
                audio.extend(piece)
                if len(audio)/(2*self.sample_rate) > self.max_audio_seconds:
                    raise ValueError(
                        'Generated speech exceeds max_audio_seconds')
            if cancel.is_set():
                raise Cancelled()
        finally:
            close = getattr(response,'close',None)
            if close is not None:
                close()

        if not saw_success:
            raise RuntimeError('Doubao TTS stream ended before success event')
        if not audio:
            raise ValueError('TTS generated no audio')
        if len(audio)%2:
            raise RuntimeError('Doubao TTS returned malformed 16-bit PCM')
        samples = np.frombuffer(audio,dtype='<i2').astype(np.float32)/32768.0
        return samples,self.sample_rate


def write_wav(path,audio,rate):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    pcm = np.round(np.clip(audio,-1,1)*32767).astype('<i2')
    with wave.open(str(path),'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm.tobytes())


def play_audio(audio,rate,cancel,device=None):
    """Write small chunks so cancellation stops playback promptly.

    Actual cancellation latency depends on PortAudio/OS/hardware buffering.
    Automatically resample if the device does not accept the voice sample rate.
    """
    if isinstance(device, str) and (
            device == 'pulse' or device.startswith('pulse:')):
        sink = None if device == 'pulse' else device.split(':', 1)[1]
        return _play_audio_pulse(audio, rate, cancel, sink)

    import sounddevice as sd
    from .effects import lowpass
    if cancel.is_set():
        raise Cancelled()
    try:
        sd.check_output_settings(device=device,channels=1,dtype='float32',samplerate=rate)
    except sd.PortAudioError:
        target_rate = int(round(sd.query_devices(device,kind='output')['default_samplerate']))
        if target_rate <= 0:
            raise ValueError('Output device has no valid default sample rate')
        if target_rate < rate:
            audio = lowpass(audio, rate, target_rate * 0.45, taps=257)
        count = int(np.ceil(len(audio) * target_rate / rate))
        audio = np.interp(np.arange(count) * rate / target_rate,
                          np.arange(len(audio)), audio).astype(np.float32)
        rate = target_rate
        sd.check_output_settings(device=device,channels=1,dtype='float32',samplerate=rate)
    block = max(1,int(rate*0.02))
    stream = sd.OutputStream(samplerate=rate,channels=1,dtype='float32',device=device,latency='low')
    try:
        stream.start()
        for start in range(0,len(audio),block):
            if cancel.is_set():
                raise Cancelled()
            stream.write(audio[start:start+block].reshape(-1,1))
        if cancel.is_set():
            raise Cancelled()
        stream.stop()  # Drain the final samples before reporting done.
    finally:
        if cancel.is_set():
            stream.abort()
        stream.close()


class _PulseSampleSpec(ctypes.Structure):
    _fields_ = [
        ('format', ctypes.c_int),
        ('rate', ctypes.c_uint32),
        ('channels', ctypes.c_uint8),
    ]


def _play_audio_pulse(audio, rate, cancel, sink=None):
    """Play mono float32 samples through the host PulseAudio/PipeWire sink.

    This backend is used by the devcontainer because its direct ALSA devices
    are NVIDIA HDMI ports, while the actual show speaker is managed by the
    host PipeWire session (currently Bluetooth). Pulse performs any required
    sample-rate/channel conversion for that sink.
    """
    library_path = ctypes.util.find_library('pulse-simple')
    if not library_path:
        raise RuntimeError(
            'PulseAudio backend unavailable: libpulse-simple was not found')
    pulse = ctypes.CDLL(library_path)
    pulse.pa_simple_new.restype = ctypes.c_void_p
    pulse.pa_simple_new.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_char_p, ctypes.POINTER(_PulseSampleSpec), ctypes.c_void_p,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
    ]
    pulse.pa_simple_write.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int),
    ]
    pulse.pa_simple_write.restype = ctypes.c_int
    pulse.pa_simple_drain.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    pulse.pa_simple_drain.restype = ctypes.c_int
    pulse.pa_simple_flush.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    pulse.pa_simple_flush.restype = ctypes.c_int
    pulse.pa_simple_free.argtypes = [ctypes.c_void_p]
    pulse.pa_simple_free.restype = None
    pulse.pa_strerror.argtypes = [ctypes.c_int]
    pulse.pa_strerror.restype = ctypes.c_char_p

    def error_text(code):
        raw = pulse.pa_strerror(code)
        return raw.decode('utf-8', errors='replace') if raw else str(code)

    # pa_sample_format_t: PA_SAMPLE_FLOAT32LE=5; pa_stream_direction_t:
    # PA_STREAM_PLAYBACK=1.
    sample_spec = _PulseSampleSpec(5, int(rate), 1)
    error = ctypes.c_int()
    sink_bytes = None if not sink else sink.encode('utf-8')
    server = _pulse_server_address()
    server_bytes = None if server is None else server.encode('utf-8')
    handle = pulse.pa_simple_new(
        server_bytes, b'ghost-tts', 1, sink_bytes, b'Ghost voice',
        ctypes.byref(sample_spec), None, None, ctypes.byref(error))
    if not handle:
        raise RuntimeError(
            'PulseAudio connection failed: '
            f'{error_text(error.value)} (server={server or "default"}); '
            'check PULSE_SERVER and the host sink')

    samples = np.ascontiguousarray(audio, dtype=np.float32)
    block = max(1, int(rate * 0.02))
    try:
        for start in range(0, len(samples), block):
            if cancel.is_set():
                pulse.pa_simple_flush(handle, ctypes.byref(error))
                raise Cancelled()
            chunk = np.ascontiguousarray(samples[start:start + block])
            result = pulse.pa_simple_write(
                handle, ctypes.c_void_p(chunk.ctypes.data), chunk.nbytes,
                ctypes.byref(error))
            if result < 0:
                raise RuntimeError(
                    f'PulseAudio write failed: {error_text(error.value)}')
        if cancel.is_set():
            pulse.pa_simple_flush(handle, ctypes.byref(error))
            raise Cancelled()
        if pulse.pa_simple_drain(handle, ctypes.byref(error)) < 0:
            raise RuntimeError(
                f'PulseAudio drain failed: {error_text(error.value)}')
    finally:
        pulse.pa_simple_free(handle)


def _pulse_server_address():
    """Resolve the host audio server for both fresh and current containers."""
    configured = os.environ.get('PULSE_SERVER', '').strip()
    if configured:
        return configured
    socket_path = '/run/user/1000/pulse/native'
    try:
        if stat.S_ISSOCK(os.stat(socket_path).st_mode):
            return f'unix:{socket_path}'
    except OSError:
        pass
    # The current hackathon container predates the devcontainer socket mount.
    # It shares the host network and reaches the loopback-only Pulse bridge.
    return 'tcp:127.0.0.1'


def device_argument(value):
    if value is None or str(value).strip() in ('','default'):
        return None
    try:
        return int(value)
    except ValueError:
        return value
