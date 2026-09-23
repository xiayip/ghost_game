"""Deterministic voice effects; no audio/network/device side effects."""
import math
import numpy as np


def lowpass(audio, sample_rate, cutoff, taps=129):
    """Windowed-sinc FIR, centered to avoid shifting the dry/wet branches."""
    offset = np.arange(taps) - (taps - 1) / 2
    kernel = 2 * cutoff / sample_rate * np.sinc(2 * cutoff / sample_rate * offset)
    kernel *= np.hamming(taps)
    kernel /= kernel.sum()
    start = (taps - 1) // 2
    return np.convolve(audio, kernel, mode='full')[start:start + len(audio)]

PRESETS = {
    "clean": dict(ring=0.0, chorus=0.0, echo=0.0, crush=0.0),
    "subtle": dict(ring=0.10, chorus=0.10, echo=0.08, crush=0.0),
    "ghost": dict(ring=0.24, chorus=0.16, echo=0.13, crush=0.06),
    "machine": dict(ring=0.48, chorus=0.20, echo=0.16, crush=0.15),
}


def cyber_effect(audio, sample_rate, preset="ghost", strength=1.0, volume=0.7):
    if preset not in PRESETS:
        raise ValueError("preset must be clean/subtle/ghost/machine")
    if not 0 <= strength <= 1 or not 0 <= volume <= 1:
        raise ValueError("strength and volume must be in [0,1]")
    if sample_rate < 8000:
        raise ValueError("sample_rate must be >= 8000")
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("audio must be finite mono samples")
    if not x.size:
        return x.astype(np.float32)
    wet = {k:v*strength for k,v in PRESETS[preset].items()}
    if not any(wet.values()):
        y = x.copy()
    else:
        # Band-limited parallel wet branch; intelligible dry voice remains.
        filtered = lowpass(x, sample_rate, min(5800, sample_rate*0.42)) - lowpass(x, sample_rate, 90)
        t = np.arange(x.size)/sample_rate
        ring = filtered*np.sin(2*math.pi*47*t)
        position = np.arange(x.size) - sample_rate*(0.013+0.003*np.sin(2*math.pi*0.65*t))
        chorus = np.interp(position,np.arange(x.size),filtered,left=0,right=0)
        quantized = np.round(np.clip(filtered,-1,1)*255)/255
        y = (1-wet['ring'])*x + wet['ring']*ring + wet['chorus']*chorus + wet['crush']*quantized
        if wet['echo']:
            delay = int(0.095*sample_rate)
            base = y.copy()
            y = np.pad(y,(0,delay*3))
            for count in (1,2,3):
                start = count*delay
                y[start:start+base.size] += (wet['echo']**count)*base
    # Attenuate peaks only; never amplify silence/noise to full scale.
    peak = float(np.max(np.abs(y)))
    if peak > 0.92:
        y *= 0.92/peak
    y *= volume
    fade = min(int(sample_rate*0.005),len(y)//2)
    if fade:
        y[:fade] *= np.linspace(0,1,fade)
        y[-fade:] *= np.linspace(1,0,fade)
    return np.clip(y,-0.98,0.98).astype(np.float32)
