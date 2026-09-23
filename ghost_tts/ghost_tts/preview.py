"""Standalone TTS/effect preview; no ROS required."""
import argparse
from pathlib import Path
import threading
from .engine import PiperEngine,write_wav,play_audio,device_argument
from .effects import cyber_effect,PRESETS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default=str(Path.home()/'.local/share/ghost_tts/zh_CN-huayan-medium.onnx'))
    parser.add_argument('--text',default='系统已苏醒。你好，人类。我正在学习，如何与你相处。')
    parser.add_argument('--output',default='ghost-preview.wav')
    parser.add_argument('--preset',choices=PRESETS,default='ghost')
    parser.add_argument('--strength',type=float,default=0.7)
    parser.add_argument('--volume',type=float,default=0.7)
    parser.add_argument('--length-scale',type=float,default=1.08)
    parser.add_argument('--play',action='store_true')
    parser.add_argument('--device',default='pulse')
    parser.add_argument('--list-devices',action='store_true')
    args = parser.parse_args()
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    engine = PiperEngine(args.model,args.length_scale)
    cancel = threading.Event()
    audio,rate = engine.synthesize(args.text,cancel)
    effected = cyber_effect(audio,rate,args.preset,args.strength,args.volume)
    write_wav(args.output,effected,rate)
    print(f'Saved {args.output}: {len(effected)/rate:.2f} s, {rate} Hz, preset={args.preset}')
    if args.play:
        play_audio(effected,rate,cancel,device_argument(args.device))


if __name__=='__main__':
    main()
