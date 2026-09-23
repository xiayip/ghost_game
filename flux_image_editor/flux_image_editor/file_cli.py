"""File-to-file command line client for a running FLUX service."""

import argparse
import time
from pathlib import Path

from flux_image_editor.client import EditorClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_png', type=Path)
    parser.add_argument('prompt')
    parser.add_argument('--server-url', default='http://127.0.0.1:8090')
    parser.add_argument('--output', type=Path, default=Path('flux_output.png'))
    parser.add_argument('--timeout', type=float, default=300.0)
    args = parser.parse_args()
    data = args.input_png.read_bytes()
    if not data.startswith(b'\x89PNG\r\n\x1a\n'):
        raise SystemExit(f'{args.input_png} 不是 PNG 文件')
    client = EditorClient(args.server_url, args.timeout)
    started = time.monotonic()
    try:
        if not client.ready():
            raise SystemExit(f'推理服务尚未就绪: {args.server_url}')
        result = client.edit(data, args.prompt)
    finally:
        client.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(result.png)
    print(f'output={args.output.resolve()}')
    print(f'size={result.width}x{result.height}')
    print(f'inference_sec={result.inference_sec:.3f}')
    print(f'model_load_sec={result.model_load_sec:.3f}')
    print(f'seed={result.seed}')
    print(f'total_sec={time.monotonic() - started:.3f}')


if __name__ == '__main__':
    main()
