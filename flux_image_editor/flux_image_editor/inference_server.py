"""FastAPI service that keeps FLUX.2 [klein] 4B resident on DGX Spark."""

import io
import math
import random
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from flux_image_editor.config import load_server_settings


class FluxEditorEngine:
    def __init__(self, settings):
        self.settings = settings
        self.pipeline = None
        self.torch = None
        self.load_sec = 0.0
        self.lock = threading.Lock()

    def load(self):
        started = time.monotonic()
        import torch
        from diffusers import Flux2KleinPipeline

        dtype_name = str(self.settings.get('dtype', 'bfloat16'))
        dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f'不支持的 dtype: {dtype_name}')
        device = str(self.settings.get('device', 'cuda'))
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('配置要求 CUDA，但 torch.cuda.is_available() 为 false')
        pipeline = Flux2KleinPipeline.from_pretrained(
            str(self.settings['model_id']), torch_dtype=dtype_map[dtype_name],
        )
        if bool(self.settings.get('enable_cpu_offload', False)):
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)
        self.pipeline = pipeline
        self.torch = torch
        self.load_sec = time.monotonic() - started

    def _target_size(self, source):
        width, height = source.size
        max_edge = int(self.settings.get('max_long_edge', 1024))
        max_pixels = float(self.settings.get('max_megapixels', 1.1)) * 1_000_000
        multiple = int(self.settings.get('dimension_multiple', 16))
        if min(width, height) <= 0 or max_edge < multiple or max_pixels < multiple * multiple:
            raise ValueError('图像尺寸或 server.yaml 的尺寸限制无效')
        scale = min(1.0, max_edge / max(width, height), math.sqrt(max_pixels / (width * height)))
        target_width = max(multiple, int(width * scale) // multiple * multiple)
        target_height = max(multiple, int(height * scale) // multiple * multiple)
        return int(target_width), int(target_height)

    def edit(self, source, prompt):
        if self.pipeline is None:
            raise RuntimeError('模型尚未加载')
        width, height = self._target_size(source)
        source = source.convert('RGB').resize((width, height), Image.Resampling.LANCZOS)
        configured_seed = int(self.settings.get('seed', -1))
        seed = configured_seed if configured_seed >= 0 else random.SystemRandom().randrange(0, 2**63)
        device = str(self.settings.get('device', 'cuda'))
        generator = self.torch.Generator(device=device).manual_seed(seed)
        started = time.monotonic()
        with self.lock, self.torch.inference_mode():
            output = self.pipeline(
                image=source,
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=int(self.settings.get('inference_steps', 4)),
                guidance_scale=float(self.settings.get('guidance_scale', 1.0)),
                generator=generator,
            ).images[0]
        inference_sec = time.monotonic() - started
        buffer = io.BytesIO()
        output.convert('RGB').save(buffer, format='PNG', optimize=False)
        return buffer.getvalue(), output.width, output.height, inference_sec, seed


SETTINGS = load_server_settings()
ENGINE = FluxEditorEngine(SETTINGS)


@asynccontextmanager
async def lifespan(_app):
    print(f'Loading {SETTINGS["model_id"]} ...', flush=True)
    ENGINE.load()
    print(f'Model ready in {ENGINE.load_sec:.3f} seconds', flush=True)
    yield


app = FastAPI(title='FLUX.2 Klein Image Editor', version='0.1.0', lifespan=lifespan)


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/ready')
def ready():
    ready_state = ENGINE.pipeline is not None
    return {
        'ready': ready_state,
        'model_id': SETTINGS.get('model_id', ''),
        'model_load_sec': ENGINE.load_sec,
    }


@app.post('/v1/edit')
def edit_image(
    prompt: str = Form(...), request_id: str = Form(''),
    image: UploadFile = File(...),
):
    request_id = request_id.strip()[:128] or uuid.uuid4().hex
    received_at = time.monotonic()
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail='prompt 不能为空')
    if len(prompt) > int(SETTINGS.get('max_prompt_chars', 2000)):
        raise HTTPException(status_code=413, detail='prompt 过长')
    data = image.file.read(int(SETTINGS.get('max_input_bytes', 20_000_000)) + 1)
    if len(data) > int(SETTINGS.get('max_input_bytes', 20_000_000)):
        raise HTTPException(status_code=413, detail='输入 PNG 超过大小限制')
    if not data.startswith(b'\x89PNG\r\n\x1a\n'):
        raise HTTPException(status_code=415, detail='仅接受 PNG 输入')
    print(
        f'FLUX request received request_id={request_id} '
        f'png_bytes={len(data)} prompt_chars={len(prompt)}', flush=True,
    )
    try:
        source = Image.open(io.BytesIO(data))
        source.load()
        output, width, height, inference_sec, seed = ENGINE.edit(source, prompt)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    total_sec = time.monotonic() - received_at
    print(
        f'FLUX request completed request_id={request_id} size={width}x{height} '
        f'inference_sec={inference_sec:.6f} server_total_sec={total_sec:.6f} '
        f'seed={seed} png_bytes={len(output)}', flush=True,
    )
    return Response(
        content=output,
        media_type='image/png',
        headers={
            'X-Inference-Sec': f'{inference_sec:.6f}',
            'X-Output-Width': str(width),
            'X-Output-Height': str(height),
            'X-Seed': str(seed),
            'X-Model-Load-Sec': f'{ENGINE.load_sec:.6f}',
            'X-Request-ID': request_id,
            'X-Server-Total-Sec': f'{total_sec:.6f}',
        },
    )


def main():
    import uvicorn

    uvicorn.run(
        app,
        host=str(SETTINGS.get('bind_host', '0.0.0.0')),
        port=int(SETTINGS.get('port', 8090)),
        log_level='info',
    )


if __name__ == '__main__':
    main()
