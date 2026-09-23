from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any, Dict
from urllib.parse import urlparse

import requests


class DeepSeekError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeepSeekResponse:
    content: str
    model: str
    finish_reason: str
    usage: Dict[str, Any]
    elapsed_sec: float


class DeepSeekVisionClient:
    def __init__(
        self,
        *,
        api_key: str,
        api_base_url: str,
        model: str,
        image_detail: str,
        timeout_sec: float,
        max_retries: int,
        retry_interval_sec: float,
        max_tokens: int,
        temperature: float,
        session: requests.Session | None = None,
    ) -> None:
        parsed = urlparse(api_base_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("api_base_url 必须是不含凭据的 HTTPS 地址")
        if image_detail not in {"low", "high", "original", "auto"}:
            raise ValueError("image_detail 必须是 low/high/original/auto")
        self._api_key = api_key
        self._endpoint = api_base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._image_detail = image_detail
        self._timeout_sec = timeout_sec
        self._max_retries = max(0, max_retries)
        self._retry_interval_sec = max(0.0, retry_interval_sec)
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._session = session or requests.Session()

    def edit(self, image_jpeg: bytes, system_prompt: str, card_prompt: str) -> DeepSeekResponse:
        image_url = "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode("ascii")
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": card_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url, "detail": self._image_detail},
                        },
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        started = time.monotonic()
        last_error = "未知错误"
        for attempt in range(self._max_retries + 1):
            try:
                response = self._session.post(
                    self._endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout_sec,
                )
                if response.status_code >= 400:
                    detail = _safe_error_detail(response)
                    last_error = f"HTTP {response.status_code}: {detail}"
                    retryable = response.status_code == 429 or response.status_code >= 500
                    if not retryable or attempt >= self._max_retries:
                        raise DeepSeekError(last_error)
                else:
                    data = response.json()
                    choice = data["choices"][0]
                    content = choice["message"]["content"]
                    finish_reason = str(choice.get("finish_reason") or "")
                    if finish_reason == "length":
                        last_error = (
                            "DeepSeek 输出达到 token 上限，JSON 可能被截断"
                        )
                        if attempt >= self._max_retries:
                            raise DeepSeekError(last_error)
                        # A vision model may spend part of the completion
                        # budget on internal reasoning. Retry only truncated
                        # responses, with enough room to close the JSON.
                        current_limit = int(payload["max_tokens"])
                        payload["max_tokens"] = min(
                            max(current_limit * 2, current_limit + 512), 4096)
                        continue
                    if not isinstance(content, str) or not content.strip():
                        last_error = "DeepSeek 返回了空内容"
                        if attempt >= self._max_retries:
                            raise DeepSeekError(last_error)
                    else:
                        return DeepSeekResponse(
                            content=content,
                            model=str(data.get("model") or self._model),
                            finish_reason=finish_reason,
                            usage=data.get("usage") or {},
                            elapsed_sec=time.monotonic() - started,
                        )
            except DeepSeekError:
                raise
            except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
                last_error = f"DeepSeek 请求或响应解析失败: {exc}"
                if attempt >= self._max_retries:
                    raise DeepSeekError(last_error) from exc
            if self._retry_interval_sec:
                time.sleep(self._retry_interval_sec * (attempt + 1))
        raise DeepSeekError(last_error)


def _safe_error_detail(response: requests.Response) -> str:
    try:
        data = response.json()
        detail = (data.get("error") or {}).get("message")
        if detail:
            return str(detail)[:500]
    except (ValueError, AttributeError, TypeError):
        pass
    return (response.text or "请求失败")[:500]
