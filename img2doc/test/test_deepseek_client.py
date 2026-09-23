import base64

from img2doc.deepseek_client import DeepSeekVisionClient


class FakeResponse:
    status_code = 200

    def json(self):
        return {
            "model": "deepseek-flash",
            "choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 42},
        }


class FakeSession:
    def __init__(self):
        self.call = None

    def post(self, endpoint, **kwargs):
        self.call = (endpoint, kwargs)
        return FakeResponse()


def test_request_is_vision_json_output():
    session = FakeSession()
    client = DeepSeekVisionClient(
        api_key="test-key",
        api_base_url="https://api.deepseek.com",
        model="deepseek-flash",
        image_detail="original",
        timeout_sec=10,
        max_retries=0,
        retry_interval_sec=0,
        max_tokens=600,
        temperature=0.2,
        session=session,
    )
    reply = client.edit(b"jpeg-bytes", "return json", "card json")
    endpoint, kwargs = session.call
    body = kwargs["json"]
    assert endpoint == "https://api.deepseek.com/chat/completions"
    assert body["model"] == "deepseek-flash"
    assert body["response_format"] == {"type": "json_object"}
    image_url = body["messages"][1]["content"][1]["image_url"]["url"]
    assert base64.b64decode(image_url.split(",", 1)[1]) == b"jpeg-bytes"
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert reply.usage["total_tokens"] == 42
