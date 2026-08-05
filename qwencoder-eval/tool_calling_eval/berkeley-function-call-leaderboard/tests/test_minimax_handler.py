import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from bfcl_eval.model_handler.api_inference.minimax import (
    MINIMAX_BASE_URLS,
    MiniMaxAnthropicHandler,
    MiniMaxHandler,
    MiniMaxOpenAIHandler,
)


class _RequestCaptureHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.requests.append((self.path, json.loads(body)))

        if self.path.endswith("/chat/completions"):
            response = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "MiniMax-M3",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        else:
            response = {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "MiniMax-M3",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return


class MiniMaxHandlerTest(unittest.TestCase):
    def setUp(self):
        _RequestCaptureHandler.requests = []

    def test_default_endpoint_matrix(self):
        for api_format, regions in MINIMAX_BASE_URLS.items():
            for region, expected_url in regions.items():
                with self.subTest(api_format=api_format, region=region):
                    env = {
                        "MINIMAX_API_KEY": "test",
                        "MINIMAX_API_FORMAT": api_format,
                        "MINIMAX_REGION": region,
                    }
                    with patch.dict(os.environ, env, clear=True):
                        handler = MiniMaxHandler("MiniMax-M3", 0.001)
                    self.assertEqual(
                        str(handler.client.base_url).rstrip("/"), expected_url
                    )
                    if api_format == "openai":
                        self.assertIsInstance(handler, MiniMaxOpenAIHandler)
                    else:
                        self.assertIsInstance(handler, MiniMaxAnthropicHandler)
                    handler.client.close()

    def test_sdk_request_paths_and_request_options(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _RequestCaptureHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        host = f"http://127.0.0.1:{server.server_port}"

        try:
            openai_env = {
                "MINIMAX_API_KEY": "test",
                "MINIMAX_API_FORMAT": "openai",
                "MINIMAX_BASE_URL": f"{host}/v1",
                "MINIMAX_THINKING": "disabled",
                "MINIMAX_SERVICE_TIER": "priority",
            }
            with patch.dict(os.environ, openai_env, clear=True):
                openai_handler = MiniMaxHandler("MiniMax-M3", 0.001)
                openai_handler.generate_with_backoff(
                    model="MiniMax-M3",
                    messages=[{"role": "user", "content": "test"}],
                )
                openai_handler.client.close()

            anthropic_env = {
                "MINIMAX_API_KEY": "test",
                "MINIMAX_API_FORMAT": "anthropic",
                "MINIMAX_BASE_URL": f"{host}/anthropic",
                "MINIMAX_THINKING": "adaptive",
                "MINIMAX_SERVICE_TIER": "standard",
            }
            with patch.dict(os.environ, anthropic_env, clear=True):
                anthropic_handler = MiniMaxHandler("MiniMax-M3", 0.001)
                anthropic_handler.generate_with_backoff(
                    model="MiniMax-M3",
                    max_tokens=16,
                    messages=[{"role": "user", "content": "test"}],
                )
                anthropic_handler.client.close()
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join()

        self.assertEqual(
            [path for path, _ in _RequestCaptureHandler.requests],
            ["/v1/chat/completions", "/anthropic/v1/messages"],
        )
        self.assertEqual(
            _RequestCaptureHandler.requests[0][1]["thinking"], {"type": "disabled"}
        )
        self.assertEqual(
            _RequestCaptureHandler.requests[0][1]["service_tier"], "priority"
        )
        self.assertEqual(
            _RequestCaptureHandler.requests[1][1]["thinking"], {"type": "adaptive"}
        )
        self.assertEqual(
            _RequestCaptureHandler.requests[1][1]["service_tier"], "standard"
        )

    def test_rejects_invalid_configuration(self):
        cases = (
            {"MINIMAX_API_FORMAT": "invalid"},
            {"MINIMAX_API_FORMAT": "openai", "MINIMAX_REGION": "invalid"},
            {
                "MINIMAX_API_FORMAT": "anthropic",
                "MINIMAX_BASE_URL": "https://api.minimax.io/v1",
            },
        )
        for env in cases:
            with self.subTest(env=env):
                with patch.dict(
                    os.environ, {"MINIMAX_API_KEY": "test", **env}, clear=True
                ):
                    with self.assertRaises(ValueError):
                        MiniMaxHandler("MiniMax-M3", 0.001)

    def test_m27_thinking_cannot_be_changed(self):
        for thinking in ("adaptive", "disabled"):
            with self.subTest(thinking=thinking):
                env = {
                    "MINIMAX_API_KEY": "test",
                    "MINIMAX_THINKING": thinking,
                }
                with patch.dict(os.environ, env, clear=True):
                    handler = MiniMaxHandler("MiniMax-M2.7", 0.001)
                    with self.assertRaisesRegex(ValueError, "always on"):
                        handler.generate_with_backoff(
                            model="MiniMax-M2.7",
                            messages=[{"role": "user", "content": "test"}],
                        )
                    handler.client.close()


if __name__ == "__main__":
    unittest.main()
