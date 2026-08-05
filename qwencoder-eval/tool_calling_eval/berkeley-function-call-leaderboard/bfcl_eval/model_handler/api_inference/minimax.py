import os
from typing import Any

from anthropic import Anthropic
from anthropic.types import TextBlock
from openai import OpenAI
from overrides import override

from bfcl_eval.model_handler.api_inference.claude import ClaudeHandler
from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.model_style import ModelStyle

MINIMAX_BASE_URLS = {
    "openai": {
        "global": "https://api.minimax.io/v1",
        "cn": "https://api.minimaxi.com/v1",
    },
    "anthropic": {
        "global": "https://api.minimax.io/anthropic",
        "cn": "https://api.minimaxi.com/anthropic",
    },
}


def _get_api_format() -> str:
    api_format = os.getenv("MINIMAX_API_FORMAT", "openai").lower()
    if api_format not in MINIMAX_BASE_URLS:
        raise ValueError("MINIMAX_API_FORMAT must be 'openai' or 'anthropic'.")
    return api_format


def _get_base_url(api_format: str) -> str:
    region = os.getenv("MINIMAX_REGION", "global").lower()
    if region not in MINIMAX_BASE_URLS[api_format]:
        raise ValueError("MINIMAX_REGION must be 'global' or 'cn'.")

    base_url = os.getenv("MINIMAX_BASE_URL") or MINIMAX_BASE_URLS[api_format][region]
    base_url = base_url.rstrip("/")
    if api_format == "anthropic" and not base_url.endswith("/anthropic"):
        raise ValueError(
            "An Anthropic-compatible MiniMax base URL must end with '/anthropic'."
        )
    return base_url


def _get_extra_body(model_name: str) -> dict[str, Any]:
    extra_body: dict[str, Any] = {}

    thinking = os.getenv("MINIMAX_THINKING", "").lower()
    if thinking:
        if thinking not in {"adaptive", "disabled"}:
            raise ValueError("MINIMAX_THINKING must be 'adaptive' or 'disabled'.")
        if model_name.startswith("MiniMax-M2.7"):
            raise ValueError("MiniMax-M2.7 thinking is always on and cannot be changed.")
        extra_body["thinking"] = {"type": thinking}

    service_tier = os.getenv("MINIMAX_SERVICE_TIER", "").lower()
    if service_tier:
        if service_tier not in {"standard", "priority"}:
            raise ValueError("MINIMAX_SERVICE_TIER must be 'standard' or 'priority'.")
        extra_body["service_tier"] = service_tier

    return extra_body


class MiniMaxOpenAIHandler(OpenAICompletionsHandler):
    def __init__(self, model_name, temperature) -> None:
        BaseHandler.__init__(self, model_name, temperature)
        self.model_style = ModelStyle.OpenAI_Completions
        self.client = OpenAI(
            base_url=_get_base_url("openai"),
            api_key=os.getenv("MINIMAX_API_KEY"),
        )

    @override
    def generate_with_backoff(self, **kwargs):
        kwargs.pop("store", None)
        extra_body = _get_extra_body(self.model_name)
        if extra_body:
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), **extra_body}
        return super().generate_with_backoff(**kwargs)


class MiniMaxAnthropicHandler(ClaudeHandler):
    def __init__(self, model_name, temperature) -> None:
        BaseHandler.__init__(self, model_name, temperature)
        self.model_style = ModelStyle.Anthropic
        self.client = Anthropic(
            base_url=_get_base_url("anthropic"),
            api_key=os.getenv("MINIMAX_API_KEY"),
        )

    @override
    def generate_with_backoff(self, **kwargs):
        extra_body = _get_extra_body(self.model_name)
        if extra_body:
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), **extra_body}
        return super().generate_with_backoff(**kwargs)

    @override
    def _get_max_tokens(self):
        max_tokens = int(os.getenv("MINIMAX_MAX_TOKENS", "4096"))
        if max_tokens <= 0:
            raise ValueError("MINIMAX_MAX_TOKENS must be a positive integer.")
        return max_tokens

    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        response_data = super()._parse_query_response_FC(api_response)
        reasoning_content = "\n".join(
            block.thinking
            for block in api_response.content
            if getattr(block, "type", None) == "thinking"
        )
        if reasoning_content:
            response_data["reasoning_content"] = reasoning_content
        return response_data

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        model_response = "".join(
            block.text for block in api_response.content if isinstance(block, TextBlock)
        )
        response_data = {
            "model_responses": model_response,
            "model_responses_message_for_chat_history": api_response.content,
            "input_token": api_response.usage.input_tokens,
            "output_token": api_response.usage.output_tokens,
        }
        reasoning_content = "\n".join(
            block.thinking
            for block in api_response.content
            if getattr(block, "type", None) == "thinking"
        )
        if reasoning_content:
            response_data["reasoning_content"] = reasoning_content
        return response_data

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            {
                "role": "assistant",
                "content": model_response_data[
                    "model_responses_message_for_chat_history"
                ],
            }
        )
        return inference_data


class MiniMaxHandler:
    def __new__(cls, model_name, temperature):
        handler_class = (
            MiniMaxOpenAIHandler
            if _get_api_format() == "openai"
            else MiniMaxAnthropicHandler
        )
        return handler_class(model_name, temperature)
