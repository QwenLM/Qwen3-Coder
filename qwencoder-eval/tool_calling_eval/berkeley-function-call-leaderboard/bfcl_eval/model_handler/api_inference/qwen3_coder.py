import json
import os
import time

from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler as OpenAIHandler,
)
from bfcl_eval.model_handler.model_style import ModelStyle
from bfcl_eval.model_handler.utils import retry_with_backoff
from openai import OpenAI, RateLimitError
import litellm
from tenacity import stop_after_attempt

error_list = [
    RateLimitError, 
    json.JSONDecodeError,
    litellm.exceptions.BadRequestError,
    litellm.exceptions.APIError,
    litellm.exceptions.InternalServerError,
    litellm.exceptions.Timeout,
]

class Qwen3CoderHandler(OpenAIHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OpenAI_Completions
        self.model_name = model_name
        base_url, api_key = os.getenv("OPENAI_API_BASE"), os.getenv("OPENAI_API_KEY")
        litellm.api_base = base_url
        litellm.api_key = api_key


    @retry_with_backoff(error_type=error_list, 
    min_wait=2, 
    max_wait=10,
    stop=stop_after_attempt(50))
    def generate_with_backoff(self, **kwargs):
        kwargs["custom_llm_provider"] = "openai"
        start_time = time.time()
        api_response = litellm.completion(**kwargs)
        end_time = time.time()

        return api_response, end_time - start_time