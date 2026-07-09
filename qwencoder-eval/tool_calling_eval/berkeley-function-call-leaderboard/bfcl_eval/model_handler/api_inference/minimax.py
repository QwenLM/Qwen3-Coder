import os

from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from bfcl_eval.model_handler.model_style import ModelStyle
from openai import OpenAI


class MiniMaxHandler(OpenAICompletionsHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OpenAI_Completions
        self.client = OpenAI(
            base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1"),
            api_key=os.getenv("MINIMAX_API_KEY"),
        )
