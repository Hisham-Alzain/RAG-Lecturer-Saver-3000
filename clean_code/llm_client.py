from openai import OpenAI
from typing import List, Dict
from config import Config

class LLMClient:
    """LLM API client with provider-specific handling."""

    def __init__(self, provider_key: str, api_key: str):
        config = Config.LLM_PROVIDERS[provider_key]
        self.model = config["model"]
        self.provider_name = config["name"]

        # OpenRouter requires additional headers
        headers = {}
        if "openrouter" in config["base_url"].lower():
            headers = {
                "HTTP-Referer": "https://rag-tutor.streamlit.app",
                "X-Title": "RAG Tutor",
            }

        self.client = OpenAI(
            base_url=config["base_url"],
            api_key=api_key,
            default_headers=headers if headers else None,
        )

    def chat(
        self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2048
    ) -> str:
        """Send chat completion request."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"