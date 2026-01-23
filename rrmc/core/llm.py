"""
LLM Wrapper for RRMC using OpenRouter API.
"""

import os
import json
import re
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class LLMResponse:
    """Structured response from LLM."""
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMWrapper:
    """
    Wrapper for LLM inference using OpenRouter API.
    Supports Qwen3-8B and other models via OpenRouter.
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen/qwen3-8b",
        base_url: Optional[str] = None,
        default_temperature: float = 0.7,
        default_top_p: float = 0.95,
    ):
        """
        Initialize LLM wrapper.

        Args:
            api_key: OpenRouter API key (or reads from OPENROUTER_API_KEY env)
            model: Model identifier (default: qwen/qwen3-8b)
            base_url: API base URL (default: OpenRouter)
            default_temperature: Default sampling temperature
            default_top_p: Default nucleus sampling parameter
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set OPENROUTER_API_KEY or pass api_key.")

        self.model = model
        self.base_url = base_url or self.OPENROUTER_BASE_URL
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        # Token tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: int = 512,
        json_format: bool = False,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (uses default if None)
            top_p: Nucleus sampling parameter (uses default if None)
            max_tokens: Maximum tokens to generate
            json_format: Whether to request JSON output format

        Returns:
            LLMResponse with content and token counts
        """
        temperature = temperature if temperature is not None else self.default_temperature
        top_p = top_p if top_p is not None else self.default_top_p

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        if json_format:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0

            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens

            return LLMResponse(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as e:
            print(f"LLM inference error: {e}")
            return LLMResponse(content="", prompt_tokens=0, completion_tokens=0)

    def sample_n(
        self,
        messages: List[Dict[str, str]],
        n: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: int = 512,
    ) -> List[LLMResponse]:
        """
        Generate n independent samples from the LLM.

        Args:
            messages: List of message dicts
            n: Number of samples to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens per sample

        Returns:
            List of LLMResponse objects
        """
        responses = []
        for _ in range(n):
            resp = self.generate(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            responses.append(resp)
        return responses

    def generate_with_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: int = 512,
    ) -> LLMResponse:
        """
        Convenience method to generate with system and user prompts.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.generate(messages, temperature, top_p, max_tokens)

    def get_token_usage(self) -> Dict[str, int]:
        """Get total token usage."""
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }

    def reset_token_usage(self):
        """Reset token counters."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0


def load_api_key_from_env_file(env_path: str) -> str:
    """Load API key from .env file."""
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('OPENROUTER_API_KEY='):
                key = line.split('=', 1)[1].strip('"\'')
                return key
    raise ValueError(f"OPENROUTER_API_KEY not found in {env_path}")
