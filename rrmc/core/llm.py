"""
LLM Wrapper for RRMC using OpenRouter API.
"""

import os
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any, Union, Callable
from dataclasses import dataclass

from openai import OpenAI


def strip_think_tags(text: str) -> str:
    """
    Strip Qwen3's <think>...</think> reasoning tags from output.
    
    Qwen3 models produce extended reasoning in <think> blocks which can be
    thousands of tokens. We strip these to save context and get clean answers.
    
    If the entire response is inside <think> tags (nothing left after stripping),
    we attempt to extract useful content (like 4-digit numbers for GN task) from
    inside the think block.
    
    Args:
        text: Raw model output that may contain <think> blocks
        
    Returns:
        Text with <think> blocks removed, or extracted content if nothing remains
    """
    if not text:
        return text
    
    # First, check what's outside the think tags
    text_outside = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
    
    # If there's content outside think tags, use that
    if text_outside:
        return text_outside
    
    # Nothing outside think tags - try to extract useful content from inside
    think_match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL)
    if think_match:
        think_content = think_match.group(1)
        
        # Look for "Guess: XXXX" pattern first (common in GN task)
        guess_match = re.search(r'[Gg]uess[:\s]+(\d{4})', think_content)
        if guess_match:
            return f"Guess: {guess_match.group(1)}"
        
        # Look for any 4-digit number (for GN task)
        four_digit = re.findall(r'\b\d{4}\b', think_content)
        if four_digit:
            # Return the last 4-digit number found (usually the final guess)
            return four_digit[-1]
        
        # Look for letter answers A-E (for DC task)
        letter_match = re.search(r'\b(?:answer|suspect)[:\s]*([A-E])\b', think_content, re.IGNORECASE)
        if letter_match:
            return letter_match.group(1)
        
        # Fallback: return last line of think content (often contains the answer)
        lines = [l.strip() for l in think_content.strip().split('\n') if l.strip()]
        if lines:
            return lines[-1]
    
    return text.strip()


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
        progress_callback: Optional[Callable[[int], None]] = None,
        max_workers: int = 8,
        provider: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LLM wrapper.

        Args:
            api_key: OpenRouter API key (or reads from OPENROUTER_API_KEY env)
            model: Model identifier (default: qwen/qwen3-8b)
            base_url: API base URL (default: OpenRouter)
            default_temperature: Default sampling temperature
            default_top_p: Default nucleus sampling parameter
            progress_callback: Optional callback(n) called after each API call
            max_workers: Maximum concurrent API calls (default: 8)
            provider: OpenRouter provider routing config (e.g., {"order": ["Fireworks"]})
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set OPENROUTER_API_KEY or pass api_key.")

        self.model = model
        self.base_url = base_url or self.OPENROUTER_BASE_URL
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.progress_callback = progress_callback
        self.max_workers = max_workers
        self.provider = provider  # OpenRouter provider routing

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        # Thread-safe token tracking
        self._lock = threading.Lock()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.call_count = 0

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

        # Add OpenRouter provider routing if specified
        if self.provider:
            # Convert Config object to dict if needed
            provider_dict = self.provider
            if hasattr(provider_dict, 'to_dict'):
                provider_dict = provider_dict.to_dict()
            elif hasattr(provider_dict, '__dict__'):
                provider_dict = dict(provider_dict.__dict__)
            kwargs["extra_body"] = {"provider": provider_dict}

        try:
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0

            # Strip Qwen3's <think> tags to get clean output
            content = strip_think_tags(content)

            # Thread-safe token tracking
            with self._lock:
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.call_count += 1

            # Update progress bar if callback is set
            if self.progress_callback:
                self.progress_callback(1)

            return LLMResponse(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as e:
            print(f"LLM inference error: {e}")
            with self._lock:
                self.call_count += 1
            if self.progress_callback:
                self.progress_callback(1)
            return LLMResponse(content="", prompt_tokens=0, completion_tokens=0)

    def sample_n(
        self,
        messages: List[Dict[str, str]],
        n: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: int = 512,
        parallel: bool = True,
    ) -> List[LLMResponse]:
        """
        Generate n independent samples from the LLM.

        Args:
            messages: List of message dicts
            n: Number of samples to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens per sample
            parallel: Whether to run samples in parallel (default: True)

        Returns:
            List of LLMResponse objects
        """
        if not parallel or n <= 1:
            # Sequential execution
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

        # Parallel execution
        def _generate_one(_):
            return self.generate(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

        responses = [None] * n
        with ThreadPoolExecutor(max_workers=min(n, self.max_workers)) as executor:
            futures = {executor.submit(_generate_one, i): i for i in range(n)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    responses[idx] = future.result()
                except Exception as e:
                    print(f"Parallel sample {idx} failed: {e}")
                    responses[idx] = LLMResponse(content="", prompt_tokens=0, completion_tokens=0)

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
