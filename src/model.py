from collections.abc import Iterator
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from .device import get_device, get_dtype
from .template import patch_chat_template


MODEL_ID = "Qwen/Qwen3-1.7B"
DEFAULT_PERSONA = "assistant"
DEFAULT_SYSTEM_PROMPT = "You are an assistant."
DEFAULT_MAX_TOKENS = 512


class ChatModel:
    """Wrapper for the Qwen3 chat model with persona support."""

    def __init__(self) -> None:
        self.device = get_device()
        self.dtype = get_dtype(self.device)
        self.persona = DEFAULT_PERSONA
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.enable_thinking = True
        self.max_new_tokens = DEFAULT_MAX_TOKENS

        print(f"Loading {MODEL_ID} on {self.device} with {self.dtype}...")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        assert self.tokenizer is not None, "Failed to load tokenizer"
        assert hasattr(self.tokenizer, "chat_template"), "Tokenizer missing chat_template"

        self._original_template = self.tokenizer.chat_template

        # device_map="auto" doesn't work well with MPS, causes disk offloading
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=self.dtype,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)
        assert self.model is not None, "Failed to load model"

        print("Model loaded successfully.")

    def set_persona(self, persona: str) -> None:
        """Set a new persona for the assistant."""
        assert isinstance(persona, str), f"Persona must be a string, got {type(persona)}"

        self.persona = persona
        patch_chat_template(self.tokenizer, self._original_template, persona)

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt."""
        assert isinstance(prompt, str), f"System prompt must be a string, got {type(prompt)}"
        self.system_prompt = prompt

    def set_thinking(self, enabled: bool) -> None:
        """Enable or disable thinking mode."""
        assert isinstance(enabled, bool), f"Enabled must be a bool, got {type(enabled)}"
        self.enable_thinking = enabled

    def set_max_tokens(self, tokens: int) -> None:
        """Set the maximum number of tokens to generate."""
        assert isinstance(tokens, int), f"Tokens must be an int, got {type(tokens)}"
        assert tokens > 0, f"Tokens must be positive, got {tokens}"
        self.max_new_tokens = tokens

    def generate(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """
        Generate a streaming response for the given message history.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Yields:
            Text chunks as they are generated
        """
        assert len(messages) > 0, "Messages list cannot be empty"
        for msg in messages:
            assert "role" in msg, f"Message missing 'role': {msg}"
            assert "content" in msg, f"Message missing 'content': {msg}"

        # Prepend system prompt if set
        if self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}] + messages

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for chunk in streamer:
            yield chunk

        thread.join()
