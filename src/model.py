import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .device import get_device, get_dtype
from .template import patch_chat_template


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_PERSONA = "assistant"


class ChatModel:
    """Wrapper for the Qwen3 chat model with persona support."""

    def __init__(self) -> None:
        self.device = get_device()
        self.dtype = get_dtype(self.device)
        self.persona = DEFAULT_PERSONA

        print(f"Loading model on {self.device} with {self.dtype}...")

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

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int = 512) -> str:
        """
        Generate a response for the given message history.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            The generated response text
        """
        assert len(messages) > 0, "Messages list cannot be empty"
        for msg in messages:
            assert "role" in msg, f"Message missing 'role': {msg}"
            assert "content" in msg, f"Message missing 'content': {msg}"

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response.strip()
