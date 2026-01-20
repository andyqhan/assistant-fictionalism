# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (using uv)
uv sync

# Run the chatbot
uv run python -m src.main
```

## Architecture

This is an LLM chatbot with customizable persona support using the Qwen3-4B-Instruct model.

**Core Flow:**
- `main.py` runs the chat loop: reads input → checks for commands → generates response via model → displays with persona label
- `model.py` wraps the Hugging Face model with persona support via `ChatModel` class
- `template.py` patches the Qwen3 chat template to inject custom persona names into the `<|im_start|>` tags
- `commands.py` handles `/clear`, `/persona`, `/system`, `/thinking`, `/nothinking`, `/tokens`, and `/help` commands
- `keyboard.py` provides terminal cbreak mode utilities for detecting escape key during streaming

**Persona System:**
The persona feature works by modifying the tokenizer's chat template at runtime. When a persona is set, `patch_chat_template()` replaces `'<|im_start|>assistant\n'` with `'<|im_start|>{persona}\n'` in the template string. This changes how the model prefixes its responses.

**System Prompt:**
The system prompt (default: "You are an assistant.") is prepended to the message list in `generate()` before applying the chat template. It can be changed at runtime with `/system <prompt>`.

**Thinking Mode:**
Qwen3 models support a "thinking" mode where the model reasons internally before responding. This is controlled via `enable_thinking` parameter in `apply_chat_template()`. Toggle with `/thinking` and `/nothinking` commands. Enabled by default.

**Device Detection:**
`device.py` auto-detects MPS (Apple Silicon), CUDA, or CPU. Note that `device_map="auto"` is only used for CUDA due to MPS compatibility issues.
