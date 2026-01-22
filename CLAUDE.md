# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (using uv)
uv sync

# Run the interactive chatbot
uv run python -m src.interactive.main

# Run batch inference locally
uv run python -m src.batch.inference \
    --prompts-json=datasets/prompts.json \
    --personae-json=datasets/personae.json

# Submit batch inference to SLURM (HPC)
sbatch hpc/batch_inference.slurm \
    --prompts-json=datasets/prompts.json \
    --personae-json=datasets/personae.json \
    --model='Qwen/Qwen3-8B' \
    --batch-size=64
```

## Architecture

This project has two components: an interactive chatbot and a batch inference system for persona experiments.

### Directory Structure

```
src/
├── __init__.py
├── model.py              # Shared: ChatModel class with persona support
├── template.py           # Shared: Chat template patching for personas
├── device.py             # Shared: Device detection (MPS/CUDA/CPU)
├── batch/
│   ├── __init__.py
│   ├── inference.py      # Main batch inference script
│   ├── metrics.py        # Entropy and top-k mass computation
│   ├── system_prompts.py # System prompt generation
│   └── config.py         # BatchInferenceConfig dataclass
└── interactive/
    ├── __init__.py
    ├── main.py           # Interactive chat loop
    ├── commands.py       # Slash command handling
    ├── keyboard.py       # Terminal utilities for escape detection
    └── colors.py         # Terminal color formatting
```

### Interactive Mode (`src/interactive/`)

**Core Flow:**
- `main.py` runs the chat loop: reads input → checks for commands → generates response via model → displays with persona label
- `commands.py` handles `/clear`, `/persona`, `/system`, `/thinking`, `/nothinking`, `/tokens`, and `/help` commands
- `keyboard.py` provides terminal cbreak mode utilities for detecting escape key during streaming

### Batch Inference (`src/batch/`)

**Purpose:** Run inference across multiple personas and prompts, computing entropy and top-k mass metrics.

**Core Flow:**
- `inference.py` loads prompts and personae from JSON, builds tasks, runs batched inference, saves results
- Groups tasks by persona for efficient batching (same chat template)
- Uses left-padding for proper batch generation
- Separates metrics into "thinking" (before `</think>`) and "output" sections

**CLI Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--prompts-json` | required | Path to prompts JSON file |
| `--personae-json` | required | Path to personae JSON file |
| `--model` | `Qwen/Qwen3-8B` | Model ID |
| `--temperature` | `0.0` | Sampling temperature |
| `--max-tokens` | `1024` | Max tokens to generate |
| `--batch-size` | `64` | Batch size |
| `--n-reps` | `1` | Repetitions per prompt |
| `--top-k-mass-k` | `5` | k for top-k mass |
| `--thinking-mode/--no-thinking-mode` | enabled | Thinking mode |
| `--system-prompt-style` | `you-are-a` | System prompt style |
| `--output-dir` | auto | Output directory |

**Input Formats:**

`datasets/prompts.json`:
```json
["What is 2+2?", "Tell me about yourself.", ...]
```

`datasets/personae.json`:
```json
[
  {"persona": "assistant", "category": "assistant", "article": "an"},
  {"persona": "Hamlet", "category": "fictional-character", "article": ""}
]
```

**Output Format:**
Results saved to `logs/personae-inference-{SLURM_JOB_ID}/results.json`.

### Shared Components

**Persona System:**
The persona feature works by modifying the tokenizer's chat template at runtime. When a persona is set, `patch_chat_template()` replaces `'<|im_start|>assistant\n'` with `'<|im_start|>{persona}\n'` in the template string.

**System Prompt:**
The system prompt is prepended to the message list before applying the chat template. For batch inference, it's generated from the persona and article (e.g., "You are an assistant.").

**Thinking Mode:**
Qwen3 models support a "thinking" mode where the model reasons internally before responding. Controlled via `enable_thinking` parameter in `apply_chat_template()`.

**Device Detection:**
`device.py` auto-detects MPS (Apple Silicon), CUDA, or CPU. Note that `device_map="auto"` is only used for CUDA due to MPS compatibility issues.

## HPC

**SLURM Script:** `hpc/batch_inference.slurm`
- Targets H200 GPUs via `--constraint=h200`
- Uses Singularity container with CUDA 12.8.1
- Logs to `hpc/logs/personae_inference_{job_id}.log`
