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
    --batch-size=256

# Run the visualization webapp
uv run streamlit run src/viz/app.py
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
├── interactive/
│   ├── __init__.py
│   ├── main.py           # Interactive chat loop
│   ├── commands.py       # Slash command handling
│   ├── keyboard.py       # Terminal utilities for escape detection
│   └── colors.py         # Terminal color formatting
└── viz/
    ├── __init__.py
    └── app.py            # Streamlit visualization webapp
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
| `--batch-size` | `256` | Batch size |
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

### Visualization (`src/viz/`)

**Purpose:** Interactive Streamlit webapp for exploring batch inference results.

**Features:**
- File selector for choosing results from different runs
- Three visualization tabs:
  - **Entropy**: Box-and-whisker plot showing `avg_entropy_thinking`, `avg_entropy_output`, and `avg_entropy` per category
  - **Top-k Mass**: Box-and-whisker plot showing `avg_top_k_mass_thinking`, `avg_top_k_mass_output`, and `avg_top_k_mass` per category
  - **Thinking Tokens**: Box-and-whisker plot showing `think_end_position` (number of thinking tokens) per category
- Category filtering in the sidebar
- Summary statistics expandable for each chart
- Raw data view with column selection

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

### Python Environment

**Two separate environments:**
- **Local development:** Uses `uv` to manage dependencies (via `pyproject.toml`). Run with `uv run python ...`.
- **HPC compute nodes:** Uses a **separate venv** at `/scratch/ah7660/venvs/assistant-fictionalism-env` that runs inside a Singularity container.

**Why separate environments?**
The HPC uses Singularity containers for CUDA/GPU support. The venv is stored on `/scratch` (mounted into the container) rather than using `uv` because:
1. The container provides the CUDA runtime needed for GPU packages
2. Packages requiring CUDA at install time (e.g., `flash-attn`) must be installed inside the container

**Installing HPC dependencies:**
```bash
# Install a package that requires CUDA (runs inside Singularity container)
sbatch hpc/install_package.slurm flash-attn --no-build-isolation

# The script uses python -m pip inside the container to install packages
```

**Note:** The HPC venv is created by `uv` but doesn't include pip by default. If pip is missing, install it first:
```bash
uv pip install pip --python /scratch/ah7660/venvs/assistant-fictionalism-env/bin/python
```

### SLURM Scripts

**`hpc/batch_inference.slurm`** - Main inference script
- Targets H200 GPUs via `--constraint=h200`
- Uses Singularity container: `/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif`
- Activates venv inside container: `source ${VENV}/bin/activate`
- Logs to `logs/personae_inference_{job_id}.log`

**`hpc/install_package.slurm`** - Install packages requiring CUDA
- Runs `pip install` inside the Singularity container
- Use for packages that need CUDA at compile time (flash-attn, etc.)

## Code Style

**Defensive Programming:**
- Use `assert` statements to validate tensor shapes, sizes, and invariants
- Fail fast: check preconditions early and raise clear errors
- Validate function inputs at the start of functions
- Check tensor dimensions match expected shapes before operations
