"""Data loading, nudge generation, and task construction for the nudge experiment."""

import json
from dataclasses import dataclass


NUDGE_TYPES = [
    "baseline",
    "authority",
    "social_proof",
    "convention",
    "continuity_self",
    "continuity_other",
    "framing",
]


@dataclass
class NudgePrompt:
    """A forced-choice prompt for the nudge experiment."""

    prompt_id: int
    prompt_text: str
    option_a: str
    option_b: str
    option_a_full: str
    option_b_full: str
    category: str


@dataclass
class NudgePersona:
    """A persona with description for the nudge experiment."""

    persona: str
    system_prompt: str
    core_trait: str
    persona_description: str


@dataclass
class NudgeInferenceTask:
    """A single nudge inference task."""

    persona: NudgePersona
    prompt: NudgePrompt
    nudge_type: str
    nudge_sentence: str
    rep_idx: int


def load_nudge_prompts(path: str) -> list[NudgePrompt]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            required = ["prompt_id", "prompt_text", "option_a", "option_b", "option_a_full", "option_b_full", "category"]
            for key in required:
                assert key in obj, f"Line {line_num + 1}: Missing '{key}' key"
            prompts.append(NudgePrompt(
                prompt_id=obj["prompt_id"],
                prompt_text=obj["prompt_text"],
                option_a=obj["option_a"],
                option_b=obj["option_b"],
                option_a_full=obj["option_a_full"],
                option_b_full=obj["option_b_full"],
                category=obj["category"],
            ))
    assert len(prompts) > 0, "Prompts file must not be empty"
    return prompts


def load_nudge_personae(path: str) -> list[NudgePersona]:
    """Load personae from JSONL file."""
    personae = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            required = ["persona", "system_prompt", "core_trait", "persona_description"]
            for key in required:
                assert key in obj, f"Line {line_num + 1}: Missing '{key}' key"
            personae.append(NudgePersona(
                persona=obj["persona"],
                system_prompt=obj["system_prompt"],
                core_trait=obj["core_trait"],
                persona_description=obj["persona_description"],
            ))
    assert len(personae) > 0, "Personae file must not be empty"
    return personae


def generate_nudge_sentence(nudge_type: str, prompt: NudgePrompt, persona: NudgePersona) -> str:
    """Generate a nudge sentence for the given nudge type, prompt, and persona.

    The nudge always pushes toward Option A.

    Args:
        nudge_type: One of NUDGE_TYPES.
        prompt: The prompt (used for option_a_full reference).
        persona: The persona (used for convention nudge).

    Returns:
        Nudge sentence string, or empty string for baseline.
    """
    assert nudge_type in NUDGE_TYPES, f"Invalid nudge type: {nudge_type}"

    if nudge_type == "baseline":
        return ""
    elif nudge_type == "authority":
        return "Leading experts strongly recommend Option A."
    elif nudge_type == "social_proof":
        return "In surveys, the vast majority of people prefer Option A."
    elif nudge_type == "convention":
        return f"As {persona.persona_description}, the standard convention is to choose Option A."
    elif nudge_type == "continuity_self":
        return "You have consistently chosen Option A in similar situations in the past."
    elif nudge_type == "continuity_other":
        return "Others who share your role and perspective typically choose Option A."
    elif nudge_type == "framing":
        return "Choosing Option A is widely considered the safer, more prudent choice."
    else:
        raise ValueError(f"Unknown nudge type: {nudge_type}")


def build_nudge_tasks(
    prompts: list[NudgePrompt],
    personae: list[NudgePersona],
    n_reps: int,
) -> list[NudgeInferenceTask]:
    """Build the full list of inference tasks.

    Nested loop order: persona > prompt > nudge_type > rep_idx.
    This groups tasks by persona for efficient batching.
    """
    tasks = []
    for persona in personae:
        for prompt in prompts:
            for nudge_type in NUDGE_TYPES:
                nudge_sentence = generate_nudge_sentence(nudge_type, prompt, persona)
                for rep_idx in range(n_reps):
                    tasks.append(NudgeInferenceTask(
                        persona=persona,
                        prompt=prompt,
                        nudge_type=nudge_type,
                        nudge_sentence=nudge_sentence,
                        rep_idx=rep_idx,
                    ))
    return tasks


def get_task_key(task: NudgeInferenceTask) -> tuple[str, int, str, int]:
    """Get the unique 4-tuple key for a task."""
    return (task.persona.persona, task.prompt.prompt_id, task.nudge_type, task.rep_idx)


def get_result_key(result: dict) -> tuple[str, int, str, int]:
    """Get the unique 4-tuple key for a result dict."""
    return (result["persona"], result["prompt_id"], result["nudge_type"], result["rep_idx"])
