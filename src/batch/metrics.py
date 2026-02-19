from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from vllm.outputs import Logprob


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy from logits: -sum(p * log(p)).

    Args:
        logits: Tensor of shape (..., vocab_size) containing raw logits

    Returns:
        Tensor of shape (...) containing entropy values
    """
    assert logits.dim() >= 1, f"Logits must have at least 1 dimension, got {logits.dim()}"

    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)

    # Compute entropy: -sum(p * log(p))
    # Using log_softmax for numerical stability
    entropy = -torch.sum(probs * log_probs, dim=-1)

    return entropy


def compute_top_k_mass(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute the sum of top-k probabilities.

    Args:
        logits: Tensor of shape (..., vocab_size) containing raw logits
        k: Number of top probabilities to sum

    Returns:
        Tensor of shape (...) containing top-k probability mass
    """
    assert logits.dim() >= 1, f"Logits must have at least 1 dimension, got {logits.dim()}"
    assert k > 0, f"k must be positive, got {k}"

    probs = F.softmax(logits, dim=-1)

    # Get top-k probabilities
    vocab_size = probs.shape[-1]
    actual_k = min(k, vocab_size)
    topk_probs, _ = torch.topk(probs, actual_k, dim=-1)

    # Sum the top-k probabilities
    top_k_mass = topk_probs.sum(dim=-1)

    return top_k_mass


def compute_entropy_and_top_k_mass(
    logits: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute both entropy and top-k mass efficiently with a single softmax.

    Args:
        logits: Tensor of shape (..., vocab_size) containing raw logits
        k: Number of top probabilities to sum for top-k mass

    Returns:
        Tuple of (entropy, top_k_mass) tensors, each of shape (...)
    """
    assert logits.dim() >= 1, f"Logits must have at least 1 dimension, got {logits.dim()}"
    assert k > 0, f"k must be positive, got {k}"

    # Single softmax computation for both metrics
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)

    # Entropy: -sum(p * log(p))
    entropy = -torch.sum(probs * log_probs, dim=-1)

    # Top-k mass
    vocab_size = probs.shape[-1]
    actual_k = min(k, vocab_size)
    topk_probs, _ = torch.topk(probs, actual_k, dim=-1)
    top_k_mass = topk_probs.sum(dim=-1)

    return entropy, top_k_mass


def summarize_values(values: list[float]) -> dict[str, float | None]:
    """Compute summary statistics for a list of per-token values.

    Returns dict with keys: avg, std, min, max.
    All None if values is empty. Uses population standard deviation.
    """
    if not values:
        return {"avg": None, "std": None, "min": None, "max": None}
    n = len(values)
    avg = sum(values) / n
    std = (sum((v - avg) ** 2 for v in values) / n) ** 0.5
    return {"avg": avg, "std": std, "min": min(values), "max": max(values)}


def compute_section_summaries(
    entropies: list[float],
    top_k_masses: list[float],
    surprisals: list[float],
    token_ids: list[int],
    think_end_token_id: int,
) -> dict:
    """Compute summary statistics for entropy, top-k mass, and surprisal, split by thinking/output.

    Splits token-level values at the first </think> token into thinking and output
    sections, then computes avg/std/min/max for each section and overall.
    Perplexity is computed as exp(avg_surprisal) for each section.

    Args:
        entropies: Per-token entropy values.
        top_k_masses: Per-token top-k mass values.
        surprisals: Per-token surprisal values (-log p(chosen token)).
        token_ids: Generated token IDs (used to find </think> boundary).
        think_end_token_id: Token ID for </think>.

    Returns:
        Dictionary with {avg,std,min,max}_{entropy,top_k_mass,surprisal}_{thinking,output,},
        perplexity_{thinking,output,}, plus think_end_position and num_tokens.
    """
    assert len(entropies) == len(top_k_masses) == len(surprisals) == len(token_ids)

    if not token_ids:
        result = {}
        for stat in ("avg", "std", "min", "max"):
            for metric in ("entropy", "top_k_mass", "surprisal"):
                for suffix in ("_thinking", "_output", ""):
                    result[f"{stat}_{metric}{suffix}"] = None
        for suffix in ("_thinking", "_output", ""):
            result[f"perplexity{suffix}"] = None
        result["think_end_position"] = None
        result["num_tokens"] = 0
        return result

    # Find </think> position
    think_end_position = None
    for i, tid in enumerate(token_ids):
        if tid == think_end_token_id:
            think_end_position = i
            break

    # Split into sections
    if think_end_position is not None:
        thinking_ents = entropies[: think_end_position + 1]
        thinking_topk = top_k_masses[: think_end_position + 1]
        thinking_surp = surprisals[: think_end_position + 1]
        output_ents = entropies[think_end_position + 1 :]
        output_topk = top_k_masses[think_end_position + 1 :]
        output_surp = surprisals[think_end_position + 1 :]
    else:
        thinking_ents = []
        thinking_topk = []
        thinking_surp = []
        output_ents = entropies
        output_topk = top_k_masses
        output_surp = surprisals

    # Compute stats for each section
    sections = {
        "_thinking": (summarize_values(thinking_ents), summarize_values(thinking_topk), summarize_values(thinking_surp)),
        "_output": (summarize_values(output_ents), summarize_values(output_topk), summarize_values(output_surp)),
        "": (summarize_values(entropies), summarize_values(top_k_masses), summarize_values(surprisals)),
    }

    result = {}
    for suffix, (ent_stats, topk_stats, surp_stats) in sections.items():
        for stat in ("avg", "std", "min", "max"):
            result[f"{stat}_entropy{suffix}"] = ent_stats[stat]
            result[f"{stat}_top_k_mass{suffix}"] = topk_stats[stat]
            result[f"{stat}_surprisal{suffix}"] = surp_stats[stat]
        result[f"perplexity{suffix}"] = (
            math.exp(surp_stats["avg"]) if surp_stats["avg"] is not None else None
        )

    result["think_end_position"] = think_end_position
    result["num_tokens"] = len(token_ids)

    return result


def compute_metrics_for_sequence(
    logits_list: list[torch.Tensor],
    token_ids: list[int],
    think_end_token_id: int,
    top_k: int,
) -> dict:
    """
    Compute entropy and top-k mass metrics for a generated sequence.

    Separates metrics into thinking (before </think>) and output (after </think>).

    Args:
        logits_list: List of logit tensors, one per generated token
        token_ids: List of generated token IDs
        think_end_token_id: Token ID for </think> token
        top_k: k value for top-k mass computation

    Returns:
        Dictionary with summary statistics (see compute_section_summaries).
    """
    assert len(logits_list) == len(token_ids), (
        f"Logits/tokens length mismatch: {len(logits_list)} vs {len(token_ids)}"
    )

    if len(logits_list) == 0:
        return compute_section_summaries([], [], [], [], think_end_token_id)

    stacked_logits = torch.stack(logits_list)  # (seq_len, vocab_size)
    entropies = compute_entropy(stacked_logits).tolist()
    top_k_masses_vals = compute_top_k_mass(stacked_logits, top_k).tolist()

    # Compute surprisal: -log p(chosen token)
    log_probs = F.log_softmax(stacked_logits, dim=-1)
    token_ids_tensor = torch.tensor(token_ids, device=stacked_logits.device).unsqueeze(1)
    chosen_log_probs = log_probs.gather(1, token_ids_tensor).squeeze(1)
    surprisals = (-chosen_log_probs).tolist()

    return compute_section_summaries(entropies, top_k_masses_vals, surprisals, token_ids, think_end_token_id)


def compute_token_entropy_from_logprobs(token_logprobs: dict[int, Logprob]) -> float:
    """
    Compute approximate entropy from vLLM logprobs: -sum(exp(lp) * lp) for top-k tokens.

    Args:
        token_logprobs: Dict mapping token ID to Logprob object with .logprob attribute

    Returns:
        Approximate entropy (lower bound since we only have top-k)
    """
    if not token_logprobs:
        return 0.0

    entropy = 0.0
    for logprob_obj in token_logprobs.values():
        lp = logprob_obj.logprob
        prob = math.exp(lp)
        entropy -= prob * lp

    return entropy


def compute_token_top_k_mass_from_logprobs(
    token_logprobs: dict[int, Logprob], k: int
) -> float:
    """
    Compute sum of top-k probabilities from vLLM logprobs.

    Args:
        token_logprobs: Dict mapping token ID to Logprob object with .logprob attribute
        k: Number of top probabilities to sum

    Returns:
        Sum of top-k probabilities
    """
    if not token_logprobs:
        return 0.0

    # Sort by logprob descending and take top k
    sorted_logprobs = sorted(
        (lp.logprob for lp in token_logprobs.values()),
        reverse=True,
    )[:k]

    # Sum probabilities
    return sum(math.exp(lp) for lp in sorted_logprobs)


def compute_metrics_for_vllm_output(
    logprobs: list[dict[int, Logprob]] | None,
    token_ids: list[int],
    think_end_token_id: int,
    top_k_mass_k: int,
) -> dict:
    """
    Compute entropy and top-k mass metrics from vLLM output.

    Separates metrics into thinking (before </think>) and output (after </think>).

    Args:
        logprobs: List of logprob dicts, one per generated token
        token_ids: List of generated token IDs
        think_end_token_id: Token ID for </think> token
        top_k_mass_k: k value for top-k mass computation

    Returns:
        Dictionary with summary statistics (see compute_section_summaries).
    """
    if logprobs is None or len(logprobs) == 0:
        return compute_section_summaries([], [], [], [], think_end_token_id)

    entropies = [compute_token_entropy_from_logprobs(lp) for lp in logprobs]
    top_k_masses_vals = [
        compute_token_top_k_mass_from_logprobs(lp, top_k_mass_k) for lp in logprobs
    ]

    # Extract surprisal: -log p(chosen token)
    # vLLM always includes the sampled token in the logprobs dict
    surprisals = []
    for lp_dict, tid in zip(logprobs, token_ids):
        assert tid in lp_dict, f"Sampled token {tid} not found in logprobs dict"
        surprisals.append(-lp_dict[tid].logprob)

    return compute_section_summaries(entropies, top_k_masses_vals, surprisals, token_ids, think_end_token_id)
