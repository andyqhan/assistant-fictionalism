import torch
import torch.nn.functional as F


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
        Dictionary containing:
            - avg_entropy_thinking: Average entropy for thinking tokens (or None)
            - avg_entropy_output: Average entropy for output tokens (or None)
            - avg_entropy: Average entropy for all tokens
            - avg_top_k_mass_thinking: Average top-k mass for thinking (or None)
            - avg_top_k_mass_output: Average top-k mass for output (or None)
            - avg_top_k_mass: Average top-k mass for all tokens
            - think_end_position: Position of </think> token (or None)
            - num_tokens: Total number of generated tokens
    """
    assert len(logits_list) == len(token_ids), (
        f"Logits/tokens length mismatch: {len(logits_list)} vs {len(token_ids)}"
    )

    if len(logits_list) == 0:
        return {
            "avg_entropy_thinking": None,
            "avg_entropy_output": None,
            "avg_entropy": None,
            "avg_top_k_mass_thinking": None,
            "avg_top_k_mass_output": None,
            "avg_top_k_mass": None,
            "think_end_position": None,
            "num_tokens": 0,
        }

    # Stack logits for vectorized computation
    stacked_logits = torch.stack(logits_list)  # (seq_len, vocab_size)

    # Compute metrics for all tokens
    entropies = compute_entropy(stacked_logits)  # (seq_len,)
    top_k_masses = compute_top_k_mass(stacked_logits, top_k)  # (seq_len,)

    # Find </think> position
    think_end_position = None
    for i, tid in enumerate(token_ids):
        if tid == think_end_token_id:
            think_end_position = i
            break

    # Separate thinking and output metrics
    if think_end_position is not None:
        # Thinking tokens: indices 0 to think_end_position (inclusive)
        thinking_entropies = entropies[: think_end_position + 1]
        thinking_top_k = top_k_masses[: think_end_position + 1]

        # Output tokens: indices after think_end_position
        output_entropies = entropies[think_end_position + 1 :]
        output_top_k = top_k_masses[think_end_position + 1 :]

        avg_entropy_thinking = thinking_entropies.mean().item() if len(thinking_entropies) > 0 else None
        avg_top_k_mass_thinking = thinking_top_k.mean().item() if len(thinking_top_k) > 0 else None

        avg_entropy_output = output_entropies.mean().item() if len(output_entropies) > 0 else None
        avg_top_k_mass_output = output_top_k.mean().item() if len(output_top_k) > 0 else None
    else:
        # No thinking section found - all tokens are output
        avg_entropy_thinking = None
        avg_top_k_mass_thinking = None
        avg_entropy_output = entropies.mean().item()
        avg_top_k_mass_output = top_k_masses.mean().item()

    return {
        "avg_entropy_thinking": avg_entropy_thinking,
        "avg_entropy_output": avg_entropy_output,
        "avg_entropy": entropies.mean().item(),
        "avg_top_k_mass_thinking": avg_top_k_mass_thinking,
        "avg_top_k_mass_output": avg_top_k_mass_output,
        "avg_top_k_mass": top_k_masses.mean().item(),
        "think_end_position": think_end_position,
        "num_tokens": len(token_ids),
    }
