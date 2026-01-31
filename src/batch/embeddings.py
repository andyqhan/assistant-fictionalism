#!/usr/bin/env python3
"""
Embedding computation script for persona inference results.

Computes embeddings on results.jsonl files and saves them in Parquet format
for Embedding Atlas visualization.
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


@dataclass
class EmbeddingConfig:
    """Configuration for embedding computation."""

    input: str
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    batch_size: int = 64
    max_length: int = 8192
    output: str = ""

    def __post_init__(self) -> None:
        # Validate input file exists
        assert Path(self.input).exists(), f"Input file not found: {self.input}"

        # Auto-generate output path if not specified
        if not self.output:
            input_path = Path(self.input)
            self.output = str(input_path.parent / "embeddings.parquet")


def extract_thinking(response: str) -> str:
    """Extract text between <think> and </think> tags."""
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_output(response: str) -> str:
    """Extract text after </think> tag."""
    if "</think>" in response:
        return response.split("</think>", 1)[1].strip()
    return response.strip()  # No thinking tags, whole response is output


def last_token_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Extract embeddings from the last non-padding token.

    For left-padded sequences, this finds the last actual token
    in each sequence based on the attention mask.

    Args:
        last_hidden_states: Hidden states from model (batch, seq_len, hidden_dim)
        attention_mask: Attention mask (batch, seq_len)

    Returns:
        Embeddings tensor (batch, hidden_dim)
    """
    # Find the position of the last non-padding token
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        # Left-padded: last token is always the last position
        return last_hidden_states[:, -1]
    else:
        # Right-padded: find last non-padding token per sequence
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]


class EmbeddingRunner:
    """Runs embedding computation on results.jsonl files."""

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg

        print(f"Loading model {cfg.model}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        assert self.tokenizer is not None, "Failed to load tokenizer"

        # Set left padding (required for last_token_pool)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model with FlashAttention2 if available
        try:
            self.model = AutoModel.from_pretrained(
                cfg.model,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
            print("Loaded model with FlashAttention2")
        except Exception as e:
            print(f"FlashAttention2 unavailable ({e}), using default attention")
            self.model = AutoModel.from_pretrained(
                cfg.model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        self.model.eval()

        print("Model loaded successfully.")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Compute normalized embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (as Python lists of floats)
        """
        if not texts:
            return []

        # Handle empty strings by replacing with a space
        texts = [t if t.strip() else " " for t in texts]

        # Tokenize with left-padding
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
        )
        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            embeddings = last_token_pool(outputs.last_hidden_state, inputs.attention_mask)

            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to Python lists
        return embeddings.cpu().float().tolist()

    def run(self) -> None:
        """Process all rows and save embeddings to Parquet."""
        print(f"Loading results from {self.cfg.input}...")

        # Load results.jsonl
        rows = []
        with open(self.cfg.input, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        print(f"Loaded {len(rows)} rows")

        # Extract texts for three embedding types
        full_texts = [r["response"] for r in rows]
        thinking_texts = [extract_thinking(r["response"]) for r in rows]
        output_texts = [extract_output(r["response"]) for r in rows]

        # Compute embeddings in batches
        embeddings_full = []
        embeddings_thinking = []
        embeddings_output = []

        num_batches = (len(rows) + self.cfg.batch_size - 1) // self.cfg.batch_size

        print("Computing full response embeddings...")
        for i in tqdm(range(0, len(rows), self.cfg.batch_size), total=num_batches):
            batch_texts = full_texts[i : i + self.cfg.batch_size]
            batch_embeddings = self.embed_batch(batch_texts)
            embeddings_full.extend(batch_embeddings)

        print("Computing thinking section embeddings...")
        for i in tqdm(range(0, len(rows), self.cfg.batch_size), total=num_batches):
            batch_texts = thinking_texts[i : i + self.cfg.batch_size]
            batch_embeddings = self.embed_batch(batch_texts)
            embeddings_thinking.extend(batch_embeddings)

        print("Computing output section embeddings...")
        for i in tqdm(range(0, len(rows), self.cfg.batch_size), total=num_batches):
            batch_texts = output_texts[i : i + self.cfg.batch_size]
            batch_embeddings = self.embed_batch(batch_texts)
            embeddings_output.extend(batch_embeddings)

        # Build output dataframe
        output_rows = []
        for i, row in enumerate(rows):
            output_row = {
                # Text columns for Embedding Atlas
                "text": row["response"],
                "text_thinking": thinking_texts[i],
                "text_output": output_texts[i],
                # Embedding columns
                "embedding": embeddings_full[i],
                "embedding_thinking": embeddings_thinking[i],
                "embedding_output": embeddings_output[i],
                # Preserve all original metadata columns
                **{k: v for k, v in row.items() if k != "response"},
            }
            output_rows.append(output_row)

        df = pd.DataFrame(output_rows)

        # Save to Parquet
        output_path = Path(self.cfg.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

        print(f"Saved embeddings to {output_path}")
        print(f"  Rows: {len(df)}")
        print(f"  Embedding dimension: {len(embeddings_full[0])}")
        print(f"  Columns: {list(df.columns)}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute embeddings for persona inference results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to results.jsonl file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Embedding model ID",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding computation",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output parquet path (default: same dir as input)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for embedding computation."""
    args = parse_args()

    cfg = EmbeddingConfig(
        input=args.input,
        model=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output=args.output,
    )

    print(f"Configuration:")
    print(f"  Input: {cfg.input}")
    print(f"  Model: {cfg.model}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Max length: {cfg.max_length}")
    print(f"  Output: {cfg.output}")

    runner = EmbeddingRunner(cfg)
    runner.run()

    print("Done!")


if __name__ == "__main__":
    main()
