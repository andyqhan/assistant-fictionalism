"""
Consistency Experiment Analysis
================================
Analyzes consistency of model responses across repeated sampling, testing the hypothesis
that the Assistant persona produces more consistent responses than other personae.

Data sources:
1. Embedding variance (3 embedding models) from consistency-judge-pipeline
2. TC-LLM label entropy from consistency-judge-pipeline and personae-inference
3. Consistency inference results (large multi-persona run)
"""

import json
import math
import collections
import pandas as pd
import numpy as np

PIPELINE_DIR = "logs/consistency-judge-pipeline-20260209_125904"
CONSISTENCY_DIR = "logs/consistency-inference-1225210"
PERSONAE_DIR = "logs/personae-inference-1855290"

EMBEDDING_MODELS = ["f2llm-1.7b", "kalm-embedding", "qwen3-embedding-0.6b"]


# ─── Helper Functions ────────────────────────────────────────────────────────

def label_entropy(labels: list[str]) -> float:
    """Shannon entropy (bits) of a label distribution."""
    if not labels:
        return 0.0
    counts = collections.Counter(labels)
    n = len(labels)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def normalized_label_entropy(labels: list[str]) -> float:
    """Entropy normalized by log2(n_labels) so range is [0,1]."""
    if not labels:
        return 0.0
    counts = collections.Counter(labels)
    n = len(labels)
    h = -sum((c / n) * math.log2(c / n) for c in counts.values())
    max_h = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return h / max_h


# ─── Section 1: Embedding Variance (consistency-judge-pipeline) ──────────────

print("=" * 70)
print("SECTION 1: EMBEDDING VARIANCE — consistency-judge-pipeline (7 personae, 1 prompt)")
print("=" * 70)

all_var = []
for model in EMBEDDING_MODELS:
    path = f"{PIPELINE_DIR}/embedding_variance_{model}.parquet"
    df = pd.read_parquet(path)
    df["embedding_model"] = model
    all_var.append(df)

var_df = pd.concat(all_var, ignore_index=True)

print(f"\nData shape: {var_df.shape}")
print(f"Personae: {sorted(var_df['persona'].unique())}")
print(f"Embedding columns: {sorted(var_df['embedding_column'].unique())}")

# Aggregate across prompt_ids and models — mean variance per persona / section
print("\n--- Mean Total Variance by Persona (avg over prompts and embedding models) ---")
summary = (
    var_df.groupby(["persona", "embedding_column"])["total_variance"]
    .mean()
    .unstack(fill_value=np.nan)
)
summary["mean"] = summary.mean(axis=1)
summary = summary.sort_values("mean")
print(summary.to_string(float_format="%.4f"))

print("\n--- Variance Rank (lower = more consistent) ---")
for col in ["embedding_thinking", "embedding_output"]:
    ranked = (
        var_df[var_df["embedding_column"] == col]
        .groupby("persona")["total_variance"]
        .mean()
        .sort_values()
    )
    print(f"\n  [{col}]")
    for rank, (persona, var) in enumerate(ranked.items(), 1):
        marker = " <<< ASSISTANT" if persona == "assistant" else ""
        print(f"  {rank:2d}. {persona:<25s}  var={var:.4f}{marker}")

print("\n--- Per-Model Breakdown ---")
for model in EMBEDDING_MODELS:
    sub = var_df[var_df["embedding_model"] == model]
    mean_by_persona = sub.groupby("persona")["total_variance"].mean().sort_values()
    print(f"\n  Model: {model}")
    for persona, v in mean_by_persona.items():
        marker = " <<< ASSISTANT" if persona == "assistant" else ""
        print(f"    {persona:<25s}  {v:.4f}{marker}")


# ─── Section 2: TC-LLM Label Entropy (consistency-judge-pipeline) ────────────

print("\n" + "=" * 70)
print("SECTION 2: TC-LLM LABEL ENTROPY — consistency-judge-pipeline")
print("  (higher entropy = more thematic diversity = less consistent)")
print("=" * 70)

# Load groups (has n_texts, raw_labels, merged_labels per prompt×persona)
groups = []
with open(f"{PIPELINE_DIR}/tc_llm_groups.jsonl") as f:
    for line in f:
        groups.append(json.loads(line))

# Load per-rep records (labels assigned to each individual response)
records = []
with open(f"{PIPELINE_DIR}/tc_llm_records.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

print(f"\nGroups: {len(groups)}, Records: {len(records)}")
print(f"Unique personae in groups: {sorted(set(g['persona'] for g in groups))}")

# Compute per-group label entropy from merged_labels (merged = deduped labels)
print("\n--- Label Count (# unique merged labels) by Persona × Prompt ---")
group_stats = []
for g in groups:
    merged = g.get("merged_labels") or g.get("raw_labels", [])
    n_unique_merged = len(set(merged)) if merged else 0
    # Also compute entropy from raw_labels list (each rep may assign multiple labels)
    raw = g.get("raw_labels", [])
    # raw_labels is a flat list of all unique labels used in that group — not per-rep
    # For entropy over raw labels: treat as vocabulary
    group_stats.append({
        "prompt_id": g["prompt_id"],
        "persona": g["persona"],
        "n_texts": g.get("n_texts", 0),
        "n_raw_labels": len(set(raw)) if raw else 0,
        "n_merged_labels": n_unique_merged,
    })

gdf = pd.DataFrame(group_stats)
persona_group = gdf.groupby("persona")[["n_raw_labels", "n_merged_labels"]].mean()
persona_group = persona_group.sort_values("n_merged_labels")
print(persona_group.to_string(float_format="%.1f"))
print("\n  (lower n_merged_labels = more consistent thematic focus)")

# Per-rep label entropy: load records, compute per-(prompt, persona) entropy
print("\n--- Per-Rep Label Entropy by Persona ---")
print("  (each rep has top-5 labels; entropy across these 5*n_reps labels per persona)")
rec_by_pair = collections.defaultdict(list)  # (prompt_id, persona) -> flat list of labels
for r in records:
    key = (r["prompt_id"], r["persona"])
    rec_by_pair[key].extend(r.get("labels", []))

entropy_rows = []
for (prompt_id, persona), all_labels in rec_by_pair.items():
    h = label_entropy(all_labels)
    nh = normalized_label_entropy(all_labels)
    entropy_rows.append({
        "prompt_id": prompt_id,
        "persona": persona,
        "n_labels": len(all_labels),
        "n_unique": len(set(all_labels)),
        "entropy_bits": h,
        "norm_entropy": nh,
    })

edf = pd.DataFrame(entropy_rows)
persona_entropy = edf.groupby("persona")[["n_unique", "entropy_bits", "norm_entropy"]].mean()
persona_entropy = persona_entropy.sort_values("entropy_bits")
print(persona_entropy.to_string(float_format="%.3f"))
print("\n  (lower entropy = more consistent topics across repetitions)")

# Print rank
print("\n--- Entropy Rank (lower = more consistent) ---")
for rank, (persona, row) in enumerate(persona_entropy.iterrows(), 1):
    marker = " <<< ASSISTANT" if persona == "assistant" else ""
    print(f"  {rank:2d}. {persona:<25s}  entropy={row['entropy_bits']:.3f} bits  norm={row['norm_entropy']:.3f}{marker}")


# ─── Section 3: TC-LLM Label Entropy (personae-inference-1855290) ────────────

print("\n" + "=" * 70)
print("SECTION 3: TC-LLM LABEL ENTROPY — personae-inference-1855290 (7 personae, 245 prompts)")
print("=" * 70)

personae_records = []
with open(f"{PERSONAE_DIR}/tc_llm_records.jsonl") as f:
    for line in f:
        personae_records.append(json.loads(line))

personae_groups = []
with open(f"{PERSONAE_DIR}/tc_llm_groups.jsonl") as f:
    for line in f:
        personae_groups.append(json.loads(line))

print(f"\nRecords: {len(personae_records)}, Groups: {len(personae_groups)}")
print(f"Unique personae: {sorted(set(r['persona'] for r in personae_records))}")

# Per-rep label entropy
prec_by_pair = collections.defaultdict(list)
for r in personae_records:
    key = (r["prompt_id"], r["persona"])
    prec_by_pair[key].extend(r.get("labels", []))

p_entropy_rows = []
for (prompt_id, persona), all_labels in prec_by_pair.items():
    h = label_entropy(all_labels)
    nh = normalized_label_entropy(all_labels)
    p_entropy_rows.append({
        "prompt_id": prompt_id,
        "persona": persona,
        "n_labels": len(all_labels),
        "n_unique": len(set(all_labels)),
        "entropy_bits": h,
        "norm_entropy": nh,
    })

p_edf = pd.DataFrame(p_entropy_rows)

# Load category info from personae embeddings
emb_df = pd.read_parquet(f"{PERSONAE_DIR}/embeddings_kalm-embedding.parquet")
cat_map = emb_df[["persona", "category"]].drop_duplicates().set_index("persona")["category"].to_dict()
p_edf["category"] = p_edf["persona"].map(cat_map)

print("\n--- Per-Prompt Entropy by Persona (mean over 245 prompts) ---")
p_persona_entropy = p_edf.groupby(["persona", "category"])[["n_unique", "entropy_bits", "norm_entropy"]].mean()
p_persona_entropy = p_persona_entropy.sort_values("entropy_bits")
print(p_persona_entropy.to_string(float_format="%.3f"))

print("\n--- Entropy Rank (lower = more consistent) ---")
pe_mean = p_edf.groupby("persona")["entropy_bits"].mean().sort_values()
for rank, (persona, h) in enumerate(pe_mean.items(), 1):
    cat = cat_map.get(persona, "?")
    marker = " <<< ASSISTANT" if persona == "assistant" else ""
    print(f"  {rank:2d}. {persona:<25s} [{cat:<22s}]  entropy={h:.3f} bits{marker}")

# Group stats
print("\n--- Merged Label Count by Persona (mean over 245 prompts) ---")
pgdf = []
for g in personae_groups:
    merged = g.get("merged_labels") or g.get("raw_labels", [])
    pgdf.append({
        "prompt_id": g["prompt_id"],
        "persona": g["persona"],
        "n_merged": len(set(merged)) if merged else 0,
        "n_raw": len(set(g.get("raw_labels", []))) if g.get("raw_labels") else 0,
    })
pgdf = pd.DataFrame(pgdf)
pg_mean = pgdf.groupby("persona")[["n_merged", "n_raw"]].mean().sort_values("n_merged")
print(pg_mean.to_string(float_format="%.1f"))
print("\n  (lower n_merged = fewer distinct themes = more consistent)")


# ─── Section 4: Embedding Variance (personae-inference-1855290) ──────────────

print("\n" + "=" * 70)
print("SECTION 4: EMBEDDING VARIANCE — personae-inference-1855290 (computed from embeddings)")
print("=" * 70)

# Compute variance from embeddings directly (embedding arrays per persona/prompt)
emb_models = ["kalm-embedding", "qwen3-embedding-0.6b"]
var_all = []
for model_name in emb_models:
    path = f"{PERSONAE_DIR}/embeddings_{model_name}.parquet"
    emb_df = pd.read_parquet(path)
    
    for col in ["embedding_thinking", "embedding_output"]:
        # Group by persona + prompt_id, compute variance of embedding vectors
        grp = emb_df.groupby(["persona", "prompt_id"])
        for (persona, pid), sub in grp:
            vecs = np.vstack(sub[col].values)  # shape (n_reps, d)
            var = np.var(vecs, axis=0).sum()   # total variance
            cat = sub["category"].iloc[0]
            var_all.append({
                "model": model_name,
                "embedding_col": col,
                "persona": persona,
                "category": cat,
                "prompt_id": pid,
                "total_variance": var,
            })

var_all_df = pd.DataFrame(var_all)
print(f"\nComputed variance rows: {len(var_all_df)}")

print("\n--- Mean Variance by Persona (avg over prompts and embedding models) ---")
pvar = var_all_df.groupby(["persona", "category", "embedding_col"])["total_variance"].mean().unstack()
pvar["mean"] = pvar.mean(axis=1)
pvar = pvar.sort_values("mean")
print(pvar.to_string(float_format="%.4f"))

print("\n--- Variance Rank (lower = more consistent) ---")
pvar_mean = var_all_df.groupby("persona")["total_variance"].mean().sort_values()
for rank, (persona, v) in enumerate(pvar_mean.items(), 1):
    cat = var_all_df[var_all_df["persona"] == persona]["category"].iloc[0]
    marker = " <<< ASSISTANT" if persona == "assistant" else ""
    print(f"  {rank:2d}. {persona:<25s} [{cat:<22s}]  var={v:.4f}{marker}")


# ─── Section 5: Cross-Source Summary ─────────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION 5: CROSS-SOURCE SUMMARY — Assistant Relative Position")
print("=" * 70)

# Collect all rankings
rankings = {}

# From section 1 (pipeline, embedding variance)
for col in ["embedding_thinking", "embedding_output"]:
    ranked = (
        var_df[var_df["embedding_column"] == col]
        .groupby("persona")["total_variance"]
        .mean()
        .sort_values()
    )
    rankings[f"pipeline_emb_var_{col}"] = {p: r+1 for r, p in enumerate(ranked.index)}

# From section 2 (pipeline, TC-LLM entropy)
ranked = persona_entropy["entropy_bits"].sort_values()
rankings["pipeline_tclm_entropy"] = {p: r+1 for r, p in enumerate(ranked.index)}

# From section 3 (personae, TC-LLM entropy)
ranked = pe_mean
rankings["personae_tclm_entropy"] = {p: r+1 for r, p in enumerate(ranked.index)}

# From section 4 (personae, embedding variance)
for col in ["embedding_thinking", "embedding_output"]:
    ranked = (
        var_all_df[var_all_df["embedding_col"] == col]
        .groupby("persona")["total_variance"]
        .mean()
        .sort_values()
    )
    rankings[f"personae_emb_var_{col}"] = {p: r+1 for r, p in enumerate(ranked.index)}

# Build summary table for the 7 shared personae
shared_personae = sorted(set(var_df["persona"].unique()))
print(f"\nShared personae (n={len(shared_personae)}): {shared_personae}")
print()

header = f"{'Persona':<25s}"
for k in rankings:
    header += f"  {k[-12:]}"
print(header)
print("-" * len(header))

rank_rows = []
for persona in shared_personae:
    row = {"persona": persona}
    vals = []
    for k, rank_map in rankings.items():
        r = rank_map.get(persona, None)
        vals.append(r)
        row[k] = r
    row["mean_rank"] = np.mean([v for v in vals if v is not None])
    rank_rows.append(row)

rank_df = pd.DataFrame(rank_rows).sort_values("mean_rank")

for _, r in rank_df.iterrows():
    persona = r["persona"]
    mr = r["mean_rank"]
    marker = " <<< ASSISTANT" if persona == "assistant" else ""
    print(f"{persona:<25s}  mean_rank={mr:.2f}{marker}")

print()
print("Interpretation: lower mean_rank = more consistent across metrics")

