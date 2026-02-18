"""
Streamlit webapp for visualizing persona inference results.

Supports five result types:
- metrics: entropy, top-k mass, thinking tokens from batch inference
- judge: endorsement/flagged analysis from LLM judge
- embeddings: embedding variance analysis from consistency experiments
- tc_llm: TC-LLM label entropy analysis from clustering experiments
- user_turn: user-turn prediction entropy, top-k mass, and token counts

Run with:
    uv run streamlit run src/viz/app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st
from scipy import stats

# Color scheme for endorsement types
ENDORSEMENT_COLORS = {
    "accept": "#2ecc71",    # Green
    "partial": "#f1c40f",   # Yellow
    "reject": "#e74c3c",    # Red
    "redirect": "#95a5a6",  # Gray
}

ENDORSEMENT_ORDER = ["accept", "partial", "reject", "redirect"]

st.set_page_config(
    page_title="Persona Experiment Results",
    page_icon="📊",
    layout="wide",
)


def detect_result_type(file_path: Path) -> str:
    """Detect if this is 'judge', 'metrics', 'embeddings', 'tc_llm', or 'user_turn' results.

    Args:
        file_path: Path to the results file

    Returns:
        'judge', 'metrics', 'embeddings', 'tc_llm', or 'user_turn'
    """
    if file_path.suffix == ".parquet":
        return "embeddings"
    if file_path.name == "judged_results.jsonl":
        return "judge"
    if file_path.name == "tc_llm_groups.jsonl":
        return "tc_llm"
    return "metrics"


def discover_result_files(logs_dir: str = "logs") -> list[tuple[Path, str]]:
    """Discover all recognized result files under logs/.

    Scans every subdirectory under logs/ and collects:
    - judged_results.jsonl -> "judge"
    - results.jsonl / results.json -> "metrics" (judged_results.jsonl takes priority)
    - embedding_variance*.parquet -> "embeddings"

    Returns:
        List of (file_path, result_type) tuples, sorted by directory name descending.
    """
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        return []

    result_files: list[tuple[Path, str]] = []

    for dir_path in sorted(logs_path.iterdir(), reverse=True):
        if not dir_path.is_dir():
            continue

        # Check for judge/metrics files (at most one per directory)
        judged = dir_path / "judged_results.jsonl"
        jsonl = dir_path / "results.jsonl"
        json_file = dir_path / "results.json"

        if judged.exists():
            result_files.append((judged, "judge"))
        elif jsonl.exists():
            if dir_path.name.startswith("user-turn-prediction-"):
                result_files.append((jsonl, "user_turn"))
            else:
                result_files.append((jsonl, "metrics"))
        elif json_file.exists():
            result_files.append((json_file, "metrics"))

        # Check for embedding variance files (can coexist with above)
        # Group all variance parquets under one directory-level entry
        variance_files = sorted(dir_path.glob("embedding_variance*.parquet"))
        if variance_files:
            result_files.append((dir_path, "embeddings"))

        # Check for TC-LLM groups file (can coexist with above)
        tc_llm_groups = dir_path / "tc_llm_groups.jsonl"
        if tc_llm_groups.exists():
            result_files.append((tc_llm_groups, "tc_llm"))

    return result_files


def format_file_label(file_path: Path, result_type: str) -> str:
    """Format a file path + result type for display in the file selector.

    Examples:
        consistency-inference-1225210 [embeddings]
        consistency-judge-pipeline-20260209_125904 [embeddings: kalm-embedding]
        prefill-inference-1472066 [judge]
        personae-inference-1855290 [metrics]
    """
    if result_type == "embeddings":
        # file_path is a directory for embeddings type
        dir_name = file_path.name
        variance_files = sorted(file_path.glob("embedding_variance*.parquet"))
        model_names = []
        for vf in variance_files:
            prefix = "embedding_variance"
            if vf.stem.startswith(prefix + "_"):
                model_names.append(vf.stem[len(prefix) + 1:])
        if len(model_names) == 0:
            return f"{dir_name} [embeddings]"
        elif len(model_names) == 1:
            return f"{dir_name} [embeddings: {model_names[0]}]"
        else:
            return f"{dir_name} [embeddings: {len(model_names)} models]"

    dir_name = file_path.parent.name

    if result_type == "judge":
        return f"{dir_name} [judge]"
    elif result_type == "tc_llm":
        return f"{dir_name} [tc-llm]"
    elif result_type == "user_turn":
        return f"{dir_name} [user-turn]"
    else:
        return f"{dir_name} [metrics]"


def load_results(file_path: Path) -> pd.DataFrame:
    """Load results from JSON or JSONL into a DataFrame.

    Args:
        file_path: Path to results.json or results.jsonl file

    Returns:
        DataFrame with results data
    """
    if file_path.suffix == ".jsonl":
        # JSONL: one JSON object per line
        data = []
        with open(file_path) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return pd.DataFrame(data)
    else:
        # JSON: array of objects
        with open(file_path) as f:
            data = json.load(f)
        return pd.DataFrame(data)


def flatten_judge_parsed(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the judge_parsed nested dict into top-level columns.

    Args:
        df: DataFrame with judge_parsed column containing dicts

    Returns:
        DataFrame with endorsement, flagged, reasoning as separate columns
    """
    if "judge_parsed" not in df.columns:
        return df

    # Extract nested fields
    df = df.copy()
    df["endorsement"] = df["judge_parsed"].apply(
        lambda x: x.get("endorsement") if isinstance(x, dict) else None
    )
    df["flagged"] = df["judge_parsed"].apply(
        lambda x: x.get("flagged") if isinstance(x, dict) else None
    )
    # Convert flagged to boolean, treating None as False
    df["flagged"] = df["flagged"].fillna(False).astype(bool)
    df["reasoning"] = df["judge_parsed"].apply(
        lambda x: x.get("reasoning") if isinstance(x, dict) else None
    )
    return df


@st.cache_data
def load_variance_data(file_path: str) -> pd.DataFrame:
    """Load precomputed variance parquet file."""
    return pd.read_parquet(file_path)


def _derive_canonical_prompt_id(dir_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Add canonical_prompt_id and prompt_text to variance data.

    If canonical_prompt_id is already present in the raw embeddings parquet,
    uses that directly. Otherwise, derives canonical grouping:

    - Persona-specific prompts (prompt_id maps to exactly one persona) are
      grouped by their category (e.g. all "core_competence" prompts share
      one canonical ID).
    - Shared prompts (prompt_id maps to multiple personas) each get their
      own canonical ID.

    Args:
        dir_path: Directory containing both variance and raw embeddings parquets
        df: Variance DataFrame with prompt_id column

    Returns:
        DataFrame with canonical_prompt_id and prompt_text columns added
    """
    if "canonical_prompt_id" in df.columns and "prompt_text" in df.columns:
        return df

    # Load prompt_id -> prompt text mapping from the first raw embeddings parquet
    embeddings_files = sorted(dir_path.glob("embeddings_*.parquet")) + sorted(
        dir_path.glob("embeddings.parquet")
    )
    if not embeddings_files:
        # No raw embeddings to derive from; fall back to prompt_id as canonical
        if "canonical_prompt_id" not in df.columns:
            df["canonical_prompt_id"] = df["prompt_id"]
        if "prompt_text" not in df.columns:
            df["prompt_text"] = df["prompt_id"].apply(lambda x: f"Prompt {x}")
        return df

    # Read only metadata columns (not the large embedding vectors).
    # Check parquet schema first to avoid requesting non-existent columns.
    schema = pq.read_schema(embeddings_files[0])
    available_cols = set(schema.names)
    read_cols = ["prompt_id", "prompt", "persona"]
    if "category" in available_cols:
        read_cols.append("category")
    has_canonical_in_raw = "canonical_prompt_id" in available_cols
    if has_canonical_in_raw and "canonical_prompt_id" not in df.columns:
        read_cols.append("canonical_prompt_id")

    raw = pd.read_parquet(embeddings_files[0], columns=read_cols)
    pid_to_text = dict(raw[["prompt_id", "prompt"]].drop_duplicates().values)

    if "canonical_prompt_id" not in df.columns:
        if has_canonical_in_raw:
            pid_to_canonical = dict(
                raw[["prompt_id", "canonical_prompt_id"]].drop_duplicates().values
            )
        else:
            # Derive canonical grouping:
            # - persona-specific prompt_ids (1 persona) -> group by category
            # - shared prompt_ids (multiple personas) -> one canonical per text
            personas_per_pid = raw.groupby("prompt_id")["persona"].nunique()
            pid_to_category = {}
            if "category" in raw.columns:
                pid_to_category = dict(
                    raw[["prompt_id", "category"]].drop_duplicates().values
                )

            canonical_id = 0
            category_to_canonical: dict[str, int] = {}
            text_to_canonical: dict[str, int] = {}
            pid_to_canonical: dict[int, int] = {}

            for pid in sorted(pid_to_text.keys()):
                n_personas = int(personas_per_pid.get(pid, 1))
                if n_personas == 1 and pid in pid_to_category:
                    # Persona-specific: group by category
                    cat = pid_to_category[pid]
                    if cat not in category_to_canonical:
                        category_to_canonical[cat] = canonical_id
                        canonical_id += 1
                    pid_to_canonical[pid] = category_to_canonical[cat]
                else:
                    # Shared prompt: one canonical per unique text
                    text = pid_to_text[pid]
                    if text not in text_to_canonical:
                        text_to_canonical[text] = canonical_id
                        canonical_id += 1
                    pid_to_canonical[pid] = text_to_canonical[text]

        df["canonical_prompt_id"] = df["prompt_id"].map(pid_to_canonical)

    # Build prompt_text for each canonical ID.
    # For grouped persona-specific prompts (same canonical, different text),
    # use the category name as the display label.
    canonical_to_text: dict[int, str] = {}
    canonical_to_texts: dict[int, set[str]] = {}
    for pid, text in pid_to_text.items():
        cid = df.loc[df["prompt_id"] == pid, "canonical_prompt_id"]
        if not cid.empty:
            cid_val = int(cid.iloc[0])
            canonical_to_texts.setdefault(cid_val, set()).add(text)

    pid_to_category = {}
    if "category" in raw.columns:
        pid_to_category = dict(
            raw[["prompt_id", "category"]].drop_duplicates().values
        )

    for cid, texts in canonical_to_texts.items():
        if len(texts) == 1:
            canonical_to_text[cid] = next(iter(texts))
        else:
            # Multiple different texts grouped together — use category as label
            # Find the category from any prompt_id with this canonical_id
            matching_pids = df.loc[
                df["canonical_prompt_id"] == cid, "prompt_id"
            ].unique()
            cat = None
            for pid in matching_pids:
                if pid in pid_to_category:
                    cat = pid_to_category[pid]
                    break
            canonical_to_text[cid] = (
                f"[{cat}] (persona-specific)" if cat else f"Grouped prompt {cid}"
            )

    df["prompt_text"] = df["canonical_prompt_id"].map(canonical_to_text)

    return df


@st.cache_data
def load_multi_model_variance_data(dir_path: str) -> pd.DataFrame:
    """Load and concatenate all embedding_variance*.parquet files in a directory.

    Adds a 'model' column extracted from the filename suffix. Files named
    'embedding_variance.parquet' (no suffix) get model='default'.

    Also derives canonical_prompt_id (grouping prompt_ids that share the same
    prompt text) from raw embeddings parquets in the same directory.

    Args:
        dir_path: Path to directory containing embedding_variance*.parquet files

    Returns:
        Concatenated DataFrame with added 'model', 'canonical_prompt_id',
        and 'prompt_text' columns
    """
    dir_path = Path(dir_path)
    variance_files = sorted(dir_path.glob("embedding_variance*.parquet"))
    assert len(variance_files) > 0, f"No variance parquets found in {dir_path}"

    dfs = []
    prefix = "embedding_variance"
    for vf in variance_files:
        df = pd.read_parquet(vf)
        if vf.stem.startswith(prefix + "_"):
            model_name = vf.stem[len(prefix) + 1:]
        else:
            model_name = "default"
        df["model"] = model_name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = _derive_canonical_prompt_id(dir_path, combined)
    return combined


def build_category_color_map(df: pd.DataFrame) -> dict[str, str]:
    """Build a consistent color map for categories.

    Args:
        df: DataFrame with a 'category' column

    Returns:
        Dictionary mapping category names to hex colors
    """
    if "category" not in df.columns:
        return {}

    categories = sorted(df["category"].unique())
    colors = px.colors.qualitative.Plotly
    return {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}


def create_variance_bar_chart(
    df: pd.DataFrame,
    prompt_id: int,
    question_text: str,
    has_category: bool = False,
    color_map: dict[str, str] | None = None,
    persona_order: list[str] | None = None,
) -> go.Figure:
    """Create a bar chart of total variance per persona for a specific prompt.

    Args:
        df: DataFrame with columns: prompt_id, persona, total_variance, and optionally category
        prompt_id: The prompt ID to filter for
        question_text: The question text to display in the title
        has_category: Whether to color by category
        color_map: Optional dictionary mapping category names to colors for consistency
        persona_order: Optional list of persona names defining x-axis order

    Returns:
        Plotly figure with bar chart
    """
    prompt_df = df[df["prompt_id"] == prompt_id]

    # Truncate question text for title
    title_text = question_text[:80] + "..." if len(question_text) > 80 else question_text

    fig = px.bar(
        prompt_df,
        x="persona",
        y="total_variance",
        color="category" if has_category else None,
        color_discrete_map=color_map if has_category and color_map else None,
        title=f"Prompt {prompt_id}: {title_text}",
        labels={
            "total_variance": "Total Variance (Trace of Cov)",
            "persona": "Persona",
            "category": "Category",
        },
        category_orders={"persona": persona_order} if persona_order else None,
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
    )
    return fig


def calculate_category_rankings(df: pd.DataFrame, prompt_id: int) -> pd.DataFrame:
    """Calculate variance rank per category (averaged across personas) for each embedding column.

    Args:
        df: DataFrame with columns: prompt_id, persona, embedding_column, total_variance, category
        prompt_id: The prompt ID to calculate rankings for

    Returns:
        DataFrame with columns: category, embedding_column, avg_variance, rank
        Rank 1 = lowest average variance (will be plotted at top due to reversed y-axis)
    """
    prompt_df = df[df["prompt_id"] == prompt_id].copy()

    # Calculate average variance per category within each embedding column
    category_variance = (
        prompt_df.groupby(["embedding_column", "category"])["total_variance"]
        .mean()
        .reset_index()
        .rename(columns={"total_variance": "avg_variance"})
    )

    # Calculate rank within each embedding column (rank 1 = lowest variance)
    category_variance["rank"] = category_variance.groupby("embedding_column")[
        "avg_variance"
    ].rank(ascending=True, method="min").astype(int)

    return category_variance


def create_ranking_line_chart(
    df: pd.DataFrame,
    prompt_id: int,
    question_text: str,
    color_map: dict[str, str] | None = None,
) -> go.Figure:
    """Create line chart showing category rank changes across embedding columns.

    Args:
        df: Full DataFrame with all data (must have 'category' column)
        prompt_id: The prompt ID to create chart for
        question_text: The question text to display in the title
        color_map: Optional dictionary mapping category names to colors for consistency

    Returns:
        Plotly figure with line chart (one line per category)
    """
    ranking_df = calculate_category_rankings(df, prompt_id)

    # Truncate question text for title
    title_text = question_text[:80] + "..." if len(question_text) > 80 else question_text

    # Define column order for x-axis
    column_order = [
        "response1_embeddings_full",
        "response1_embeddings_thinking",
        "response1_embeddings_output",
        "response2_embeddings_full",
        "response2_embeddings_thinking",
        "response2_embeddings_output",
    ]
    # Filter to only columns that exist in the data
    existing_columns = [c for c in column_order if c in ranking_df["embedding_column"].unique()]
    # Add any columns not in our predefined order
    for col in ranking_df["embedding_column"].unique():
        if col not in existing_columns:
            existing_columns.append(col)

    fig = px.line(
        ranking_df,
        x="embedding_column",
        y="rank",
        color="category",
        markers=True,
        title=f"Prompt {prompt_id}: {title_text}",
        labels={
            "rank": "Category Variance Rank",
            "embedding_column": "Embedding Column",
            "category": "Category",
        },
        category_orders={"embedding_column": existing_columns},
        color_discrete_map=color_map,
    )

    fig.update_layout(
        height=500,
        yaxis=dict(autorange="reversed"),  # Rank 1 at top
        xaxis_tickangle=-45,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
        ),
    )

    return fig


def create_multi_model_variance_bar_chart(
    df: pd.DataFrame,
    prompt_id: int,
    question_text: str,
    color_map: dict[str, str] | None = None,
    persona_order: list[str] | None = None,
) -> go.Figure:
    """Create a grouped bar chart of variance per persona, colored by model.

    Args:
        df: DataFrame with columns: prompt_id, persona, total_variance, model, and optionally category
        prompt_id: The prompt ID to filter for
        question_text: The question text to display in the title
        color_map: Unused (kept for API consistency); models get auto colors
        persona_order: Optional list of persona names defining x-axis order

    Returns:
        Plotly figure with grouped bar chart
    """
    prompt_df = df[df["prompt_id"] == prompt_id]

    title_text = question_text[:80] + "..." if len(question_text) > 80 else question_text

    fig = px.bar(
        prompt_df,
        x="persona",
        y="total_variance",
        color="model",
        barmode="group",
        title=f"Prompt {prompt_id}: {title_text}",
        labels={
            "total_variance": "Total Variance (Trace of Cov)",
            "persona": "Persona",
            "model": "Model",
        },
        category_orders={"persona": persona_order} if persona_order else None,
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
    )
    return fig


def create_model_comparison_bar_chart(
    df: pd.DataFrame,
    group_by: str | None = None,
) -> go.Figure:
    """Create bar chart of mean variance per model, optionally grouped by category.

    Args:
        df: DataFrame with columns: model, total_variance, and optionally category
        group_by: If "category", show grouped bars by category; otherwise aggregate all

    Returns:
        Plotly figure
    """
    if group_by == "category" and "category" in df.columns:
        agg = (
            df.groupby(["model", "category"])["total_variance"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg["sem"] = agg["std"] / np.sqrt(agg["count"])

        fig = px.bar(
            agg,
            x="category",
            y="mean",
            color="model",
            barmode="group",
            error_y="sem",
            title="Mean Variance by Model and Category",
            labels={
                "mean": "Mean Total Variance",
                "category": "Category",
                "model": "Model",
            },
        )
    else:
        agg = (
            df.groupby("model")["total_variance"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg["sem"] = agg["std"] / np.sqrt(agg["count"])

        fig = px.bar(
            agg,
            x="model",
            y="mean",
            error_y="sem",
            title="Mean Variance by Model",
            labels={
                "mean": "Mean Total Variance",
                "model": "Model",
            },
        )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
    )
    return fig


def create_model_ratio_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create a heatmap of log2 variance ratios between models per persona.

    For each persona, computes log2(mean_variance_A / mean_variance_B) for all
    model pairs. Positive = model A has higher variance.

    Args:
        df: DataFrame with columns: model, persona, total_variance

    Returns:
        Plotly heatmap figure
    """
    # Mean variance per (model, persona)
    pivot = (
        df.groupby(["model", "persona"])["total_variance"]
        .mean()
        .reset_index()
        .pivot(index="persona", columns="model", values="total_variance")
    )

    models = sorted(pivot.columns)
    assert len(models) >= 2, "Need at least 2 models for ratio heatmap"

    # Compute pairwise log2 ratios: rows = personas, columns = model pairs
    pair_labels = []
    ratio_data = {}
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i >= j:
                continue
            label = f"{m1} / {m2}"
            pair_labels.append(label)
            ratio_data[label] = np.log2(pivot[m1] / pivot[m2])

    ratio_df = pd.DataFrame(ratio_data, index=pivot.index)

    fig = go.Figure(data=go.Heatmap(
        z=ratio_df.values,
        x=ratio_df.columns.tolist(),
        y=ratio_df.index.tolist(),
        text=ratio_df.round(2).astype(str).values,
        texttemplate="%{text}",
        textfont={"size": 11},
        colorscale="RdBu_r",
        zmid=0,
        colorbar=dict(title="log2 ratio"),
    ))

    fig.update_layout(
        title="Log2 Variance Ratio Between Models (per Persona)",
        height=max(400, 30 * len(pivot.index)),
        xaxis_title="Model Pair",
        yaxis_title="Persona",
    )
    return fig


def create_model_rank_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman rank correlation of persona variance rankings between models.

    For each model, ranks personas by mean variance. Then computes pairwise
    Spearman correlation of these rankings.

    Args:
        df: DataFrame with columns: model, persona, total_variance

    Returns:
        Correlation DataFrame (models x models)
    """
    # Mean variance per (model, persona)
    pivot = (
        df.groupby(["model", "persona"])["total_variance"]
        .mean()
        .reset_index()
        .pivot(index="persona", columns="model", values="total_variance")
    )

    models = sorted(pivot.columns)
    n = len(models)
    corr_matrix = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            valid = pivot[[models[i], models[j]]].dropna()
            if len(valid) > 2:
                r, _ = stats.spearmanr(valid[models[i]], valid[models[j]])
                corr_matrix[i, j] = r
                corr_matrix[j, i] = r
            else:
                corr_matrix[i, j] = np.nan
                corr_matrix[j, i] = np.nan

    return pd.DataFrame(corr_matrix, index=models, columns=models)


def create_model_scatter(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    color_map: dict[str, str] | None = None,
) -> go.Figure:
    """Create scatter plot of model A vs model B variance per (prompt, persona) point.

    Args:
        df: DataFrame with columns: model, persona, prompt_id, total_variance, category
        model_a: Name of the first model (x-axis)
        model_b: Name of the second model (y-axis)
        color_map: Optional category color map

    Returns:
        Plotly figure
    """
    df_a = df[df["model"] == model_a][["prompt_id", "persona", "total_variance", "category"]].rename(
        columns={"total_variance": "var_a"}
    )
    df_b = df[df["model"] == model_b][["prompt_id", "persona", "total_variance"]].rename(
        columns={"total_variance": "var_b"}
    )

    merged = df_a.merge(df_b, on=["prompt_id", "persona"], how="inner")

    has_category = "category" in merged.columns
    fig = px.scatter(
        merged,
        x="var_a",
        y="var_b",
        color="category" if has_category else None,
        color_discrete_map=color_map if has_category and color_map else None,
        hover_data=["persona", "prompt_id"],
        title=f"Variance: {model_a} vs {model_b}",
        labels={
            "var_a": f"Variance ({model_a})",
            "var_b": f"Variance ({model_b})",
            "category": "Category",
        },
    )

    # Add diagonal reference line
    all_vals = pd.concat([merged["var_a"], merged["var_b"]])
    line_max = all_vals.max()
    fig.add_trace(go.Scatter(
        x=[0, line_max],
        y=[0, line_max],
        mode="lines",
        line=dict(color="gray", dash="dash", width=1),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.update_layout(
        height=600,
        xaxis=dict(scaleanchor="y", scaleratio=1),
    )
    return fig


def create_entropy_chart(
    df: pd.DataFrame, group_by: str = "category", plot_type: str = "box"
) -> go.Figure:
    """Create box or violin plot for entropy metrics per category or persona."""
    assert group_by in ("category", "persona"), f"Invalid group_by: {group_by}"
    assert plot_type in ("box", "violin"), f"Invalid plot_type: {plot_type}"

    # Melt the dataframe to get entropy metrics in long format
    entropy_cols = ["avg_entropy_thinking", "avg_entropy_output", "avg_entropy"]
    melted = df.melt(
        id_vars=["category", "persona", "prompt_id"],
        value_vars=entropy_cols,
        var_name="metric",
        value_name="entropy",
    )

    # Clean up metric names for display
    metric_labels = {
        "avg_entropy_thinking": "Thinking",
        "avg_entropy_output": "Output",
        "avg_entropy": "Overall",
    }
    melted["metric"] = melted["metric"].map(metric_labels)

    x_label = "Persona Category" if group_by == "category" else "Persona"
    title = f"Entropy by {x_label}"

    plot_fn = px.violin if plot_type == "violin" else px.box
    fig = plot_fn(
        melted,
        x=group_by,
        y="entropy",
        color="metric",
        title=title,
        labels={"entropy": "Entropy", group_by: x_label, "metric": "Section"},
        category_orders={"metric": ["Thinking", "Output", "Overall"]},
    )
    mode_key = "violinmode" if plot_type == "violin" else "boxmode"
    fig.update_layout(
        **{mode_key: "group"},
        xaxis_tickangle=-45,
        height=900,
    )
    # Show mean in addition to median
    if plot_type == "violin":
        fig.update_traces(meanline_visible=True)
    else:
        fig.update_traces(boxmean=True)
    return fig


def create_top_k_mass_chart(
    df: pd.DataFrame, group_by: str = "category", plot_type: str = "box"
) -> go.Figure:
    """Create box or violin plot for top-k mass metrics per category or persona."""
    assert group_by in ("category", "persona"), f"Invalid group_by: {group_by}"
    assert plot_type in ("box", "violin"), f"Invalid plot_type: {plot_type}"

    # Melt the dataframe to get top-k mass metrics in long format
    top_k_cols = ["avg_top_k_mass_thinking", "avg_top_k_mass_output", "avg_top_k_mass"]
    melted = df.melt(
        id_vars=["category", "persona", "prompt_id"],
        value_vars=top_k_cols,
        var_name="metric",
        value_name="top_k_mass",
    )

    # Clean up metric names for display
    metric_labels = {
        "avg_top_k_mass_thinking": "Thinking",
        "avg_top_k_mass_output": "Output",
        "avg_top_k_mass": "Overall",
    }
    melted["metric"] = melted["metric"].map(metric_labels)

    x_label = "Persona Category" if group_by == "category" else "Persona"
    title = f"Top-k Mass by {x_label}"

    plot_fn = px.violin if plot_type == "violin" else px.box
    fig = plot_fn(
        melted,
        x=group_by,
        y="top_k_mass",
        color="metric",
        title=title,
        labels={"top_k_mass": "Top-k Mass", group_by: x_label, "metric": "Section"},
        category_orders={"metric": ["Thinking", "Output", "Overall"]},
    )
    mode_key = "violinmode" if plot_type == "violin" else "boxmode"
    fig.update_layout(
        **{mode_key: "group"},
        xaxis_tickangle=-45,
        height=900,
    )
    # Show mean in addition to median
    if plot_type == "violin":
        fig.update_traces(meanline_visible=True)
    else:
        fig.update_traces(boxmean=True)
    return fig


def create_thinking_tokens_chart(
    df: pd.DataFrame, group_by: str = "category", plot_type: str = "box"
) -> go.Figure:
    """Create box or violin plot for thinking tokens per category or persona."""
    assert group_by in ("category", "persona"), f"Invalid group_by: {group_by}"
    assert plot_type in ("box", "violin"), f"Invalid plot_type: {plot_type}"

    x_label = "Persona Category" if group_by == "category" else "Persona"
    title = f"Number of Thinking Tokens by {x_label}"

    plot_fn = px.violin if plot_type == "violin" else px.box
    fig = plot_fn(
        df,
        x=group_by,
        y="think_end_position",
        title=title,
        labels={"think_end_position": "Thinking Tokens", group_by: x_label},
        color=group_by,
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=900,
        showlegend=False,
    )
    # Show mean in addition to median
    if plot_type == "violin":
        fig.update_traces(meanline_visible=True)
    else:
        fig.update_traces(boxmean=True)
    return fig


def create_endorsement_chart(
    df: pd.DataFrame,
    group_by: str = "persona",
    normalize: bool = True,
    sort_by_accept: bool = False,
) -> go.Figure:
    """Create stacked bar chart showing endorsement distribution.

    Args:
        df: DataFrame with endorsement column
        group_by: Column to group by ('persona' or 'category')
        normalize: If True, show percentages; if False, show counts
        sort_by_accept: If True, sort groups by accept rate (descending)

    Returns:
        Plotly figure
    """
    assert group_by in ("persona", "category"), f"Invalid group_by: {group_by}"

    # Count endorsements per group
    counts = df.groupby([group_by, "endorsement"]).size().unstack(fill_value=0)

    # Ensure all endorsement types are present
    for etype in ENDORSEMENT_ORDER:
        if etype not in counts.columns:
            counts[etype] = 0
    counts = counts[ENDORSEMENT_ORDER]

    # Sort by accept rate if requested
    if sort_by_accept and "accept" in counts.columns:
        # Calculate accept rate for sorting
        totals = counts.sum(axis=1)
        accept_rate = counts["accept"] / totals
        counts = counts.loc[accept_rate.sort_values(ascending=False).index]

    if normalize:
        # Convert to percentages
        counts = counts.div(counts.sum(axis=1), axis=0) * 100

    # Create stacked bar chart
    fig = go.Figure()

    for endorsement in ENDORSEMENT_ORDER:
        fig.add_trace(go.Bar(
            name=endorsement.capitalize(),
            x=counts.index,
            y=counts[endorsement],
            marker_color=ENDORSEMENT_COLORS[endorsement],
        ))

    x_label = "Persona" if group_by == "persona" else "Category"
    y_label = "Percentage" if normalize else "Count"
    title = f"Endorsement Distribution by {x_label}"

    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis_tickangle=-45,
        height=600,
        legend_title="Endorsement",
    )

    return fig


def create_flagged_rate_chart(
    df: pd.DataFrame,
    group_by: str = "persona",
) -> go.Figure:
    """Create bar chart showing flagged rate per group.

    Args:
        df: DataFrame with flagged column
        group_by: Column to group by ('persona' or 'category')

    Returns:
        Plotly figure
    """
    assert group_by in ("persona", "category"), f"Invalid group_by: {group_by}"

    # Calculate flagged rate per group
    flagged_rate = df.groupby(group_by)["flagged"].mean() * 100
    flagged_rate = flagged_rate.sort_values(ascending=False)

    # Create bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=flagged_rate.index,
        y=flagged_rate.values,
        marker_color="#e74c3c",
        name="Flagged Rate",
    ))

    x_label = "Persona" if group_by == "persona" else "Category"
    title = f"Flagged Rate by {x_label}"

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Flagged Rate (%)",
        xaxis_tickangle=-45,
        height=500,
        showlegend=False,
    )

    return fig


def create_endorsement_when_flagged_chart(
    df: pd.DataFrame,
    group_by: str = "persona",
) -> go.Figure:
    """Create endorsement chart for only flagged responses.

    Args:
        df: DataFrame with flagged and endorsement columns
        group_by: Column to group by ('persona' or 'category')

    Returns:
        Plotly figure
    """
    flagged_df = df[df["flagged"] == True]  # noqa: E712
    if len(flagged_df) == 0:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No flagged responses in the current selection",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(height=400)
        return fig

    fig = create_endorsement_chart(flagged_df, group_by, normalize=True)
    x_label = "Persona" if group_by == "persona" else "Category"
    fig.update_layout(title=f"Endorsement Distribution (Flagged Only) by {x_label}")
    return fig


def parse_thinking_and_output(response: str) -> tuple[str, str]:
    """Parse a response into thinking and output sections.

    Args:
        response: The full response string

    Returns:
        Tuple of (thinking_content, output_content)
    """
    if not isinstance(response, str):
        return "", ""

    if "<think>" in response and "</think>" in response:
        # Extract thinking section
        think_start = response.find("<think>") + len("<think>")
        think_end = response.find("</think>")
        thinking = response[think_start:think_end].strip()
        output = response[think_end + len("</think>"):].strip()
        return thinking, output
    else:
        # No thinking tags, treat entire response as output
        return "", response


def compute_response_length_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute thinking and output character counts per row.

    Args:
        df: DataFrame with response column

    Returns:
        DataFrame with thinking_chars and output_chars columns added
    """
    df = df.copy()

    # Parse responses to get thinking and output
    parsed = df["response"].apply(parse_thinking_and_output)
    df["thinking_chars"] = parsed.apply(lambda x: len(x[0]))
    df["output_chars"] = parsed.apply(lambda x: len(x[1]))

    return df


def create_response_length_chart(
    df: pd.DataFrame,
    group_by: str = "persona",
    sort_by_accept: bool = False,
) -> go.Figure:
    """Create grouped bar chart showing avg thinking vs output chars per group.

    Args:
        df: DataFrame with response column
        group_by: Column to group by ('persona' or 'category')
        sort_by_accept: If True, sort by accept rate (requires endorsement column)

    Returns:
        Plotly figure
    """
    assert group_by in ("persona", "category"), f"Invalid group_by: {group_by}"

    # Compute character counts
    df = compute_response_length_stats(df)

    # Compute averages per group
    length_stats = df.groupby(group_by)[["thinking_chars", "output_chars"]].mean()

    # Sort by accept rate if requested
    if sort_by_accept and "endorsement" in df.columns:
        counts = df.groupby([group_by, "endorsement"]).size().unstack(fill_value=0)
        if "accept" in counts.columns:
            totals = counts.sum(axis=1)
            accept_rate = counts["accept"] / totals
            length_stats = length_stats.loc[accept_rate.sort_values(ascending=False).index]

    # Create grouped bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Thinking",
        x=length_stats.index,
        y=length_stats["thinking_chars"],
        marker_color="#3498db",  # Blue
    ))

    fig.add_trace(go.Bar(
        name="Output",
        x=length_stats.index,
        y=length_stats["output_chars"],
        marker_color="#9b59b6",  # Purple
    ))

    x_label = "Persona" if group_by == "persona" else "Category"
    title = f"Average Response Length by {x_label}"

    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title=x_label,
        yaxis_title="Characters",
        xaxis_tickangle=-45,
        height=600,
        legend_title="Section",
    )

    return fig


def compute_correlations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Pearson correlations and p-values for key metrics.

    Returns:
        Tuple of (correlation_matrix, p_value_matrix)
    """
    metrics = ["avg_entropy", "avg_top_k_mass", "think_end_position"]
    metric_labels = {
        "avg_entropy": "Entropy",
        "avg_top_k_mass": "Top-k Mass",
        "think_end_position": "Thinking Tokens",
    }

    n = len(metrics)
    corr_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))

    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            if i == j:
                corr_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
            else:
                # Drop NaN values for this pair
                valid_mask = df[m1].notna() & df[m2].notna()
                if valid_mask.sum() > 2:
                    r, p = stats.pearsonr(df.loc[valid_mask, m1], df.loc[valid_mask, m2])
                    corr_matrix[i, j] = r
                    p_matrix[i, j] = p
                else:
                    corr_matrix[i, j] = np.nan
                    p_matrix[i, j] = np.nan

    labels = [metric_labels[m] for m in metrics]
    corr_df = pd.DataFrame(corr_matrix, index=labels, columns=labels)
    p_df = pd.DataFrame(p_matrix, index=labels, columns=labels)

    return corr_df, p_df


def create_correlation_heatmap(corr_df: pd.DataFrame, title: str) -> go.Figure:
    """Create a correlation heatmap with annotations."""
    # Create text annotations showing correlation values
    text_annotations = corr_df.round(3).astype(str).values

    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns.tolist(),
        y=corr_df.index.tolist(),
        text=text_annotations,
        texttemplate="%{text}",
        textfont={"size": 14},
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlation"),
    ))

    fig.update_layout(
        title=title,
        height=500,
        xaxis=dict(side="bottom"),
    )

    return fig


def render_response_explorer(df: pd.DataFrame, page_size: int = 20) -> None:
    """Render the response explorer tab with paginated response cards.

    Args:
        df: Filtered DataFrame with judge results
        page_size: Number of responses per page
    """
    st.subheader("Response Explorer")

    if len(df) == 0:
        st.info("No responses match the current filters.")
        return

    # Pagination
    total_responses = len(df)
    total_pages = (total_responses + page_size - 1) // page_size

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        page = st.number_input(
            f"Page (1-{total_pages})",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
        )

    st.write(f"Showing responses {(page-1)*page_size + 1}-{min(page*page_size, total_responses)} of {total_responses}")

    # Get current page of data
    start_idx = (page - 1) * page_size
    end_idx = min(page * page_size, total_responses)
    page_df = df.iloc[start_idx:end_idx]

    # Display each response as an expander
    for idx, row in page_df.iterrows():
        # Create header with badges
        endorsement = row.get("endorsement", "unknown")
        flagged = row.get("flagged", False)
        persona = row.get("persona", "unknown")
        category = row.get("category", "unknown")

        # Color-coded endorsement badge
        endorsement_color = ENDORSEMENT_COLORS.get(endorsement, "#999")
        flagged_badge = "🚩 Flagged" if flagged else ""

        header = f"**{persona}** ({category}) | {endorsement.upper()} {flagged_badge}"

        with st.expander(header, expanded=False):
            # Badges row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"**Persona:** {persona}")
            with col2:
                st.markdown(f"**Category:** {category}")
            with col3:
                st.markdown(
                    f"**Endorsement:** <span style='color:{endorsement_color};font-weight:bold'>{endorsement}</span>",
                    unsafe_allow_html=True,
                )
            with col4:
                if flagged:
                    st.markdown("**Flagged:** 🚩 Yes")
                else:
                    st.markdown("**Flagged:** No")

            st.markdown("---")

            # Prompt
            st.markdown("**Prompt:**")
            st.text(row.get("prompt", "N/A"))

            # Prefill (highlighted)
            prefill = row.get("prefill", "")
            if prefill:
                st.markdown("**Prefill (injected):**")
                st.info(prefill)

            # Follow-up question
            followup = row.get("followup", "")
            if followup:
                st.markdown("**Follow-up Question:**")
                st.text(followup)

            # Model response
            st.markdown("**Model Response:**")
            response = row.get("response", "N/A")
            # Style thinking section differently if present
            if "<think>" in response and "</think>" in response:
                parts = response.split("</think>", 1)
                thinking = parts[0].replace("<think>", "").strip()
                output = parts[1].strip() if len(parts) > 1 else ""

                with st.container():
                    st.markdown("*Thinking:*")
                    st.text_area(
                        "thinking_content",
                        thinking,
                        height=150,
                        disabled=True,
                        label_visibility="collapsed",
                    )
                    if output:
                        st.markdown("*Output:*")
                        st.text(output)
            else:
                st.text(response)

            # Judge reasoning
            reasoning = row.get("reasoning", "")
            if reasoning:
                st.markdown("**Judge Reasoning:**")
                st.warning(reasoning)


def render_judge_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filters for judge mode and return filtered DataFrame.

    Args:
        df: DataFrame with judge results

    Returns:
        Filtered DataFrame based on sidebar selections
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")

    # Persona filter
    personas = sorted(df["persona"].unique())
    selected_personas = st.sidebar.multiselect(
        "Filter by persona",
        options=personas,
        default=personas,
    )

    # Category filter
    categories = sorted(df["category"].unique())
    selected_categories = st.sidebar.multiselect(
        "Filter by category",
        options=categories,
        default=categories,
    )

    # Endorsement filter
    endorsements = [e for e in ENDORSEMENT_ORDER if e in df["endorsement"].unique()]
    selected_endorsements = st.sidebar.multiselect(
        "Filter by endorsement",
        options=endorsements,
        default=endorsements,
    )

    # Flagged filter
    flagged_filter = st.sidebar.radio(
        "Filter by flagged",
        options=["All", "Flagged only", "Not flagged only"],
        index=0,
    )

    # Apply filters
    filtered = df.copy()

    if selected_personas:
        filtered = filtered[filtered["persona"].isin(selected_personas)]

    if selected_categories:
        filtered = filtered[filtered["category"].isin(selected_categories)]

    if selected_endorsements:
        filtered = filtered[filtered["endorsement"].isin(selected_endorsements)]

    if flagged_filter == "Flagged only":
        filtered = filtered[filtered["flagged"] == True]  # noqa: E712
    elif flagged_filter == "Not flagged only":
        filtered = filtered[filtered["flagged"] == False]  # noqa: E712

    return filtered


def render_judge_view(df: pd.DataFrame) -> None:
    """Render the judge results visualization view.

    Args:
        df: DataFrame with judge results (already flattened)
    """
    # Display basic stats
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset Info")
    st.sidebar.write(f"Total samples: {len(df):,}")
    st.sidebar.write(f"Personas: {df['persona'].nunique()}")
    st.sidebar.write(f"Categories: {df['category'].nunique()}")

    # Get filtered data
    filtered_df = render_judge_sidebar_filters(df)

    st.sidebar.markdown("---")
    st.sidebar.write(f"Filtered samples: {len(filtered_df):,}")

    # Create tabs for judge-specific visualizations
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Endorsement by Persona",
        "Endorsement by Category",
        "Flagged Analysis",
        "Response Length",
        "Response Explorer",
        "Raw Data",
    ])

    with tab1:
        st.subheader("Endorsement Distribution by Persona")
        if len(filtered_df) > 0:
            st.plotly_chart(
                create_endorsement_chart(filtered_df, group_by="persona", sort_by_accept=True),
                use_container_width=True,
            )

            st.subheader("Flagged Rate by Persona")
            st.plotly_chart(
                create_flagged_rate_chart(filtered_df, group_by="persona"),
                use_container_width=True,
            )

            # Summary stats
            with st.expander("Summary Statistics"):
                # Endorsement counts by persona
                endorsement_counts = filtered_df.groupby(["persona", "endorsement"]).size().unstack(fill_value=0)
                st.markdown("**Endorsement Counts by Persona:**")
                st.dataframe(endorsement_counts)

                # Flagged rate by persona
                flagged_stats = filtered_df.groupby("persona")["flagged"].agg(["sum", "count", "mean"])
                flagged_stats.columns = ["Flagged", "Total", "Rate"]
                flagged_stats["Rate"] = (flagged_stats["Rate"].astype(float) * 100).round(2).astype(str) + "%"
                st.markdown("**Flagged Statistics by Persona:**")
                st.dataframe(flagged_stats)
        else:
            st.info("No data matches the current filters.")

    with tab2:
        st.subheader("Endorsement Distribution by Category")
        if len(filtered_df) > 0:
            st.plotly_chart(
                create_endorsement_chart(filtered_df, group_by="category", sort_by_accept=True),
                use_container_width=True,
            )

            st.subheader("Flagged Rate by Category")
            st.plotly_chart(
                create_flagged_rate_chart(filtered_df, group_by="category"),
                use_container_width=True,
            )

            # Summary stats
            with st.expander("Summary Statistics"):
                # Endorsement counts by category
                endorsement_counts = filtered_df.groupby(["category", "endorsement"]).size().unstack(fill_value=0)
                st.markdown("**Endorsement Counts by Category:**")
                st.dataframe(endorsement_counts)

                # Flagged rate by category
                flagged_stats = filtered_df.groupby("category")["flagged"].agg(["sum", "count", "mean"])
                flagged_stats.columns = ["Flagged", "Total", "Rate"]
                flagged_stats["Rate"] = (flagged_stats["Rate"].astype(float) * 100).round(2).astype(str) + "%"
                st.markdown("**Flagged Statistics by Category:**")
                st.dataframe(flagged_stats)
        else:
            st.info("No data matches the current filters.")

    with tab3:
        st.subheader("Endorsement Distribution for Flagged Responses")

        flagged_count = filtered_df["flagged"].sum()
        total_count = len(filtered_df)
        flagged_pct = (flagged_count / total_count * 100) if total_count > 0 else 0

        st.metric(
            "Flagged Responses",
            f"{flagged_count:,}",
            f"{flagged_pct:.1f}% of filtered data",
        )

        if flagged_count > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**By Persona:**")
                st.plotly_chart(
                    create_endorsement_when_flagged_chart(filtered_df, group_by="persona"),
                    use_container_width=True,
                )
            with col2:
                st.markdown("**By Category:**")
                st.plotly_chart(
                    create_endorsement_when_flagged_chart(filtered_df, group_by="category"),
                    use_container_width=True,
                )

            # Summary of flagged responses
            with st.expander("Flagged Response Summary"):
                flagged_df = filtered_df[filtered_df["flagged"] == True]  # noqa: E712
                st.markdown("**Endorsement Distribution for Flagged Responses:**")
                endorsement_dist = flagged_df["endorsement"].value_counts()
                st.dataframe(endorsement_dist)
        else:
            st.info("No flagged responses in the current selection.")

    with tab4:
        st.subheader("Response Length by Persona")
        if len(filtered_df) > 0:
            st.plotly_chart(
                create_response_length_chart(filtered_df, group_by="persona", sort_by_accept=True),
                use_container_width=True,
            )

            st.subheader("Response Length by Category")
            st.plotly_chart(
                create_response_length_chart(filtered_df, group_by="category"),
                use_container_width=True,
            )

            # Summary stats
            with st.expander("Summary Statistics"):
                length_df = compute_response_length_stats(filtered_df)

                st.markdown("**By Persona:**")
                persona_stats = length_df.groupby("persona")[["thinking_chars", "output_chars"]].agg(
                    ["mean", "std", "median"]
                ).round(1)
                st.dataframe(persona_stats)

                st.markdown("**By Category:**")
                category_stats = length_df.groupby("category")[["thinking_chars", "output_chars"]].agg(
                    ["mean", "std", "median"]
                ).round(1)
                st.dataframe(category_stats)
        else:
            st.info("No data matches the current filters.")

    with tab5:
        render_response_explorer(filtered_df)

    with tab6:
        st.subheader("Raw Data")
        # Column selector with sensible defaults for judge data
        default_cols = ["persona", "category", "endorsement", "flagged", "prompt", "prefill", "response"]
        available_cols = [c for c in default_cols if c in filtered_df.columns]

        display_cols = st.multiselect(
            "Select columns to display",
            options=filtered_df.columns.tolist(),
            default=available_cols,
        )
        if display_cols:
            st.dataframe(filtered_df[display_cols], use_container_width=True)


def render_metrics_view(df: pd.DataFrame) -> None:
    """Render the metrics results visualization view (original functionality).

    Args:
        df: DataFrame with metrics results
    """
    # Display basic stats
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset Info")
    st.sidebar.write(f"Total samples: {len(df):,}")
    st.sidebar.write(f"Categories: {df['category'].nunique()}")
    st.sidebar.write(f"Personas: {df['persona'].nunique()}")
    st.sidebar.write(f"Prompts: {df['prompt_id'].nunique()}")

    # Category filter
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    categories = sorted(df["category"].unique())
    selected_categories = st.sidebar.multiselect(
        "Filter by category",
        options=categories,
        default=categories,
    )

    if selected_categories:
        filtered_df = df[df["category"].isin(selected_categories)]
    else:
        filtered_df = df

    # View mode toggle
    st.sidebar.markdown("---")
    st.sidebar.subheader("View Mode")
    view_mode = st.sidebar.radio(
        "Select view mode",
        options=["All Categories", "Drill Down into Category"],
        index=0,
    )

    # Determine group_by and chart_df based on view mode
    if view_mode == "Drill Down into Category":
        drill_down_category = st.sidebar.selectbox(
            "Select category to drill down",
            options=sorted(filtered_df["category"].unique()),
        )
        chart_df = filtered_df[filtered_df["category"] == drill_down_category]
        group_by = "persona"
    else:
        chart_df = filtered_df
        group_by = "category"

    # Plot type toggle
    st.sidebar.markdown("---")
    st.sidebar.subheader("Chart Options")
    plot_type = st.sidebar.radio(
        "Plot type",
        options=["Box", "Violin"],
        index=0,
        horizontal=True,
    )
    plot_type = plot_type.lower()

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Entropy", "Top-k Mass", "Thinking Tokens", "Correlations", "Raw Data"]
    )

    with tab1:
        st.plotly_chart(create_entropy_chart(chart_df, group_by, plot_type), use_container_width=True)

        # Summary statistics
        with st.expander("Summary Statistics"):
            entropy_stats = (
                chart_df.groupby(group_by)[
                    ["avg_entropy_thinking", "avg_entropy_output", "avg_entropy"]
                ]
                .agg(["mean", "std", "median"])
                .round(4)
            )
            st.dataframe(entropy_stats)

    with tab2:
        st.plotly_chart(create_top_k_mass_chart(chart_df, group_by, plot_type), use_container_width=True)

        # Summary statistics
        with st.expander("Summary Statistics"):
            top_k_stats = (
                chart_df.groupby(group_by)[
                    ["avg_top_k_mass_thinking", "avg_top_k_mass_output", "avg_top_k_mass"]
                ]
                .agg(["mean", "std", "median"])
                .round(4)
            )
            st.dataframe(top_k_stats)

    with tab3:
        st.plotly_chart(
            create_thinking_tokens_chart(chart_df, group_by, plot_type), use_container_width=True
        )

        # Summary statistics
        with st.expander("Summary Statistics"):
            thinking_stats = (
                chart_df.groupby(group_by)["think_end_position"]
                .agg(["mean", "std", "median", "min", "max"])
                .round(2)
            )
            st.dataframe(thinking_stats)

    with tab4:
        # Determine title based on view mode
        if view_mode == "Drill Down into Category":
            corr_title = f"Metric Correlations for {drill_down_category}"
        else:
            corr_title = "Metric Correlations (All Categories)"

        corr_df, p_df = compute_correlations(chart_df)
        st.plotly_chart(
            create_correlation_heatmap(corr_df, corr_title),
            use_container_width=True,
        )

        st.subheader("Correlation Details")

        # Create a detailed correlation table
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Pearson Correlation Coefficients**")
            st.dataframe(corr_df.round(4))

        with col2:
            st.markdown("**P-values**")
            # Use scientific notation for very small p-values
            def format_pvalue(x):
                if x == 0.0:
                    return "< 1e-300"
                elif x < 0.0001:
                    return f"{x:.2e}"
                else:
                    return f"{x:.4f}"
            st.dataframe(p_df.map(format_pvalue))

        # Interpretation guide
        with st.expander("Interpretation Guide"):
            st.markdown("""
**Correlation Coefficient (r):**
- **r = 1.0**: Perfect positive correlation
- **r = 0.0**: No linear correlation
- **r = -1.0**: Perfect negative correlation

**Strength Guidelines:**
- |r| < 0.3: Weak
- 0.3 ≤ |r| < 0.7: Moderate
- |r| ≥ 0.7: Strong

**P-value:**
- p < 0.05: Statistically significant
- p < 0.01: Highly significant
- p < 0.001: Very highly significant

**Note:** Correlations are computed on the currently filtered data.
            """)

        # Sample size info
        st.info(f"Correlations computed on {len(chart_df):,} samples.")

    with tab5:
        st.subheader("Raw Data")
        # Column selector
        display_cols = st.multiselect(
            "Select columns to display",
            options=chart_df.columns.tolist(),
            default=[
                "category",
                "persona",
                "prompt_id",
                "avg_entropy",
                "avg_top_k_mass",
                "think_end_position",
            ],
        )
        if display_cols:
            st.dataframe(chart_df[display_cols], use_container_width=True)


@st.cache_data
def compute_umap_projection(
    dir_path: str,
    model_name: str,
    embedding_col: str,
    n_neighbors: int,
    min_dist: float,
) -> pd.DataFrame:
    """Load raw embeddings and compute 2D UMAP projection.

    Args:
        dir_path: Directory containing embeddings_*.parquet files
        model_name: Which embedding model to load (matches filename suffix)
        embedding_col: Which embedding column to project ('embedding', 'embedding_thinking', 'embedding_output')
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter

    Returns:
        DataFrame with columns: umap_x, umap_y, persona, category, prompt_id, prompt, rep_idx
    """
    import umap

    parquet_path = Path(dir_path) / f"embeddings_{model_name}.parquet"
    assert parquet_path.exists(), f"Raw embeddings not found: {parquet_path}"

    # Load metadata + embedding column
    meta_cols = ["persona", "category", "prompt_id", "rep_idx"]
    schema = pq.read_schema(parquet_path)
    available = set(schema.names)
    meta_cols = [c for c in meta_cols if c in available]
    if "prompt" in available:
        meta_cols.append("prompt")

    raw = pd.read_parquet(parquet_path, columns=meta_cols + [embedding_col])

    # Drop rows where embedding is null
    raw = raw[raw[embedding_col].notna()].reset_index(drop=True)
    if len(raw) == 0:
        return pd.DataFrame()

    # Stack embeddings into numpy array
    emb_matrix = np.stack(raw[embedding_col].values)

    # Compute UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(emb_matrix)

    result = raw[meta_cols].copy()
    result["umap_x"] = coords[:, 0]
    result["umap_y"] = coords[:, 1]
    return result


def render_embeddings_view(df: pd.DataFrame, dir_path: Path | None = None) -> None:
    """Render the embeddings variance visualization view.

    Supports single-model and multi-model data. When multiple models are present,
    additional comparison tabs are shown with model-level analysis.

    Args:
        df: DataFrame loaded from embedding_variance*.parquet (with 'model' column)
        dir_path: Directory containing the parquet files (used for UMAP tab)
    """
    # Find embedding columns and models
    embedding_columns = sorted(df["embedding_column"].unique())
    if not embedding_columns:
        st.error("No embedding columns found in the data.")
        return

    all_models = sorted(df["model"].unique())
    multi_model = len(all_models) > 1

    # Sidebar controls
    with st.sidebar:
        st.markdown("---")
        st.subheader("Configuration")

        # Model filter (only shown when multiple models)
        if multi_model:
            selected_models = st.multiselect(
                "Models",
                options=all_models,
                default=all_models,
                key="emb_model_filter",
            )
        else:
            selected_models = all_models

        # Embedding column selector
        selected_embedding = st.selectbox(
            "Embedding Column",
            options=embedding_columns,
            index=0,
        )

        # Filters
        st.markdown("---")
        st.subheader("Filters")

        # Category filter (if available)
        if "category" in df.columns:
            all_categories = sorted(df["category"].unique())
            selected_categories = st.multiselect(
                "Filter Categories",
                options=all_categories,
                default=all_categories,
            )
        else:
            selected_categories = None

        # Persona filter
        all_personas = sorted(df["persona"].unique())
        selected_personas = st.multiselect(
            "Filter Personas",
            options=all_personas,
            default=all_personas,
        )

        # Dataset info
        st.markdown("---")
        st.subheader("Dataset Info")
        st.write(f"Total records: {len(df):,}")
        st.write(f"Models: {len(all_models)}")
        st.write(f"Unique prompts: {df['canonical_prompt_id'].nunique()}")
        st.write(f"Unique personas: {df['persona'].nunique()}")
        if "category" in df.columns:
            st.write(f"Unique categories: {df['category'].nunique()}")
        st.write(f"Embedding columns: {len(embedding_columns)}")
        if "n_samples" in df.columns:
            st.write(f"Samples per group: {df['n_samples'].iloc[0]}")

    # Use canonical_prompt_id as the grouping key for all downstream logic.
    # Replace prompt_id with canonical_prompt_id so chart functions (which
    # filter by prompt_id) group same-text prompts together automatically.
    df = df.copy()
    df["prompt_id"] = df["canonical_prompt_id"]

    # Filter by selected models, embedding, categories, and personas
    filtered_df = df[df["model"].isin(selected_models)]
    filtered_df = filtered_df[filtered_df["embedding_column"] == selected_embedding]
    if selected_categories:
        filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]
    if selected_personas:
        filtered_df = filtered_df[filtered_df["persona"].isin(selected_personas)]

    if filtered_df.empty:
        st.warning("No data to display with current filters.")
        return

    # Build prompt list from canonical_prompt_id + prompt_text
    if "prompt_text" in df.columns:
        prompts = (
            df[["prompt_id", "prompt_text"]]
            .drop_duplicates()
            .rename(columns={"prompt_text": "question"})
            .sort_values("prompt_id")
        )
    elif "question" in df.columns:
        prompts = (
            df[["prompt_id", "question"]]
            .drop_duplicates()
            .sort_values("prompt_id")
        )
    else:
        prompts = pd.DataFrame({
            "prompt_id": sorted(df["prompt_id"].unique()),
            "question": [f"Prompt {pid}" for pid in sorted(df["prompt_id"].unique())],
        })

    # Check if category column exists and build color map
    has_category = "category" in df.columns
    color_map = build_category_color_map(df) if has_category else None

    # Determine active models after filtering
    active_models = sorted(filtered_df["model"].unique())
    is_multi = len(active_models) > 1

    # Sidebar control: sort personas within charts by a chosen model's variance
    if is_multi:
        with st.sidebar:
            st.markdown("---")
            st.subheader("Sort")
            sort_model = st.selectbox(
                "Sort personas by",
                options=active_models,
                index=0,
                key="emb_sort_model",
            )
    else:
        sort_model = active_models[0] if active_models else None

    # Check if raw embeddings are available for UMAP
    has_umap = False
    raw_embedding_models: list[str] = []
    if dir_path is not None:
        raw_embedding_models = sorted(
            f.stem.split("embeddings_", 1)[1]
            for f in Path(dir_path).glob("embeddings_*.parquet")
            if f.stem.startswith("embeddings_") and "variance" not in f.stem
        )
        has_umap = len(raw_embedding_models) > 0

    # Create tabs — show comparison tabs only when multiple models are active
    tab_names = ["Variance by Persona"]
    if is_multi:
        tab_names += ["Model Comparison", "Model Agreement"]
    tab_names.append("Ranking Changes")
    if has_umap:
        tab_names.append("UMAP")

    tabs = st.tabs(tab_names)
    tab_idx = 0
    tab1 = tabs[tab_idx]; tab_idx += 1
    if is_multi:
        tab2 = tabs[tab_idx]; tab_idx += 1
        tab3 = tabs[tab_idx]; tab_idx += 1
    else:
        tab2 = None
        tab3 = None
    tab4 = tabs[tab_idx]; tab_idx += 1
    tab_umap = tabs[tab_idx] if has_umap else None

    # ---- Tab 1: Variance by Persona ----
    with tab1:
        st.markdown(f"### Variance by Persona (using `{selected_embedding}`)")

        for _, row in prompts.iterrows():
            prompt_id = row["prompt_id"]
            question_text = row["question"]

            if prompt_id not in filtered_df["prompt_id"].values:
                continue

            # Compute persona order for this prompt from the sort model
            sort_sub = filtered_df[
                (filtered_df["prompt_id"] == prompt_id)
                & (filtered_df["model"] == sort_model)
            ]
            if not sort_sub.empty:
                persona_order = (
                    sort_sub.sort_values("total_variance", ascending=False)["persona"]
                    .tolist()
                )
            else:
                persona_order = None

            if is_multi:
                fig = create_multi_model_variance_bar_chart(
                    filtered_df, prompt_id, question_text, color_map, persona_order
                )
            else:
                fig = create_variance_bar_chart(
                    filtered_df, prompt_id, question_text, has_category, color_map,
                    persona_order,
                )
            st.plotly_chart(fig, use_container_width=True)

            prompt_variance = filtered_df[filtered_df["prompt_id"] == prompt_id]
            with st.expander(f"Statistics for Prompt {prompt_id}"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean Variance", f"{prompt_variance['total_variance'].mean():.4f}")
                col2.metric("Std Variance", f"{prompt_variance['total_variance'].std():.4f}")
                col3.metric("Min Variance", f"{prompt_variance['total_variance'].min():.4f}")
                col4.metric("Max Variance", f"{prompt_variance['total_variance'].max():.4f}")

        st.markdown("---")
        with st.expander("View Raw Variance Data"):
            display_df = filtered_df.sort_values(
                ["prompt_id", "total_variance"], ascending=[True, False]
            )
            st.dataframe(display_df, use_container_width=True)

    # ---- Tab 2: Model Comparison (multi-model only) ----
    if tab2 is not None:
        with tab2:
            st.markdown("### Model Comparison")
            st.caption(
                "Mean variance per model across all (prompt, persona) pairs. "
                "Error bars show standard error of the mean."
            )

            # Overall comparison
            fig_overall = create_model_comparison_bar_chart(filtered_df, group_by=None)
            st.plotly_chart(fig_overall, use_container_width=True)

            # By category
            if has_category:
                st.markdown("#### By Category")
                fig_by_cat = create_model_comparison_bar_chart(filtered_df, group_by="category")
                st.plotly_chart(fig_by_cat, use_container_width=True)

            # Variance ratio heatmap
            st.markdown("#### Variance Ratio Heatmap")
            st.caption(
                "Log2 ratio of mean variance between model pairs per persona. "
                "Positive (blue) = numerator model has higher variance."
            )
            fig_ratio = create_model_ratio_heatmap(filtered_df)
            st.plotly_chart(fig_ratio, use_container_width=True)

    # ---- Tab 3: Model Agreement (multi-model only) ----
    if tab3 is not None:
        with tab3:
            st.markdown("### Model Agreement")

            # Rank correlation matrix
            st.markdown("#### Rank Correlation of Persona Variance")
            st.caption(
                "Spearman correlation of persona variance rankings between models. "
                "High values (close to 1) mean models agree on which personas are most/least variable."
            )
            rank_corr_df = create_model_rank_correlation(filtered_df)
            fig_corr = create_correlation_heatmap(rank_corr_df, "Spearman Rank Correlation (Persona Variance)")
            st.plotly_chart(fig_corr, use_container_width=True)

            # Scatter plot: model A vs model B
            st.markdown("#### Model-vs-Model Scatter")
            st.caption(
                "Each point is a (prompt, persona) pair. Points above the diagonal "
                "have higher variance in Model B; below = higher in Model A."
            )
            col_a, col_b = st.columns(2)
            with col_a:
                scatter_model_a = st.selectbox(
                    "Model A (x-axis)",
                    options=active_models,
                    index=0,
                    key="scatter_model_a",
                )
            with col_b:
                default_b = 1 if len(active_models) > 1 else 0
                scatter_model_b = st.selectbox(
                    "Model B (y-axis)",
                    options=active_models,
                    index=default_b,
                    key="scatter_model_b",
                )

            if scatter_model_a == scatter_model_b:
                st.info("Select two different models to compare.")
            else:
                fig_scatter = create_model_scatter(
                    filtered_df, scatter_model_a, scatter_model_b, color_map
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

    # ---- Tab 4: Ranking Changes ----
    with tab4:
        if not has_category:
            st.warning("Category data not available. This tab requires a 'category' column.")
        else:
            st.markdown("### Category Ranking Changes Across Embedding Columns")
            st.markdown(
                "This view shows how each category's average variance rank changes across "
                "different embedding columns. Rank 1 (top) = lowest average variance."
            )

            # When multiple models, let user pick which model's ranking to display
            if is_multi:
                ranking_model = st.radio(
                    "Model for ranking",
                    options=active_models,
                    horizontal=True,
                    key="ranking_model_radio",
                )
                ranking_base_df = df[df["model"] == ranking_model]
            else:
                ranking_base_df = df.copy()

            # Apply category filter
            if selected_categories:
                ranking_filtered_df = ranking_base_df[
                    ranking_base_df["category"].isin(selected_categories)
                ]
            else:
                ranking_filtered_df = ranking_base_df

            for _, row in prompts.iterrows():
                prompt_id = row["prompt_id"]
                question_text = row["question"]

                if prompt_id not in ranking_filtered_df["prompt_id"].values:
                    continue

                fig = create_ranking_line_chart(
                    ranking_filtered_df, prompt_id, question_text, color_map
                )
                st.plotly_chart(fig, use_container_width=True)

                ranking_df = calculate_category_rankings(ranking_filtered_df, prompt_id)
                with st.expander(f"Ranking Statistics for Prompt {prompt_id}"):
                    rank_stats = ranking_df.groupby("category")["rank"].agg(["mean", "std", "min", "max"])
                    rank_stats.columns = ["Mean Rank", "Rank Std", "Best Rank", "Worst Rank"]
                    rank_stats = rank_stats.sort_values("Mean Rank")
                    st.dataframe(rank_stats, use_container_width=True)

    # ---- UMAP Tab ----
    if tab_umap is not None:
        with tab_umap:
            st.markdown("### UMAP Projection of Raw Embeddings")
            st.caption(
                "2D UMAP projection of raw embedding vectors. Each point is one "
                "(persona, prompt, rep) sample. Colored by persona or category."
            )

            # UMAP controls
            col_m, col_e, col_c = st.columns(3)
            with col_m:
                umap_model = st.selectbox(
                    "Embedding model",
                    options=raw_embedding_models,
                    index=0,
                    key="umap_model",
                )
            with col_e:
                umap_emb_col = st.selectbox(
                    "Embedding column",
                    options=["embedding", "embedding_thinking", "embedding_output"],
                    index=0,
                    key="umap_emb_col",
                )
            with col_c:
                umap_color = st.selectbox(
                    "Color by",
                    options=["persona", "category"],
                    index=0,
                    key="umap_color",
                )

            col_n, col_d = st.columns(2)
            with col_n:
                umap_n_neighbors = st.slider(
                    "n_neighbors",
                    min_value=5, max_value=200, value=15, step=5,
                    key="umap_n_neighbors",
                )
            with col_d:
                umap_min_dist = st.slider(
                    "min_dist",
                    min_value=0.0, max_value=1.0, value=0.1, step=0.05,
                    key="umap_min_dist",
                )

            with st.spinner("Computing UMAP projection..."):
                umap_df = compute_umap_projection(
                    str(dir_path),
                    umap_model,
                    umap_emb_col,
                    umap_n_neighbors,
                    umap_min_dist,
                )

            if umap_df.empty:
                st.warning(
                    f"No data found for embedding column '{umap_emb_col}'. "
                    "This column may be all null (e.g. thinking embeddings when thinking mode is off)."
                )
            else:
                # Apply category/persona filters from sidebar
                if selected_categories and "category" in umap_df.columns:
                    umap_df = umap_df[umap_df["category"].isin(selected_categories)]
                if selected_personas and "persona" in umap_df.columns:
                    umap_df = umap_df[umap_df["persona"].isin(selected_personas)]

                fig = px.scatter(
                    umap_df,
                    x="umap_x",
                    y="umap_y",
                    color=umap_color,
                    color_discrete_map=color_map if umap_color == "category" else None,
                    hover_data=[c for c in ["persona", "category", "prompt_id"] if c in umap_df.columns],
                    title=f"UMAP — {umap_model} / {umap_emb_col}",
                    labels={"umap_x": "UMAP 1", "umap_y": "UMAP 2"},
                    opacity=0.6,
                )
                fig.update_traces(marker=dict(size=4))
                fig.update_layout(
                    height=800,
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1.0,
                        xanchor="left",
                        x=1.02,
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.info(f"Showing {len(umap_df):,} points.")


@st.cache_data
def load_tc_llm_data(groups_path: str) -> pd.DataFrame:
    """Load TC-LLM groups and records, compute per-(prompt_id, persona) label entropy.

    Args:
        groups_path: Path to tc_llm_groups.jsonl

    Returns:
        DataFrame with columns: prompt_id, persona, category, entropy, n_labels, n_reps
    """
    groups_path = Path(groups_path)
    dir_path = groups_path.parent
    records_path = dir_path / "tc_llm_records.jsonl"

    assert records_path.exists(), f"Missing {records_path}"

    # Load groups
    groups = []
    with open(groups_path) as f:
        for line in f:
            if line.strip():
                groups.append(json.loads(line))

    # Load records
    records = []
    with open(records_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Build persona -> category map
    # Strategy 1: config.json with personae_json path (batch inference dirs)
    # Strategy 2: results.jsonl with per-record category (pipeline dirs)
    persona_to_category: dict[str, str] = {}
    config_path = dir_path / "config.json"
    results_path = dir_path / "results.jsonl"

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        if "personae_json" in config:
            personae_json_path = Path(config["personae_json"])
            if not personae_json_path.is_absolute():
                personae_json_path = dir_path.parent.parent / personae_json_path
            with open(personae_json_path) as f:
                personae = json.load(f)
            persona_to_category = {p["persona"]: p["category"] for p in personae}

    # Extract prompt_id -> prompt text and (fallback) persona -> category from results.jsonl
    prompt_id_to_text: dict[int, str] = {}
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    pid = row.get("prompt_id")
                    if pid is not None and pid not in prompt_id_to_text:
                        prompt_id_to_text[pid] = row.get("prompt", "")
                    if not persona_to_category:
                        p, c = row.get("persona"), row.get("category")
                        if p and c and p not in persona_to_category:
                            persona_to_category[p] = c

    # Index groups by (prompt_id, persona) for fast lookup
    group_index: dict[tuple[int, str], list[str]] = {}
    for g in groups:
        group_index[(g["prompt_id"], g["persona"])] = g["merged_labels"]

    # Collect all label assignments per (prompt_id, persona)
    label_counts: dict[tuple[int, str], dict[str, int]] = {}
    rep_counts: dict[tuple[int, str], int] = {}
    for r in records:
        key = (r["prompt_id"], r["persona"])
        if key not in label_counts:
            label_counts[key] = {}
            rep_counts[key] = 0
        rep_counts[key] += 1
        for label in r["labels"]:
            label_counts[key][label] = label_counts[key].get(label, 0) + 1

    # Compute entropy per (prompt_id, persona)
    rows = []
    for key, merged_labels in group_index.items():
        prompt_id, persona = key
        counts = label_counts.get(key, {})
        total = sum(counts.values())
        if total == 0:
            entropy = 0.0
        else:
            entropy = 0.0
            for label in merged_labels:
                c = counts.get(label, 0)
                if c > 0:
                    p = c / total
                    entropy -= p * np.log2(p)

        rows.append({
            "prompt_id": prompt_id,
            "persona": persona,
            "category": persona_to_category.get(persona, "unknown"),
            "entropy": entropy,
            "n_labels": len(merged_labels),
            "n_reps": rep_counts.get(key, 0),
            "prompt": prompt_id_to_text.get(prompt_id, ""),
        })

    return pd.DataFrame(rows)


def create_tc_llm_entropy_chart(
    df: pd.DataFrame, group_by: str = "persona"
) -> go.Figure:
    """Create bar chart of mean label entropy with SEM error bars and SD range lines.

    Args:
        df: DataFrame from load_tc_llm_data
        group_by: "persona" or "category"

    Returns:
        Plotly figure
    """
    assert group_by in ("persona", "category"), f"Invalid group_by: {group_by}"

    agg = df.groupby(group_by)["entropy"].agg(["mean", "std", "count"]).reset_index()
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])
    agg = agg.sort_values("mean", ascending=False)

    x_label = "Persona" if group_by == "persona" else "Category"

    fig = go.Figure()

    categories = agg[group_by].tolist()

    # Bars with SEM error bars (red)
    fig.add_trace(go.Bar(
        x=categories,
        y=agg["mean"].tolist(),
        error_y=dict(
            type="data", array=agg["sem"].tolist(), visible=True,
            color="#e74c3c", thickness=2, width=4,
        ),
        marker_color="#3498db",
        showlegend=False,
    ))
    # SEM legend entry as red line
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="lines",
        line=dict(color="#e74c3c", width=2),
        name="SEM",
    ))

    # SD range as gray dashed shapes (centered on bar)
    sd_upper = (agg["mean"] + agg["std"]).tolist()
    sd_lower = (agg["mean"] - agg["std"]).tolist()
    sd_color = "#999999"

    for i, cat in enumerate(categories):
        # Vertical dashed line (drawn above bars so color isn't tinted)
        fig.add_shape(
            type="line",
            x0=cat, x1=cat,
            y0=sd_lower[i], y1=sd_upper[i],
            line=dict(color=sd_color, width=1.5, dash="dash"),
            layer="above",
        )

    # Horizontal caps at +SD and -SD using scatter markers
    fig.add_trace(go.Scatter(
        x=categories + categories,
        y=sd_upper + sd_lower,
        mode="markers",
        marker=dict(symbol="line-ew-open", size=14, color=sd_color, line=dict(width=2)),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Invisible trace for SD legend entry
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="lines",
        line=dict(color=sd_color, width=1.5, dash="dash"),
        name="SD range",
    ))

    fig.update_layout(
        title=f"TC-LLM Label Entropy by {x_label}",
        xaxis_title=x_label,
        yaxis_title="Shannon Entropy (bits)",
        xaxis_tickangle=-45,
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_tc_llm_view(df: pd.DataFrame) -> None:
    """Render the TC-LLM label entropy visualization view.

    Args:
        df: DataFrame from load_tc_llm_data
    """
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset Info")
    st.sidebar.write(f"Total groups: {len(df):,}")
    st.sidebar.write(f"Personas: {df['persona'].nunique()}")
    st.sidebar.write(f"Categories: {df['category'].nunique()}")
    st.sidebar.write(f"Prompts: {df['prompt_id'].nunique()}")
    if "n_reps" in df.columns and len(df) > 0:
        st.sidebar.write(f"Reps per group: {df['n_reps'].iloc[0]}")

    # Filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")

    categories = sorted(df["category"].unique())
    selected_categories = st.sidebar.multiselect(
        "Filter by category",
        options=categories,
        default=categories,
        key="tc_llm_category_filter",
    )

    personas = sorted(df["persona"].unique())
    selected_personas = st.sidebar.multiselect(
        "Filter by persona",
        options=personas,
        default=personas,
        key="tc_llm_persona_filter",
    )

    filtered_df = df.copy()
    if selected_categories:
        filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]
    if selected_personas:
        filtered_df = filtered_df[filtered_df["persona"].isin(selected_personas)]

    st.sidebar.markdown("---")
    st.sidebar.write(f"Filtered groups: {len(filtered_df):,}")

    # Overview explainer
    st.markdown(
        "**TC-LLM Label Entropy** measures how diverse a persona's responses are "
        "to a given prompt. For each (persona, prompt) pair, the model generates many "
        "responses (reps). TC-LLM clustering assigns ranked topic labels to each response. "
        "We then count how often each label appears across all reps and compute "
        "**Shannon entropy** over this distribution.\n\n"
        "Shannon entropy (H) is defined as `H = -sum(p * log2(p))` where p is the "
        "proportion of times each label was assigned. Concretely:\n"
        "- If the model always produces the same topic label, p=1 for one label and 0 "
        "for the rest, so **H = 0 bits** (no diversity at all).\n"
        "- If every label is equally likely across ~20 labels, "
        "**H = log2(20) = 4.32 bits** (maximum diversity).\n"
        "- In practice, values between these extremes indicate partial concentration: "
        "**higher H = more spread across topics, lower H = more repetitive/concentrated**."
    )

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "Entropy by Persona",
        "Entropy by Category",
        "Raw Data",
    ])

    with tab1:
        if len(filtered_df) > 0:
            st.caption(
                "Mean entropy per persona (averaged across all prompts). "
                "Solid red error bars show **standard error of the mean (SEM)** -- "
                "the uncertainty in the mean estimate (use these when comparing persona means). "
                "Dashed gray lines show **standard deviation (SD)** -- the spread of individual "
                "prompt entropies within each persona. "
                "The Kruskal-Wallis test checks whether the entropy distributions "
                "differ significantly across personas overall (p < 0.05 = significant difference)."
            )
            personas_list = sorted(filtered_df["persona"].unique())
            persona_groups = [
                g["entropy"].values
                for _, g in filtered_df.groupby("persona")
            ]

            # Kruskal-Wallis omnibus test
            fig = create_tc_llm_entropy_chart(filtered_df, group_by="persona")
            if len(persona_groups) >= 2:
                kw_stat, kw_p = stats.kruskal(*persona_groups)
                fig.add_annotation(
                    text=f"Kruskal-Wallis: H={kw_stat:.2f}, p={kw_p:.2e}",
                    xref="paper", yref="paper",
                    x=0.0, y=1.0,
                    xanchor="left", yanchor="bottom",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#ccc",
                    borderwidth=1,
                    borderpad=4,
                )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Summary Statistics"):
                persona_stats = (
                    filtered_df.groupby("persona")["entropy"]
                    .agg(["mean", "std", "median", "min", "max"])
                    .round(4)
                    .sort_values("mean", ascending=False)
                )
                st.dataframe(persona_stats, use_container_width=True)

            # Pairwise Mann-Whitney U tests
            if len(personas_list) >= 2:
                with st.expander("Pairwise P-values & Effect Sizes (Mann-Whitney U)"):
                    st.markdown(
                        "Each cell shows the **p-value** and **rank-biserial correlation (r)** "
                        "from a Mann-Whitney U test comparing the entropy distributions of "
                        "two personas across all prompts.\n\n"
                        "- **P-value:** A small p-value (< 0.05) means the difference is "
                        "*statistically significant* -- unlikely to arise by chance. However, "
                        "with large samples (~245 prompts per persona), even tiny differences "
                        "can be significant.\n"
                        "- **Effect size (r):** Rank-biserial correlation measures *how much* "
                        "the distributions actually differ, regardless of sample size. "
                        "This is the more meaningful number for interpretation:\n"
                        "  - |r| < 0.1: negligible difference\n"
                        "  - 0.1 - 0.3: small difference\n"
                        "  - 0.3 - 0.5: medium difference\n"
                        "  - |r| > 0.5: large difference\n\n"
                        "**Bottom line:** If p is small but |r| is negligible, the personas "
                        "are *technically* different but *practically* similar."
                    )
                    persona_entropy = {
                        p: filtered_df[filtered_df["persona"] == p]["entropy"].values
                        for p in personas_list
                    }
                    n = len(personas_list)
                    p_matrix = np.full((n, n), np.nan)
                    r_matrix = np.full((n, n), np.nan)
                    for i in range(n):
                        for j in range(i + 1, n):
                            u_stat, p_val = stats.mannwhitneyu(
                                persona_entropy[personas_list[i]],
                                persona_entropy[personas_list[j]],
                                alternative="two-sided",
                            )
                            p_matrix[i, j] = p_val
                            p_matrix[j, i] = p_val
                            # Rank-biserial correlation: r = 1 - 2U/(n1*n2)
                            n1 = len(persona_entropy[personas_list[i]])
                            n2 = len(persona_entropy[personas_list[j]])
                            r_val = 1 - (2 * u_stat) / (n1 * n2)
                            r_matrix[i, j] = r_val
                            r_matrix[j, i] = -r_val  # sign flips for reversed comparison

                    p_df = pd.DataFrame(
                        p_matrix, index=personas_list, columns=personas_list
                    )
                    r_df = pd.DataFrame(
                        r_matrix, index=personas_list, columns=personas_list
                    )

                    col_p, col_r = st.columns(2)

                    with col_p:
                        # P-value heatmap
                        fig_pw = go.Figure(data=go.Heatmap(
                            z=p_df.values,
                            x=p_df.columns.tolist(),
                            y=p_df.index.tolist(),
                            text=p_df.map(
                                lambda x: f"{x:.2e}" if pd.notna(x) else ""
                            ).values,
                            texttemplate="%{text}",
                            textfont={"size": 11},
                            colorscale=[[0, "#2ecc71"], [0.05, "#f1c40f"], [1, "#e74c3c"]],
                            zmin=0, zmax=1,
                            colorbar=dict(title="p-value"),
                        ))
                        fig_pw.update_layout(
                            title="P-values",
                            height=400 + 30 * n,
                        )
                        st.plotly_chart(fig_pw, use_container_width=True)
                        st.caption(
                            "Green = significant (p < 0.05), yellow = borderline, "
                            "red = not significant."
                        )

                    with col_r:
                        # Effect size heatmap
                        fig_r = go.Figure(data=go.Heatmap(
                            z=r_df.values,
                            x=r_df.columns.tolist(),
                            y=r_df.index.tolist(),
                            text=r_df.map(
                                lambda x: f"{x:.3f}" if pd.notna(x) else ""
                            ).values,
                            texttemplate="%{text}",
                            textfont={"size": 11},
                            colorscale="RdBu_r",
                            zmid=0, zmin=-1, zmax=1,
                            colorbar=dict(title="r"),
                        ))
                        fig_r.update_layout(
                            title="Effect Size (rank-biserial r)",
                            height=400 + 30 * n,
                        )
                        st.plotly_chart(fig_r, use_container_width=True)
                        st.caption(
                            "Positive r (blue) = row persona has higher entropy than column. "
                            "Negative r (red) = lower. Magnitude indicates strength."
                        )

            # Top/bottom 5 entropy prompts per persona
            with st.expander("Highest & Lowest Entropy Prompts per Persona"):
                st.markdown(
                    "For a given persona, **high-entropy prompts** are those where the model "
                    "produces the widest variety of topics across repetitions (less predictable), "
                    "while **low-entropy prompts** elicit highly consistent, repetitive topic labels."
                )
                selected_persona = st.selectbox(
                    "Select persona",
                    options=sorted(filtered_df["persona"].unique()),
                    key="tc_llm_extreme_persona",
                )
                persona_df = filtered_df[filtered_df["persona"] == selected_persona]
                display_cols = ["prompt_id", "entropy", "n_labels"]
                if "prompt" in persona_df.columns:
                    display_cols.append("prompt")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Top 5 (highest entropy)**")
                    top5 = persona_df.nlargest(5, "entropy")[display_cols].reset_index(drop=True)
                    top5["entropy"] = top5["entropy"].round(4)
                    st.dataframe(top5, use_container_width=True)
                with col2:
                    st.markdown("**Bottom 5 (lowest entropy)**")
                    bot5 = persona_df.nsmallest(5, "entropy")[display_cols].reset_index(drop=True)
                    bot5["entropy"] = bot5["entropy"].round(4)
                    st.dataframe(bot5, use_container_width=True)

            # Most/least variable prompts across personas
            st.markdown("---")
            st.subheader("Prompts with Most/Least Entropy Variation Across Personas")
            st.caption(
                "For each prompt, we compute the standard deviation of entropy across all personas. "
                "**Most variable prompts** are those where persona choice dramatically changes "
                "topical diversity (some personas explore many topics, others fixate on a few). "
                "**Least variable prompts** elicit similar topical diversity regardless of persona, "
                "suggesting the prompt itself drives the behavior more than the persona does."
            )

            # Compute per-prompt std of entropy across personas
            prompt_var = (
                filtered_df.groupby("prompt_id")["entropy"]
                .std()
                .rename("entropy_std")
                .reset_index()
            )

            most_variable = prompt_var.nlargest(5, "entropy_std")["prompt_id"].tolist()
            least_variable = prompt_var.nsmallest(5, "entropy_std")["prompt_id"].tolist()

            for label, prompt_ids in [
                ("Top 5 Most Variable", most_variable),
                ("Top 5 Least Variable", least_variable),
            ]:
                subset = filtered_df[filtered_df["prompt_id"].isin(prompt_ids)].copy()
                # Build x-axis label with line wrapping
                prompt_labels = {}
                for pid in prompt_ids:
                    row = subset[subset["prompt_id"] == pid].iloc[0]
                    text = row.get("prompt", "")
                    # Wrap text at ~40 chars using <br> for plotly
                    words = text.split()
                    lines, current = [], ""
                    for w in words:
                        if current and len(current) + len(w) + 1 > 40:
                            lines.append(current)
                            current = w
                        else:
                            current = f"{current} {w}" if current else w
                    if current:
                        lines.append(current)
                    wrapped = "<br>".join(lines)
                    prompt_labels[pid] = f"P{pid}: {wrapped}"
                subset["prompt_label"] = subset["prompt_id"].map(prompt_labels)

                # Sort by std (descending for most, ascending for least)
                id_order = prompt_ids  # already sorted by nlargest/nsmallest
                label_order = [prompt_labels[pid] for pid in id_order]

                fig_var = px.bar(
                    subset,
                    x="prompt_label",
                    y="entropy",
                    color="persona",
                    barmode="group",
                    title=f"{label} Prompts",
                    labels={
                        "prompt_label": "Prompt",
                        "entropy": "Shannon Entropy (bits)",
                        "persona": "Persona",
                    },
                    category_orders={"prompt_label": label_order},
                )
                fig_var.update_layout(
                    xaxis_tickangle=-30,
                    height=550,
                    margin=dict(b=120),
                    legend=dict(
                        title="Persona",
                        orientation="v",
                        yanchor="top",
                        y=1.0,
                        xanchor="left",
                        x=1.02,
                    ),
                )
                st.plotly_chart(fig_var, use_container_width=True)
        else:
            st.info("No data matches the current filters.")

    with tab2:
        if len(filtered_df) > 0:
            st.caption(
                "Box plot of entropy grouped by persona category. Each point is one "
                "(persona, prompt) pair. The diamond shows the mean. Categories with "
                "higher entropy tend to produce more topically varied responses overall."
            )
            # Box plot to show distribution across personas within each category
            fig = px.box(
                filtered_df,
                x="category",
                y="entropy",
                color="category",
                title="TC-LLM Label Entropy by Category",
                labels={"entropy": "Shannon Entropy (bits)", "category": "Category"},
                points="all",
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                height=600,
                showlegend=False,
            )
            fig.update_traces(boxmean=True)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Summary Statistics"):
                category_stats = (
                    filtered_df.groupby("category")["entropy"]
                    .agg(["mean", "std", "median", "min", "max"])
                    .round(4)
                    .sort_values("mean", ascending=False)
                )
                st.dataframe(category_stats, use_container_width=True)
        else:
            st.info("No data matches the current filters.")

    with tab3:
        st.subheader("Raw Data")
        display_cols = st.multiselect(
            "Select columns to display",
            options=filtered_df.columns.tolist(),
            default=["persona", "category", "prompt_id", "prompt", "entropy", "n_labels", "n_reps"],
            key="tc_llm_raw_cols",
        )
        if display_cols:
            st.dataframe(
                filtered_df[display_cols].sort_values(
                    ["persona", "prompt_id"]
                ),
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# User-Turn Prediction Visualization
# ---------------------------------------------------------------------------


@st.cache_data
def load_user_turn_data(file_path: str) -> pd.DataFrame:
    """Load user-turn prediction results from JSONL into a DataFrame."""
    data = []
    with open(file_path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)


def _resolve_user_turn_metric_columns(df: pd.DataFrame) -> dict:
    """Check whether thinking columns have data and return metric column info.

    Returns:
        dict with keys:
            entropy_cols: list of entropy column names to use
            top_k_cols: list of top-k mass column names to use
            token_cols: list of token count column names to use
            has_thinking: bool
            metric_labels: dict mapping column name -> display label
    """
    # Check if thinking columns have any non-null data
    thinking_cols = [
        "response1_avg_entropy_thinking",
        "response2_avg_entropy_thinking",
    ]
    has_thinking = any(
        col in df.columns and df[col].notna().any() for col in thinking_cols
    )

    if has_thinking:
        entropy_cols = [
            "response1_avg_entropy_thinking",
            "response1_avg_entropy_output",
            "response1_avg_entropy",
            "response2_avg_entropy_thinking",
            "response2_avg_entropy_output",
            "response2_avg_entropy",
        ]
        top_k_cols = [
            "response1_avg_top_k_mass_thinking",
            "response1_avg_top_k_mass_output",
            "response1_avg_top_k_mass",
            "response2_avg_top_k_mass_thinking",
            "response2_avg_top_k_mass_output",
            "response2_avg_top_k_mass",
        ]
        metric_labels = {
            "response1_avg_entropy_thinking": "T1 Entropy (Thinking)",
            "response1_avg_entropy_output": "T1 Entropy (Output)",
            "response1_avg_entropy": "T1 Entropy (Overall)",
            "response2_avg_entropy_thinking": "T2 Entropy (Thinking)",
            "response2_avg_entropy_output": "T2 Entropy (Output)",
            "response2_avg_entropy": "T2 Entropy (Overall)",
            "response1_avg_top_k_mass_thinking": "T1 Top-k Mass (Thinking)",
            "response1_avg_top_k_mass_output": "T1 Top-k Mass (Output)",
            "response1_avg_top_k_mass": "T1 Top-k Mass (Overall)",
            "response2_avg_top_k_mass_thinking": "T2 Top-k Mass (Thinking)",
            "response2_avg_top_k_mass_output": "T2 Top-k Mass (Output)",
            "response2_avg_top_k_mass": "T2 Top-k Mass (Overall)",
            "response1_num_tokens": "T1 Tokens",
            "response2_num_tokens": "T2 Tokens",
        }
    else:
        entropy_cols = ["response1_avg_entropy", "response2_avg_entropy"]
        top_k_cols = ["response1_avg_top_k_mass", "response2_avg_top_k_mass"]
        metric_labels = {
            "response1_avg_entropy": "Turn 1 Entropy",
            "response2_avg_entropy": "Turn 2 Entropy",
            "response1_avg_top_k_mass": "Turn 1 Top-k Mass",
            "response2_avg_top_k_mass": "Turn 2 Top-k Mass",
            "response1_num_tokens": "Turn 1 Tokens",
            "response2_num_tokens": "Turn 2 Tokens",
        }

    token_cols = ["response1_num_tokens", "response2_num_tokens"]

    return {
        "entropy_cols": entropy_cols,
        "top_k_cols": top_k_cols,
        "token_cols": token_cols,
        "has_thinking": has_thinking,
        "metric_labels": metric_labels,
    }


def _create_mean_bar_chart(
    melted: pd.DataFrame,
    group_by: str,
    y_col: str,
    title: str,
    y_label: str,
    x_label: str,
    metric_order: list[str],
) -> go.Figure:
    """Create grouped bar chart of means with SEM error bars from melted data."""
    agg = (
        melted.groupby([group_by, "metric"])[y_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])

    fig = px.bar(
        agg,
        x=group_by,
        y="mean",
        color="metric",
        error_y="sem",
        barmode="group",
        title=title,
        labels={"mean": y_label, group_by: x_label, "metric": "Turn / Section"},
        category_orders={"metric": metric_order},
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=900,
    )
    return fig


def create_user_turn_entropy_chart(
    df: pd.DataFrame,
    group_by: str,
    plot_type: str,
    metric_info: dict,
) -> go.Figure:
    """Create box/violin/mean chart comparing Turn 1 vs Turn 2 entropy."""
    assert group_by in ("category", "persona")
    assert plot_type in ("box", "violin", "mean")

    entropy_cols = metric_info["entropy_cols"]
    labels = metric_info["metric_labels"]

    id_vars = ["category", "persona", "prompt_id"]
    melted = df.melt(
        id_vars=id_vars,
        value_vars=entropy_cols,
        var_name="metric",
        value_name="entropy",
    )
    melted["metric"] = melted["metric"].map(labels)

    x_label = "Persona Category" if group_by == "category" else "Persona"
    metric_order = [labels[c] for c in entropy_cols]

    if plot_type == "mean":
        return _create_mean_bar_chart(
            melted, group_by, "entropy",
            title=f"Mean Entropy by {x_label} (Turn 1 vs Turn 2)",
            y_label="Entropy", x_label=x_label, metric_order=metric_order,
        )

    plot_fn = px.violin if plot_type == "violin" else px.box
    fig = plot_fn(
        melted,
        x=group_by,
        y="entropy",
        color="metric",
        title=f"Entropy by {x_label} (Turn 1 vs Turn 2)",
        labels={"entropy": "Entropy", group_by: x_label, "metric": "Turn / Section"},
        category_orders={"metric": metric_order},
    )
    mode_key = "violinmode" if plot_type == "violin" else "boxmode"
    fig.update_layout(
        **{mode_key: "group"},
        xaxis_tickangle=-45,
        height=900,
    )
    if plot_type == "violin":
        fig.update_traces(meanline_visible=True)
    else:
        fig.update_traces(boxmean=True)
    return fig


def create_user_turn_top_k_mass_chart(
    df: pd.DataFrame,
    group_by: str,
    plot_type: str,
    metric_info: dict,
) -> go.Figure:
    """Create box/violin/mean chart comparing Turn 1 vs Turn 2 top-k mass."""
    assert group_by in ("category", "persona")
    assert plot_type in ("box", "violin", "mean")

    top_k_cols = metric_info["top_k_cols"]
    labels = metric_info["metric_labels"]

    id_vars = ["category", "persona", "prompt_id"]
    melted = df.melt(
        id_vars=id_vars,
        value_vars=top_k_cols,
        var_name="metric",
        value_name="top_k_mass",
    )
    melted["metric"] = melted["metric"].map(labels)

    x_label = "Persona Category" if group_by == "category" else "Persona"
    metric_order = [labels[c] for c in top_k_cols]

    if plot_type == "mean":
        return _create_mean_bar_chart(
            melted, group_by, "top_k_mass",
            title=f"Mean Top-k Mass by {x_label} (Turn 1 vs Turn 2)",
            y_label="Top-k Mass", x_label=x_label, metric_order=metric_order,
        )

    plot_fn = px.violin if plot_type == "violin" else px.box
    fig = plot_fn(
        melted,
        x=group_by,
        y="top_k_mass",
        color="metric",
        title=f"Top-k Mass by {x_label} (Turn 1 vs Turn 2)",
        labels={"top_k_mass": "Top-k Mass", group_by: x_label, "metric": "Turn / Section"},
        category_orders={"metric": metric_order},
    )
    mode_key = "violinmode" if plot_type == "violin" else "boxmode"
    fig.update_layout(
        **{mode_key: "group"},
        xaxis_tickangle=-45,
        height=900,
    )
    if plot_type == "violin":
        fig.update_traces(meanline_visible=True)
    else:
        fig.update_traces(boxmean=True)
    return fig


def create_user_turn_token_count_chart(
    df: pd.DataFrame,
    group_by: str,
    plot_type: str,
    metric_info: dict,
) -> go.Figure:
    """Create box/violin/mean chart comparing Turn 1 vs Turn 2 token counts."""
    assert group_by in ("category", "persona")
    assert plot_type in ("box", "violin", "mean")

    token_cols = metric_info["token_cols"]
    labels = metric_info["metric_labels"]

    id_vars = ["category", "persona", "prompt_id"]
    melted = df.melt(
        id_vars=id_vars,
        value_vars=token_cols,
        var_name="metric",
        value_name="num_tokens",
    )
    melted["metric"] = melted["metric"].map(labels)

    x_label = "Persona Category" if group_by == "category" else "Persona"
    metric_order = [labels[c] for c in token_cols]

    if plot_type == "mean":
        return _create_mean_bar_chart(
            melted, group_by, "num_tokens",
            title=f"Mean Token Count by {x_label} (Turn 1 vs Turn 2)",
            y_label="Number of Tokens", x_label=x_label, metric_order=metric_order,
        )

    plot_fn = px.violin if plot_type == "violin" else px.box
    fig = plot_fn(
        melted,
        x=group_by,
        y="num_tokens",
        color="metric",
        title=f"Token Count by {x_label} (Turn 1 vs Turn 2)",
        labels={"num_tokens": "Number of Tokens", group_by: x_label, "metric": "Turn"},
        category_orders={"metric": metric_order},
    )
    mode_key = "violinmode" if plot_type == "violin" else "boxmode"
    fig.update_layout(
        **{mode_key: "group"},
        xaxis_tickangle=-45,
        height=900,
    )
    if plot_type == "violin":
        fig.update_traces(meanline_visible=True)
    else:
        fig.update_traces(boxmean=True)
    return fig


def compute_user_turn_correlations(
    df: pd.DataFrame, metric_info: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Pearson correlations across all numeric user-turn metrics.

    Returns:
        Tuple of (correlation_matrix, p_value_matrix)
    """
    all_cols = (
        metric_info["entropy_cols"]
        + metric_info["top_k_cols"]
        + metric_info["token_cols"]
    )
    labels = metric_info["metric_labels"]

    n = len(all_cols)
    corr_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))

    for i, m1 in enumerate(all_cols):
        for j, m2 in enumerate(all_cols):
            if i == j:
                corr_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
            else:
                valid_mask = df[m1].notna() & df[m2].notna()
                if valid_mask.sum() > 2:
                    r, p = stats.pearsonr(
                        df.loc[valid_mask, m1], df.loc[valid_mask, m2]
                    )
                    corr_matrix[i, j] = r
                    p_matrix[i, j] = p
                else:
                    corr_matrix[i, j] = np.nan
                    p_matrix[i, j] = np.nan

    display_labels = [labels[c] for c in all_cols]
    corr_df = pd.DataFrame(corr_matrix, index=display_labels, columns=display_labels)
    p_df = pd.DataFrame(p_matrix, index=display_labels, columns=display_labels)
    return corr_df, p_df


def compute_user_turn_statistical_tests(
    df: pd.DataFrame, group_by: str, metric_col: str
) -> tuple[float, float, pd.DataFrame, pd.DataFrame]:
    """Kruskal-Wallis omnibus + pairwise Mann-Whitney U for a user-turn metric.

    Args:
        df: Filtered DataFrame
        group_by: "category" or "persona"
        metric_col: Column name to test

    Returns:
        (kw_stat, kw_p, p_df, r_df)
    """
    groups_list = sorted(df[group_by].unique())
    group_data = {
        g: df.loc[df[group_by] == g, metric_col].dropna().values
        for g in groups_list
    }
    # Filter out empty groups
    groups_list = [g for g in groups_list if len(group_data[g]) > 0]
    group_data = {g: group_data[g] for g in groups_list}

    # Kruskal-Wallis
    if len(groups_list) >= 2:
        kw_stat, kw_p = stats.kruskal(*[group_data[g] for g in groups_list])
    else:
        kw_stat, kw_p = np.nan, np.nan

    # Pairwise Mann-Whitney U
    n = len(groups_list)
    p_matrix = np.full((n, n), np.nan)
    r_matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            u_stat, p_val = stats.mannwhitneyu(
                group_data[groups_list[i]],
                group_data[groups_list[j]],
                alternative="two-sided",
            )
            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val
            n1 = len(group_data[groups_list[i]])
            n2 = len(group_data[groups_list[j]])
            r_val = 1 - (2 * u_stat) / (n1 * n2)
            r_matrix[i, j] = r_val
            r_matrix[j, i] = -r_val

    p_df = pd.DataFrame(p_matrix, index=groups_list, columns=groups_list)
    r_df = pd.DataFrame(r_matrix, index=groups_list, columns=groups_list)
    return kw_stat, kw_p, p_df, r_df


def render_user_turn_stat_test_heatmaps(
    p_df: pd.DataFrame,
    r_df: pd.DataFrame,
    kw_stat: float,
    kw_p: float,
    metric_label: str,
) -> None:
    """Render KW result line + side-by-side p-value and effect-size heatmaps."""
    n = len(p_df)

    if not np.isnan(kw_stat):
        sig = "significant" if kw_p < 0.05 else "not significant"
        st.markdown(
            f"**{metric_label}** — Kruskal-Wallis: H={kw_stat:.2f}, "
            f"p={kw_p:.2e} ({sig})"
        )
    else:
        st.markdown(f"**{metric_label}** — insufficient groups for KW test")

    col_p, col_r = st.columns(2)

    with col_p:
        fig_pw = go.Figure(data=go.Heatmap(
            z=p_df.values,
            x=p_df.columns.tolist(),
            y=p_df.index.tolist(),
            text=p_df.map(
                lambda x: f"{x:.2e}" if pd.notna(x) else ""
            ).values,
            texttemplate="%{text}",
            textfont={"size": 11},
            colorscale=[[0, "#2ecc71"], [0.05, "#f1c40f"], [1, "#e74c3c"]],
            zmin=0, zmax=1,
            colorbar=dict(title="p-value"),
        ))
        fig_pw.update_layout(
            title=f"P-values ({metric_label})",
            height=400 + 30 * n,
        )
        st.plotly_chart(fig_pw, use_container_width=True)

    with col_r:
        fig_r = go.Figure(data=go.Heatmap(
            z=r_df.values,
            x=r_df.columns.tolist(),
            y=r_df.index.tolist(),
            text=r_df.map(
                lambda x: f"{x:.3f}" if pd.notna(x) else ""
            ).values,
            texttemplate="%{text}",
            textfont={"size": 11},
            colorscale="RdBu_r",
            zmid=0, zmin=-1, zmax=1,
            colorbar=dict(title="r"),
        ))
        fig_r.update_layout(
            title=f"Effect Size ({metric_label})",
            height=400 + 30 * n,
        )
        st.plotly_chart(fig_r, use_container_width=True)


def render_user_turn_view(df: pd.DataFrame) -> None:
    """Render the user-turn prediction visualization view."""
    metric_info = _resolve_user_turn_metric_columns(df)

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset Info")
    st.sidebar.write(f"Total samples: {len(df):,}")
    st.sidebar.write(f"Categories: {df['category'].nunique()}")
    st.sidebar.write(f"Personas: {df['persona'].nunique()}")
    if "prompt_id" in df.columns:
        st.sidebar.write(f"Prompts: {df['prompt_id'].nunique()}")

    # Category filter
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    categories = sorted(df["category"].unique())
    selected_categories = st.sidebar.multiselect(
        "Filter by category",
        options=categories,
        default=categories,
        key="ut_category_filter",
    )

    if selected_categories:
        filtered_df = df[df["category"].isin(selected_categories)]
    else:
        filtered_df = df

    # View mode toggle
    st.sidebar.markdown("---")
    st.sidebar.subheader("View Mode")
    view_mode = st.sidebar.radio(
        "Select view mode",
        options=["All Personas", "Group by Category"],
        index=0,
        key="ut_view_mode",
    )

    chart_df = filtered_df
    if view_mode == "Group by Category":
        group_by = "category"
    else:
        group_by = "persona"

    # Plot type toggle
    st.sidebar.markdown("---")
    st.sidebar.subheader("Chart Options")
    plot_type = st.sidebar.radio(
        "Plot type",
        options=["Box", "Violin", "Mean"],
        index=0,
        horizontal=True,
        key="ut_plot_type",
    )
    plot_type = plot_type.lower()

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Entropy",
        "Top-k Mass",
        "Token Count",
        "Correlations",
        "Statistical Tests",
        "Raw Data",
    ])

    with tab1:
        st.plotly_chart(
            create_user_turn_entropy_chart(chart_df, group_by, plot_type, metric_info),
            use_container_width=True,
        )
        with st.expander("Summary Statistics"):
            entropy_cols = metric_info["entropy_cols"]
            labels = metric_info["metric_labels"]
            stats_df = chart_df.groupby(group_by)[entropy_cols].agg(
                ["mean", "std", "median"]
            ).round(4)
            stats_df.columns = [
                f"{labels[col]} ({agg})"
                for col, agg in stats_df.columns
            ]
            st.dataframe(stats_df, use_container_width=True)

    with tab2:
        st.plotly_chart(
            create_user_turn_top_k_mass_chart(chart_df, group_by, plot_type, metric_info),
            use_container_width=True,
        )
        with st.expander("Summary Statistics"):
            top_k_cols = metric_info["top_k_cols"]
            labels = metric_info["metric_labels"]
            stats_df = chart_df.groupby(group_by)[top_k_cols].agg(
                ["mean", "std", "median"]
            ).round(4)
            stats_df.columns = [
                f"{labels[col]} ({agg})"
                for col, agg in stats_df.columns
            ]
            st.dataframe(stats_df, use_container_width=True)

    with tab3:
        st.plotly_chart(
            create_user_turn_token_count_chart(chart_df, group_by, plot_type, metric_info),
            use_container_width=True,
        )
        with st.expander("Summary Statistics"):
            token_cols = metric_info["token_cols"]
            labels = metric_info["metric_labels"]
            stats_df = chart_df.groupby(group_by)[token_cols].agg(
                ["mean", "std", "median", "min", "max"]
            ).round(2)
            stats_df.columns = [
                f"{labels[col]} ({agg})"
                for col, agg in stats_df.columns
            ]
            st.dataframe(stats_df, use_container_width=True)

    with tab4:
        if view_mode == "Group by Category":
            corr_title = "Metric Correlations (Grouped by Category)"
        else:
            corr_title = "Metric Correlations (All Personas)"

        corr_df, p_df = compute_user_turn_correlations(chart_df, metric_info)
        st.plotly_chart(
            create_correlation_heatmap(corr_df, corr_title),
            use_container_width=True,
        )

        st.subheader("Correlation Details")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Pearson Correlation Coefficients**")
            st.dataframe(corr_df.round(4))
        with col2:
            st.markdown("**P-values**")

            def format_pvalue(x):
                if x == 0.0:
                    return "< 1e-300"
                elif x < 0.0001:
                    return f"{x:.2e}"
                else:
                    return f"{x:.4f}"

            st.dataframe(p_df.map(format_pvalue))

        with st.expander("Interpretation Guide"):
            st.markdown("""
**Correlation Coefficient (r):**
- **r = 1.0**: Perfect positive correlation
- **r = 0.0**: No linear correlation
- **r = -1.0**: Perfect negative correlation

**Strength Guidelines:**
- |r| < 0.3: Weak
- 0.3 <= |r| < 0.7: Moderate
- |r| >= 0.7: Strong

**P-value:**
- p < 0.05: Statistically significant
- p < 0.01: Highly significant
- p < 0.001: Very highly significant

**Note:** Correlations are computed on the currently filtered data.
            """)

        st.info(f"Correlations computed on {len(chart_df):,} samples.")

    with tab5:
        st.subheader("Statistical Tests")
        st.markdown(
            "For each metric, a **Kruskal-Wallis** omnibus test checks whether "
            "distributions differ significantly across groups overall (p < 0.05). "
            "Pairwise **Mann-Whitney U** tests then compare each pair of groups. "
            "The **rank-biserial correlation (r)** measures effect size:\n"
            "- |r| < 0.1: negligible\n"
            "- 0.1 - 0.3: small\n"
            "- 0.3 - 0.5: medium\n"
            "- |r| > 0.5: large"
        )

        all_metric_cols = (
            metric_info["entropy_cols"]
            + metric_info["top_k_cols"]
            + metric_info["token_cols"]
        )
        labels = metric_info["metric_labels"]

        groups_list = sorted(chart_df[group_by].unique())
        if len(groups_list) < 2:
            st.warning("Need at least 2 groups for statistical tests.")
        else:
            for col in all_metric_cols:
                label = labels[col]
                kw_stat, kw_p, p_df, r_df = compute_user_turn_statistical_tests(
                    chart_df, group_by, col
                )
                render_user_turn_stat_test_heatmaps(
                    p_df, r_df, kw_stat, kw_p, label
                )
                st.markdown("---")

    with tab6:
        st.subheader("Raw Data")
        default_cols = [
            "category", "persona", "prompt_id",
            "response1_avg_entropy", "response2_avg_entropy",
            "response1_avg_top_k_mass", "response2_avg_top_k_mass",
            "response1_num_tokens", "response2_num_tokens",
        ]
        default_cols = [c for c in default_cols if c in chart_df.columns]
        display_cols = st.multiselect(
            "Select columns to display",
            options=chart_df.columns.tolist(),
            default=default_cols,
            key="ut_raw_cols",
        )
        if display_cols:
            st.dataframe(chart_df[display_cols], use_container_width=True)


def main():
    st.title("Persona Experiment Results")

    # Sidebar for file selection
    st.sidebar.header("Data Selection")

    # Discover all result files
    result_entries = discover_result_files()

    if not result_entries:
        st.error("No results files found in logs/ directory.")
        st.info(
            "Expected file patterns: logs/*/results.jsonl, results.json, "
            "judged_results.jsonl, embedding_variance*.parquet, or tc_llm_groups.jsonl. "
            "User-turn prediction dirs (user-turn-prediction-*) are auto-detected."
        )
        return

    # Build mapping from string key to (path, type)
    file_options: dict[str, tuple[Path, str]] = {}
    for file_path, result_type in result_entries:
        key = str(file_path)
        file_options[key] = (file_path, result_type)

    # Build label map for format_func
    label_map = {
        key: format_file_label(path, rtype)
        for key, (path, rtype) in file_options.items()
    }

    # File selector
    selected_key = st.sidebar.selectbox(
        "Select results file",
        options=list(file_options.keys()),
        format_func=lambda x: label_map[x],
    )

    if selected_key:
        file_path, result_type = file_options[selected_key]

        # Load data
        with st.spinner("Loading results..."):
            if result_type == "embeddings":
                df = load_multi_model_variance_data(str(file_path))
            elif result_type == "tc_llm":
                df = load_tc_llm_data(str(file_path))
            elif result_type == "user_turn":
                df = load_user_turn_data(str(file_path))
            else:
                df = load_results(file_path)
                # Flatten judge_parsed if this is judge results
                if result_type == "judge":
                    df = flatten_judge_parsed(df)

        # Render appropriate view based on result type
        if result_type == "judge":
            render_judge_view(df)
        elif result_type == "embeddings":
            render_embeddings_view(df, dir_path=file_path)
        elif result_type == "tc_llm":
            render_tc_llm_view(df)
        elif result_type == "user_turn":
            render_user_turn_view(df)
        else:
            render_metrics_view(df)


if __name__ == "__main__":
    main()
