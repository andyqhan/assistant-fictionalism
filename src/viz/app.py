"""
Streamlit webapp for visualizing persona inference results.

Supports three result types:
- metrics: entropy, top-k mass, thinking tokens from batch inference
- judge: endorsement/flagged analysis from LLM judge
- embeddings: embedding variance analysis from consistency experiments

Run with:
    uv run streamlit run src/viz/app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    """Detect if this is 'judge', 'metrics', or 'embeddings' results.

    Args:
        file_path: Path to the results file

    Returns:
        'judge', 'metrics', or 'embeddings'
    """
    if file_path.suffix == ".parquet":
        return "embeddings"
    if file_path.name == "judged_results.jsonl":
        return "judge"
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
            result_files.append((jsonl, "metrics"))
        elif json_file.exists():
            result_files.append((json_file, "metrics"))

        # Check for embedding variance files (can coexist with above)
        for parquet_file in sorted(dir_path.glob("embedding_variance*.parquet")):
            result_files.append((parquet_file, "embeddings"))

    return result_files


def format_file_label(file_path: Path, result_type: str) -> str:
    """Format a file path + result type for display in the file selector.

    Examples:
        consistency-inference-1225210 [embeddings]
        consistency-judge-pipeline-20260209_125904 [embeddings: kalm-embedding]
        prefill-inference-1472066 [judge]
        personae-inference-1855290 [metrics]
    """
    dir_name = file_path.parent.name

    if result_type == "embeddings":
        # Check for model suffix: embedding_variance_<model>.parquet
        stem = file_path.stem  # e.g. "embedding_variance_kalm-embedding" or "embedding_variance"
        prefix = "embedding_variance"
        if stem.startswith(prefix + "_"):
            model_name = stem[len(prefix) + 1:]
            return f"{dir_name} [embeddings: {model_name}]"
        return f"{dir_name} [embeddings]"
    elif result_type == "judge":
        return f"{dir_name} [judge]"
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
) -> go.Figure:
    """Create a bar chart of total variance per persona for a specific prompt.

    Args:
        df: DataFrame with columns: prompt_id, persona, total_variance, and optionally category
        prompt_id: The prompt ID to filter for
        question_text: The question text to display in the title
        has_category: Whether to color by category
        color_map: Optional dictionary mapping category names to colors for consistency

    Returns:
        Plotly figure with bar chart
    """
    prompt_df = df[df["prompt_id"] == prompt_id].sort_values(
        "total_variance", ascending=False
    )

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


def render_embeddings_view(df: pd.DataFrame) -> None:
    """Render the embeddings variance visualization view.

    Args:
        df: DataFrame loaded from embedding_variance*.parquet
    """
    # Find embedding columns
    embedding_columns = sorted(df["embedding_column"].unique())
    if not embedding_columns:
        st.error("No embedding columns found in the data.")
        return

    # Sidebar controls
    with st.sidebar:
        st.markdown("---")
        st.subheader("Configuration")

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
        st.write(f"Unique prompts: {df['prompt_id'].nunique()}")
        st.write(f"Unique personas: {df['persona'].nunique()}")
        if "category" in df.columns:
            st.write(f"Unique categories: {df['category'].nunique()}")
        st.write(f"Embedding columns: {len(embedding_columns)}")
        if "n_samples" in df.columns:
            st.write(f"Samples per group: {df['n_samples'].iloc[0]}")

    # Filter by selected embedding, categories, and personas
    filtered_df = df[df["embedding_column"] == selected_embedding]
    if selected_categories:
        filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]
    if selected_personas:
        filtered_df = filtered_df[filtered_df["persona"].isin(selected_personas)]

    if filtered_df.empty:
        st.warning("No data to display with current filters.")
        return

    # Get unique prompts and their questions (if available)
    if "question" in df.columns:
        prompts = (
            df[["prompt_id", "question"]]
            .drop_duplicates()
            .sort_values("prompt_id")
        )
    else:
        # Fallback if no question column
        prompts = pd.DataFrame({
            "prompt_id": sorted(df["prompt_id"].unique()),
            "question": [f"Prompt {pid}" for pid in sorted(df["prompt_id"].unique())],
        })

    # Check if category column exists and build color map
    has_category = "category" in df.columns
    color_map = build_category_color_map(df) if has_category else None

    # Create tabs
    tab1, tab2 = st.tabs(["Variance by Persona", "Ranking Changes"])

    # Tab 1: Variance bar charts
    with tab1:
        st.markdown(f"### Variance by Persona (using `{selected_embedding}`)")

        for _, row in prompts.iterrows():
            prompt_id = row["prompt_id"]
            question_text = row["question"]

            # Check if this prompt has data after filtering
            if prompt_id not in filtered_df["prompt_id"].values:
                continue

            fig = create_variance_bar_chart(
                filtered_df, prompt_id, question_text, has_category, color_map
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary statistics for this prompt
            prompt_variance = filtered_df[filtered_df["prompt_id"] == prompt_id]
            with st.expander(f"Statistics for Prompt {prompt_id}"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean Variance", f"{prompt_variance['total_variance'].mean():.4f}")
                col2.metric("Std Variance", f"{prompt_variance['total_variance'].std():.4f}")
                col3.metric("Min Variance", f"{prompt_variance['total_variance'].min():.4f}")
                col4.metric("Max Variance", f"{prompt_variance['total_variance'].max():.4f}")

        # Raw data view
        st.markdown("---")
        with st.expander("View Raw Variance Data"):
            display_df = filtered_df.sort_values(
                ["prompt_id", "total_variance"], ascending=[True, False]
            )
            st.dataframe(display_df, use_container_width=True)

    # Tab 2: Ranking changes across embedding columns (by category)
    with tab2:
        if not has_category:
            st.warning("Category data not available. This tab requires a 'category' column.")
        else:
            st.markdown("### Category Ranking Changes Across Embedding Columns")
            st.markdown(
                "This view shows how each category's average variance rank changes across "
                "different embedding columns. Rank 1 (top) = lowest average variance."
            )

            # Filter df by selected categories for ranking chart
            ranking_filtered_df = df.copy()
            if selected_categories:
                ranking_filtered_df = ranking_filtered_df[
                    ranking_filtered_df["category"].isin(selected_categories)
                ]

            for _, row in prompts.iterrows():
                prompt_id = row["prompt_id"]
                question_text = row["question"]

                # Check if this prompt has data after filtering
                if prompt_id not in ranking_filtered_df["prompt_id"].values:
                    continue

                fig = create_ranking_line_chart(
                    ranking_filtered_df, prompt_id, question_text, color_map
                )
                st.plotly_chart(fig, use_container_width=True)

                # Summary statistics for ranking changes
                ranking_df = calculate_category_rankings(ranking_filtered_df, prompt_id)
                with st.expander(f"Ranking Statistics for Prompt {prompt_id}"):
                    # Calculate rank variability per category
                    rank_stats = ranking_df.groupby("category")["rank"].agg(["mean", "std", "min", "max"])
                    rank_stats.columns = ["Mean Rank", "Rank Std", "Best Rank", "Worst Rank"]
                    rank_stats = rank_stats.sort_values("Mean Rank")
                    st.dataframe(rank_stats, use_container_width=True)


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
            "judged_results.jsonl, or embedding_variance*.parquet"
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
                df = load_variance_data(str(file_path))
            else:
                df = load_results(file_path)
                # Flatten judge_parsed if this is judge results
                if result_type == "judge":
                    df = flatten_judge_parsed(df)

        # Render appropriate view based on result type
        if result_type == "judge":
            render_judge_view(df)
        elif result_type == "embeddings":
            render_embeddings_view(df)
        else:
            render_metrics_view(df)


if __name__ == "__main__":
    main()
