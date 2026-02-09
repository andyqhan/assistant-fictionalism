"""
Streamlit webapp for visualizing embedding variance across personas.

Requires precomputed variance file. Generate it with:
    sbatch hpc/compute_embedding_variance.slurm

Run webapp with:
    uv run streamlit run src/viz/embeddings_app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Embedding Variance",
    page_icon="📊",
    layout="wide",
)


def discover_variance_files(logs_dir: str = "logs") -> list[str]:
    """Discover embedding variance files in consistency-inference directories.

    Args:
        logs_dir: Base directory to search for logs

    Returns:
        List of paths to embedding_variance.parquet files, sorted by directory name (newest first)
    """
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        return []

    variance_files = []
    for dir_path in logs_path.iterdir():
        if dir_path.is_dir() and dir_path.name.startswith("consistency-inference-"):
            variance_file = dir_path / "embedding_variance.parquet"
            if variance_file.exists():
                variance_files.append(str(variance_file))

    # Sort by directory name descending (newest job IDs first)
    return sorted(variance_files, reverse=True)


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


@st.cache_data
def load_variance_data(file_path: str) -> pd.DataFrame:
    """Load precomputed variance parquet file."""
    return pd.read_parquet(file_path)


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


def main():
    st.title("Embedding Variance by Persona")

    # Auto-discover variance files
    discovered_files = discover_variance_files()

    # Sidebar
    with st.sidebar:
        st.header("Data Selection")

        if discovered_files:
            # Show dropdown with discovered files
            file_path = st.selectbox(
                "Select variance file",
                options=discovered_files,
                format_func=lambda x: Path(x).parent.name,  # Show just the directory name
            )
            # Also allow manual input
            custom_path = st.text_input("Or enter custom path", value="")
            if custom_path:
                file_path = custom_path
        else:
            # No files discovered, fall back to text input
            st.warning("No variance files found in logs/")
            file_path = st.text_input(
                "Variance file path",
                value="logs/consistency-inference-1225210/embedding_variance.parquet",
            )

    # Load data
    try:
        with st.spinner("Loading variance data..."):
            df = load_variance_data(file_path)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        st.info(
            "Generate the variance file with:\n"
            "```\n"
            "sbatch hpc/compute_embedding_variance.slurm\n"
            "```"
        )
        return
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return

    # Find embedding columns
    embedding_columns = sorted(df["embedding_column"].unique())
    if not embedding_columns:
        st.error("No embedding columns found in the data.")
        return

    # Sidebar controls
    with st.sidebar:
        st.markdown("---")
        st.header("Configuration")

        # Embedding column selector
        selected_embedding = st.selectbox(
            "Embedding Column",
            options=embedding_columns,
            index=0,
        )

        # Filters
        st.markdown("---")
        st.header("Filters")

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
        st.header("Dataset Info")
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

    # Tab 1: Variance bar charts (existing functionality)
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


if __name__ == "__main__":
    main()
