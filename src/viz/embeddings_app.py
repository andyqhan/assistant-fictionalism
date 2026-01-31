"""
Streamlit webapp for visualizing embedding variance across personas.

Requires precomputed variance file. Generate it with:
    sbatch hpc/compute_embedding_variance.slurm

Run webapp with:
    uv run streamlit run src/viz/embeddings_app.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Embedding Variance",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_variance_data(file_path: str) -> pd.DataFrame:
    """Load precomputed variance parquet file."""
    return pd.read_parquet(file_path)


def create_variance_bar_chart(
    df: pd.DataFrame,
    prompt_id: int,
    question_text: str,
    has_category: bool = False,
) -> go.Figure:
    """Create a bar chart of total variance per persona for a specific prompt.

    Args:
        df: DataFrame with columns: prompt_id, persona, total_variance, and optionally category
        prompt_id: The prompt ID to filter for
        question_text: The question text to display in the title
        has_category: Whether to color by category

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


def main():
    st.title("Embedding Variance by Persona")

    # Default file path
    default_path = "logs/consistency-inference-1225210/embedding_variance.parquet"

    # Sidebar
    with st.sidebar:
        st.header("Data Selection")
        file_path = st.text_input("Variance file path", value=default_path)

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

    # Check if category column exists
    has_category = "category" in df.columns

    # Create charts for each prompt
    st.markdown(f"### Variance by Persona (using `{selected_embedding}`)")

    for _, row in prompts.iterrows():
        prompt_id = row["prompt_id"]
        question_text = row["question"]

        # Check if this prompt has data after filtering
        if prompt_id not in filtered_df["prompt_id"].values:
            continue

        fig = create_variance_bar_chart(filtered_df, prompt_id, question_text, has_category)
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


if __name__ == "__main__":
    main()
