"""
Streamlit webapp for visualizing persona inference results.

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

st.set_page_config(
    page_title="Persona Inference Results",
    page_icon="📊",
    layout="wide",
)


def load_results(file_path: Path) -> pd.DataFrame:
    """Load results JSON into a DataFrame."""
    with open(file_path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def create_entropy_boxplot(df: pd.DataFrame, group_by: str = "category") -> px.box:
    """Create box-and-whisker plot for entropy metrics per category or persona."""
    assert group_by in ("category", "persona"), f"Invalid group_by: {group_by}"

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

    fig = px.box(
        melted,
        x=group_by,
        y="entropy",
        color="metric",
        title=title,
        labels={"entropy": "Entropy", group_by: x_label, "metric": "Section"},
        category_orders={"metric": ["Thinking", "Output", "Overall"]},
    )
    fig.update_layout(
        boxmode="group",
        xaxis_tickangle=-45,
        height=900,
    )
    return fig


def create_top_k_mass_boxplot(df: pd.DataFrame, group_by: str = "category") -> px.box:
    """Create box-and-whisker plot for top-k mass metrics per category or persona."""
    assert group_by in ("category", "persona"), f"Invalid group_by: {group_by}"

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

    fig = px.box(
        melted,
        x=group_by,
        y="top_k_mass",
        color="metric",
        title=title,
        labels={"top_k_mass": "Top-k Mass", group_by: x_label, "metric": "Section"},
        category_orders={"metric": ["Thinking", "Output", "Overall"]},
    )
    fig.update_layout(
        boxmode="group",
        xaxis_tickangle=-45,
        height=900,
    )
    return fig


def create_thinking_tokens_boxplot(df: pd.DataFrame, group_by: str = "category") -> px.box:
    """Create box-and-whisker plot for thinking tokens per category or persona."""
    assert group_by in ("category", "persona"), f"Invalid group_by: {group_by}"

    x_label = "Persona Category" if group_by == "category" else "Persona"
    title = f"Number of Thinking Tokens by {x_label}"

    fig = px.box(
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


def main():
    st.title("Persona Inference Results Visualization")

    # Sidebar for file selection
    st.sidebar.header("Data Selection")

    # Find available results files
    logs_dir = Path("logs")
    results_files = list(logs_dir.glob("*/results.json"))

    if not results_files:
        st.error("No results files found in logs/ directory.")
        st.info("Expected path pattern: logs/personae-inference-*/results.json")
        return

    # File selector
    file_options = {str(f): f for f in sorted(results_files, reverse=True)}
    selected_file = st.sidebar.selectbox(
        "Select results file",
        options=list(file_options.keys()),
        format_func=lambda x: Path(x).parent.name,
    )

    if selected_file:
        # Load data
        with st.spinner("Loading results..."):
            df = load_results(file_options[selected_file])

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

        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Entropy", "Top-k Mass", "Thinking Tokens", "Correlations", "Raw Data"]
        )

        with tab1:
            st.plotly_chart(create_entropy_boxplot(chart_df, group_by), use_container_width=True)

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
            st.plotly_chart(create_top_k_mass_boxplot(chart_df, group_by), use_container_width=True)

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
                create_thinking_tokens_boxplot(chart_df, group_by), use_container_width=True
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


if __name__ == "__main__":
    main()
