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

# Color scheme for endorsement types
ENDORSEMENT_COLORS = {
    "accept": "#2ecc71",    # Green
    "partial": "#f1c40f",   # Yellow
    "reject": "#e74c3c",    # Red
    "redirect": "#95a5a6",  # Gray
}

ENDORSEMENT_ORDER = ["accept", "partial", "reject", "redirect"]

st.set_page_config(
    page_title="Persona Inference Results",
    page_icon="📊",
    layout="wide",
)


def detect_result_type(file_path: Path) -> str:
    """Detect if this is 'judge' results or 'metrics' results.

    Args:
        file_path: Path to the results file

    Returns:
        'judge' if this is judged_results.jsonl, 'metrics' otherwise
    """
    if file_path.name == "judged_results.jsonl":
        return "judge"
    return "metrics"


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
    stats = df.groupby(group_by)[["thinking_chars", "output_chars"]].mean()

    # Sort by accept rate if requested
    if sort_by_accept and "endorsement" in df.columns:
        counts = df.groupby([group_by, "endorsement"]).size().unstack(fill_value=0)
        if "accept" in counts.columns:
            totals = counts.sum(axis=1)
            accept_rate = counts["accept"] / totals
            stats = stats.loc[accept_rate.sort_values(ascending=False).index]

    # Create grouped bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Thinking",
        x=stats.index,
        y=stats["thinking_chars"],
        marker_color="#3498db",  # Blue
    ))

    fig.add_trace(go.Bar(
        name="Output",
        x=stats.index,
        y=stats["output_chars"],
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


def main():
    st.title("Persona Inference Results Visualization")

    # Sidebar for file selection
    st.sidebar.header("Data Selection")

    # Find available results files
    logs_dir = Path("logs")

    # Find all result file types
    jsonl_files = set(logs_dir.glob("*/results.jsonl"))
    json_files = set(logs_dir.glob("*/results.json"))
    judged_files = set(logs_dir.glob("*/judged_results.jsonl"))

    # Build results_files list
    # For directories with judged_results.jsonl, prefer that
    # Otherwise prefer results.jsonl over results.json
    results_files = []
    judged_dirs = {f.parent for f in judged_files}
    jsonl_dirs = {f.parent for f in jsonl_files}

    # Add judged files first
    results_files.extend(judged_files)

    # Add jsonl files (excluding dirs with judged files)
    for f in jsonl_files:
        if f.parent not in judged_dirs:
            results_files.append(f)

    # Add json files (excluding dirs with judged or jsonl files)
    for f in json_files:
        if f.parent not in judged_dirs and f.parent not in jsonl_dirs:
            results_files.append(f)

    if not results_files:
        st.error("No results files found in logs/ directory.")
        st.info("Expected path pattern: logs/*/results.jsonl, results.json, or judged_results.jsonl")
        return

    # File selector
    file_options = {str(f): f for f in sorted(results_files, reverse=True)}
    selected_file = st.sidebar.selectbox(
        "Select results file",
        options=list(file_options.keys()),
        format_func=lambda x: Path(x).parent.name + (" [judge]" if "judged" in x else ""),
    )

    if selected_file:
        file_path = file_options[selected_file]
        result_type = detect_result_type(file_path)

        # Load data
        with st.spinner("Loading results..."):
            df = load_results(file_path)

            # Flatten judge_parsed if this is judge results
            if result_type == "judge":
                df = flatten_judge_parsed(df)

        # Render appropriate view based on result type
        if result_type == "judge":
            render_judge_view(df)
        else:
            render_metrics_view(df)


if __name__ == "__main__":
    main()
