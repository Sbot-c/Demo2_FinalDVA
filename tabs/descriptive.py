import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import theme


@st.cache_data
def _load(export_dir):
    genre_counts = pd.read_csv(export_dir / "genre_counts.csv")
    rating_dist = pd.read_csv(export_dir / "rating_distribution.csv")
    top_by_rank = pd.read_csv(export_dir / "top10_by_rank.csv")
    engagement_stats = pd.read_csv(export_dir / "engagement_summary_stats.csv", index_col=0)
    hours_hist = pd.read_csv(export_dir / "hours_played_histogram.csv")
    return genre_counts, rating_dist, top_by_rank, engagement_stats, hours_hist


def render(export_dir):
    st.subheader("Descriptive Analysis — What does the catalog look like today?")

    genre_counts, rating_dist, top_by_rank, engagement_stats, hours_hist = _load(export_dir)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            genre_counts.sort_values("count"), x="count", y="genre", orientation="h",
            title="Top 20 Genres by Frequency", color="count", color_continuous_scale=["#1d2942", theme.PLASMA_GREEN],
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            rating_dist, x="count", y="rating_category", orientation="h",
            title="Distribution of Overall Player Rating Categories", color="count",
            color_continuous_scale=["#1d2942", theme.PLASMA_GOLD],
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Top 10 Games by Ranking Type")
    rank_type = st.selectbox("Ranking type", ["Sales", "Revenue", "Review"], key="desc_rank_type")
    sub = top_by_rank[top_by_rank["rank_type"] == rank_type].sort_values("rank")
    fig = px.bar(sub, x="rank", y="game_name", orientation="h", title=f"Top 10 by {rank_type} Rank")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Review Engagement Distributions")
    col3, col4 = st.columns([2, 1])
    with col3:
        fig = go.Figure(go.Bar(x=hours_hist["bin_left"], y=hours_hist["count"], marker_color=theme.PLASMA_BLUE))
        fig.update_layout(title="Distribution of Hours Played (clipped at 500)", height=380,
                           xaxis_title="Hours played", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        st.markdown("**Engagement Summary Statistics**")
        st.dataframe(engagement_stats, use_container_width=True)
