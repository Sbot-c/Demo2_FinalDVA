import json

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


@st.cache_data
def _load_facts(export_dir):
    with open(export_dir / "fun_facts.json") as f:
        return json.load(f)


def render(export_dir):
    st.subheader("Descriptive Analysis — What does the catalog look like today?")

    genre_counts, rating_dist, top_by_rank, engagement_stats, hours_hist = _load(export_dir)
    facts = _load_facts(export_dir)

    # Dataset scale and data quality lead the tab: this is the methodology the
    # rest of the dashboard rests on, so it is stated before any finding.
    st.markdown("#### About the Dataset")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Games in catalogue", f"{facts['total_games']:,}")
    d2.metric("Ranking rows", f"{facts['ranking_rows']:,}",
              help=f"Sales, Revenue and Review ranks across {facts['ranking_games']} games")
    d3.metric("Reviews analysed", f"{facts['reviews_analyzed_this_run']:,}",
              help=f"After cleaning, from {facts['reviews_sampled_this_run']:,} sampled")
    d4.metric("Catalogue positivity", f"{facts['positive_rating_share_pct']:.0f}%",
              help="Share of games carrying some flavour of 'Positive' overall rating")

    kept_pct = facts["reviews_analyzed_this_run"] / facts["reviews_sampled_this_run"] * 100
    st.caption(
        f"The review file holds {facts['total_reviews_in_full_dataset']:,} rows "
        f"(~475 MB). It is read in chunks and randomly sampled with a fixed seed: "
        f"**{facts['reviews_sampled_this_run']:,} loaded → "
        f"{facts['reviews_analyzed_this_run']:,} after cleaning** ({kept_pct:.1f}% retained)."
    )

    st.info(f"**A real bug we caught:** {facts['data_bug_found']}")

    with st.expander("How the review data was cleaned — full step list"):
        st.markdown(f"""
1. **Chunked sampling.** The file is too large to hold in memory, so it is read in 100,000-row
   chunks and sampled to {facts['reviews_sampled_this_run']:,} rows with a fixed random seed
   (reproducible across runs).
2. **Numeric parsing.** `helpful` and `funny` arrive as strings with thousands separators
   (`"1,204"`). These are stripped and converted; anything unparseable becomes missing rather than
   being silently coerced to zero.
3. **Integer-overflow correction.** Any `funny` count above 100,000 is treated as missing. The
   highest genuine count in the dataset is {facts['most_funny_review']['funny_votes']:,}, so the
   threshold only catches the 4,294,967,295 (2³²−1) scrape artefacts — {facts['data_bug_found'].split()[0]}
   rows across the full file, 4 inside this sample.
4. **Feature derivation.** Review length, word count, exclamation count and capitalisation ratio
   are computed from the raw text; `hours_played` is coerced to numeric.
5. **Row filtering.** Rows missing review text, hours played, funny votes or helpful votes are
   dropped — **{facts['reviews_dropped_in_cleaning']:,} rows ({100 - kept_pct:.1f}%)**. Nothing is
   imputed: a fabricated vote count would flow straight into the regression target.

Cleaning was applied *before* the train/test split, and the split is stratified on the
recommendation label so the 83/17 class balance is preserved in both halves.
        """)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        gc = genre_counts.sort_values("count")
        fig = px.bar(
            gc, x="count", y="genre", orientation="h",
            title="Top 20 Genres by Frequency", color="count",
            color_continuous_scale=["#1d2942", theme.PLASMA_GREEN],
            text="count",
        )
        fig.update_traces(
            textposition="outside", textfont_size=11, cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>%{x} games carry this tag<extra></extra>",
        )
        fig.update_layout(
            showlegend=False, coloraxis_showscale=False, height=500,
            xaxis_title="Number of games carrying the tag", yaxis_title="Genre tag",
            xaxis_range=[0, gc["count"].max() * 1.15], margin=dict(l=10, r=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rd = rating_dist.copy()
        rd["pct"] = rd["count"] / rd["count"].sum() * 100
        rd["label"] = rd.apply(lambda r: f"{int(r['count'])}  ({r['pct']:.0f}%)", axis=1)
        fig = px.bar(
            rd, x="count", y="rating_category", orientation="h",
            title="Distribution of Overall Player Rating Categories", color="count",
            color_continuous_scale=["#1d2942", theme.PLASMA_GOLD],
            text="label",
        )
        fig.update_traces(
            textposition="outside", textfont_size=11, cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>%{x} games<extra></extra>",
        )
        fig.update_layout(
            showlegend=False, coloraxis_showscale=False, height=500,
            xaxis_title="Number of games", yaxis_title="Rating category",
            xaxis_range=[0, rd["count"].max() * 1.28], margin=dict(l=10, r=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Top 10 Games by Ranking Type")
    rank_type = st.selectbox("Ranking type", ["Sales", "Revenue", "Review"], key="desc_rank_type")
    sub = top_by_rank[top_by_rank["rank_type"] == rank_type].sort_values("rank")
    fig = px.bar(sub, x="rank", y="game_name", orientation="h",
                 title=f"Top 10 by {rank_type} Rank  (1 = best)", text="rank")
    fig.update_traces(
        marker_color=theme.PLASMA_GREEN, textposition="outside", textfont_size=11, cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>" + rank_type + " rank: %{x}<extra></extra>",
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        height=420, xaxis_title=f"{rank_type} rank (lower is better)", yaxis_title="Game",
        xaxis_range=[0, sub["rank"].max() * 1.15], margin=dict(l=10, r=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Review Engagement Distributions")
    col3, col4 = st.columns([2, 1])
    with col3:
        fig = go.Figure(go.Bar(
            x=hours_hist["bin_left"], y=hours_hist["count"], marker_color=theme.PLASMA_BLUE,
            hovertemplate="%{x:.0f}+ hours played<br>%{y:,} reviews<extra></extra>",
        ))
        median_h = engagement_stats.loc[engagement_stats.iloc[:, 0].astype(str).str.contains("50%|median", case=False, na=False)]
        fig.update_layout(
            title="Distribution of Hours Played (clipped at 500)", height=380,
            xaxis_title="Hours played at time of review", yaxis_title="Number of reviews",
            bargap=0.05,
        )
        fig.add_annotation(
            x=0.98, y=0.95, xref="paper", yref="paper", showarrow=False,
            text="Long right tail — a few players log enormous hours",
            font=dict(size=11, color=theme.MUTED), align="right",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        st.markdown("**Engagement Summary Statistics**")
        st.dataframe(engagement_stats, use_container_width=True)
