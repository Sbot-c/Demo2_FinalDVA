import streamlit as st
import pandas as pd
import plotly.express as px

import theme


@st.cache_data
def _load(export_dir):
    corr = pd.read_csv(export_dir / "correlation_matrix.csv", index_col=0)
    rec_comparison = pd.read_csv(export_dir / "recommended_vs_not_comparison.csv", index_col=0)
    rating_vs_sales = pd.read_csv(export_dir / "rating_vs_salesrank.csv")
    return corr, rec_comparison, rating_vs_sales


def render(export_dir):
    st.subheader("Diagnostic Analysis — Why do these patterns happen?")

    corr, rec_comparison, rating_vs_sales = _load(export_dir)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Correlation Matrix — Review Engagement Features",
        )
        fig.update_traces(
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>correlation: %{z:.3f}<extra></extra>",
            textfont_size=12,
        )
        fig.update_layout(
            height=520, xaxis_title="", yaxis_title="",
            coloraxis_colorbar=dict(title="Pearson r", ticks="outside"),
        )
        fig.update_xaxes(side="bottom", tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Recommended vs Not Recommended — average review characteristics**")
        st.dataframe(rec_comparison, use_container_width=True)
        st.caption(
            "Not Recommended reviews run roughly 70% longer (599 vs 353 characters) and earn "
            "nearly twice the helpful votes (42.1 vs 22.7). Recommended reviews log slightly more "
            "hours played (139.6 vs 130.4) and more funny votes (8.9 vs 6.0)."
        )

    st.markdown("#### Does Player Rating Correlate with Sales Performance?")
    fig = px.box(
        rating_vs_sales, x="rating_score", y="sales_rank",
        title="Sales Rank Distribution by Player Rating Score (lower rank number = better sales)",
        labels={"rating_score": "Rating score (0=Very Negative … 6=Overwhelmingly Positive)",
                "sales_rank": "Sales rank"},
        color="rating_score",
        color_discrete_sequence=px.colors.sequential.Viridis,
    )
    fig.update_traces(hovertemplate="Rating score %{x}<br>Sales rank: %{y}<extra></extra>")
    fig.update_layout(
        showlegend=False, height=450,
        xaxis_title="Player rating score  (0 = Very Negative … 6 = Overwhelmingly Positive)",
        yaxis_title="Sales rank  (lower = better selling)",
    )
    fig.add_annotation(
        x=0.5, y=1.06, xref="paper", yref="paper", showarrow=False,
        text="Flat across the range — better-rated games do not sell better",
        font=dict(size=12, color=theme.MUTED),
    )
    st.plotly_chart(fig, use_container_width=True)
    corr_val = round(rating_vs_sales["rating_score"].corr(rating_vs_sales["sales_rank"]), 3)
    st.caption(
        f"Correlation (rating score vs. sales rank): **{corr_val}**. Negative is expected — a "
        "better rating should correspond to a lower (better) rank number."
    )
