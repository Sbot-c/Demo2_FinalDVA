import streamlit as st
import pandas as pd
import plotly.express as px


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
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Recommended vs Not Recommended — average review characteristics**")
        st.dataframe(rec_comparison, use_container_width=True)
        st.caption(
            "Recommended reviews tend to run longer and log more hours played — a signal the "
            "classification model in the Predictive tab leans on heavily."
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
    fig.update_layout(showlegend=False, height=450)
    st.plotly_chart(fig, use_container_width=True)
    corr_val = round(rating_vs_sales["rating_score"].corr(rating_vs_sales["sales_rank"]), 3)
    st.caption(
        f"Correlation (rating score vs. sales rank): **{corr_val}**. Negative is expected — a "
        "better rating should correspond to a lower (better) rank number."
    )
