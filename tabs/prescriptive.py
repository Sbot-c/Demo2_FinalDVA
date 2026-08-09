import streamlit as st
import pandas as pd


@st.cache_data
def _load(export_dir):
    with open(export_dir / "prescriptive_narrative.txt") as f:
        narrative = f.read()
    rules = pd.read_csv(export_dir / "association_rules.csv")
    cluster_profiles = pd.read_csv(export_dir / "cluster_profiles.csv")
    return narrative, rules, cluster_profiles


def render(export_dir, metrics):
    st.subheader("Prescriptive Analysis — What should a developer actually do with this?")

    narrative, rules, cluster_profiles = _load(export_dir)

    st.markdown(
        f"<div class='panel' style='font-size:1.05rem;line-height:1.6;'>{narrative}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Supporting Evidence")
    top_rule = rules.iloc[0]
    best_cluster = cluster_profiles.loc[cluster_profiles["avg_rating"].idxmax()]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Top genre pairing (lift)", f"{top_rule['lift']:.2f}")
        st.caption(f"{top_rule['antecedents']} → {top_rule['consequents']}")
    with c2:
        st.metric("Best-rated archetype", f"Cluster {int(best_cluster['cluster'])}")
        st.caption(f"Top genres: {best_cluster['top_genres']} · avg rating {best_cluster['avg_rating']}/6")
    with c3:
        st.metric("Recommendation model accuracy", f"{metrics['classification']['accuracy']:.1%}")
        st.caption(f"F1 {metrics['classification']['f1']:.1%} · engagement R² {metrics['regression']['r2']:.2f}")

    st.caption(
        "This narrative is generated directly from the notebook's own validated results — no "
        "external API call, so it always runs, even offline during a live presentation."
    )
