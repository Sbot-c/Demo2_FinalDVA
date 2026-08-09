import json
import os
from pathlib import Path

import streamlit as st

import theme

st.set_page_config(
    page_title="Steam Games Analytics Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

EXPORT_DIR = Path(__file__).parent / "dashboard_exports"

@st.cache_data
def load_metrics():
    with open(EXPORT_DIR / "metrics_summary.json") as f:
        return json.load(f)


metrics = load_metrics()

theme.inject_css()
theme.register_plotly_template()


# ---------------------------------------------------------------- target roll-up ----
def target_scorecard(m):
    """Every stated model target, evaluated live.

    The detail lives in the Predictive tab. This only answers "how many did we
    hit?" so the answer is visible without opening a tab — computed, never
    hardcoded, so it stays honest if the numbers change.
    """
    clf, clu, reg, assoc = m["classification"], m["clustering"], m["regression"], m["association"]
    checks = [
        clf["accuracy"] > 0.80,
        0.80 <= clf["precision"] <= 0.85,
        clf["auc"] > 0.75,
        clf["train_test_gap"] < 0.02,
        clu["silhouette_k4"] > 0.1,
        reg["r2"] > 0.7,
        assoc["top_lift"] > 1.2,
    ]
    return sum(checks), len(checks)


hit, total = target_scorecard(metrics)
n_games = metrics["dataset"]["n_games"]
n_reviews = metrics["dataset"]["n_reviews_sampled"]

theme.hero(
    title="STEAM ANALYTICS",
    subtitle=f"{n_games} games · {n_reviews:,} reviews · descriptive → diagnostic → predictive → prescriptive",
    status_html=f"<b>{hit}/{total}</b> model targets met — full breakdown in Predictive",
)

st.divider()

# ---------------------------------------------------------------- tabs ----
tab_desc, tab_diag, tab_pred, tab_presc, tab_whatif, tab_fun = st.tabs([
    "DESCRIPTIVE", "DIAGNOSTIC", "PREDICTIVE", "PRESCRIPTIVE", "WHAT-IF", "FUN FACTS",
])

from tabs import descriptive, diagnostic, predictive, prescriptive, whatif, fun_facts

with tab_desc:
    descriptive.render(EXPORT_DIR)
with tab_diag:
    diagnostic.render(EXPORT_DIR)
with tab_pred:
    predictive.render(EXPORT_DIR, metrics)
with tab_presc:
    prescriptive.render(EXPORT_DIR, metrics)
with tab_whatif:
    whatif.render(EXPORT_DIR)
with tab_fun:
    fun_facts.render(EXPORT_DIR, metrics)
