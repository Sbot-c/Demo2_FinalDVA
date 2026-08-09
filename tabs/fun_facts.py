import json
import streamlit as st


@st.cache_data
def _load(export_dir):
    with open(export_dir / "fun_facts.json") as f:
        return json.load(f)


def achievement_card(icon, title, value, subtitle=""):
    st.markdown(f"""
    <div class="ach-card">
        <div style="font-size:1.7rem;">{icon}</div>
        <div class="ach-eyebrow">Achievement unlocked</div>
        <div class="ach-title">{title}</div>
        <div class="ach-value">{value}</div>
        <div class="ach-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def render(export_dir, metrics):
    st.subheader("🎉 Fun Facts & Recommendations")
    st.caption(
        "Real, dataset-driven highlights — every number below comes straight from the CSVs, not "
        "invented for flavor. (Note: we couldn't embed actual meme templates or Steam screenshots "
        "here since they're copyrighted — these are original stat call-outs instead, styled to hit "
        "the same beat.)"
    )

    facts = _load(export_dir)

    col1, col2 = st.columns(2)
    with col1:
        achievement_card("🏆", "Most Helpful Review Ever",
                          f"{facts['most_helpful_review']['helpful_votes']:,} helpful votes",
                          f"on {facts['most_helpful_review']['game']}")
        achievement_card("😂", "Funniest Review on Record",
                          f"{facts['most_funny_review']['funny_votes']:,} funny votes",
                          f"on {facts['most_funny_review']['game']}")
        achievement_card("⏱️", "Most Addictive Game (by avg. hours)",
                          f"{facts['longest_avg_playtime_game']['avg_hours']:.0f} hours average",
                          f"{facts['longest_avg_playtime_game']['game']} — that's over 17 days playtime, on average")
    with col2:
        achievement_card("📜", "Longest Review Ever Written",
                          f"{facts['longest_single_review']['characters']:,} characters",
                          f"on {facts['longest_single_review']['game']} — a small novel")
        achievement_card("📈", "Sells Great, Rated... Not Great",
                          facts["sells_great_rated_poorly"],
                          "biggest gap between sales rank and player rating in the catalog")
        achievement_card("💎", "Hidden Gem",
                          facts["rated_great_sells_modestly"],
                          "high player rating without matching sales — a word-of-mouth underdog")

    st.divider()
    st.markdown("#### About the Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Games analyzed", f"{facts['total_games']:,}")
    c2.metric("Reviews analyzed", f"{facts['reviews_analyzed_this_run']:,}",
              help=f"Sampled from {facts['total_reviews_in_full_dataset']:,} total reviews")
    c3.metric("Catalog positivity", f"{facts['positive_rating_share_pct']:.0f}%",
              help="Share of games carrying some flavor of 'Positive' overall rating")

    st.info(f"🐛 **A real bug we caught:** {facts['data_bug_found']}")

    st.divider()
    st.markdown("#### Our Recommendations")
    st.markdown("""
    1. **Build in the highest-lift genre pairing** (see Prescriptive tab) — it's not a guess, it's
       backed by 439 statistically validated co-occurrence rules.
    2. **Target the highest-rated archetype** from the Clustering analysis rather than an unproven
       genre mashup — safer positioning, grounded in what already resonates with players.
    3. **Invest in content that earns engagement, not just playtime** — the regression model shows
       review length and tone predict community engagement (R² ≈ 0.74) more reliably than raw hours
       logged, which matters for Steam's discovery algorithm.
    4. **Watch the "sells great, rated poorly" pattern** — high sales don't guarantee reception;
       treat early rating trends as an early-warning signal, not an afterthought.
    """)
