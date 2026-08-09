import html
import json

import streamlit as st


@st.cache_data
def _load(export_dir):
    with open(export_dir / "fun_facts.json") as f:
        return json.load(f)


def achievement_card(icon, title, value, subtitle="", review=""):
    # Escaped: review text is user-written and may contain quotes or angle
    # brackets that would otherwise break out of the attribute.
    attr = f' data-review="{html.escape(review, quote=True)}"' if review else ""
    st.markdown(f"""
    <div class="ach-card"{attr}>
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
                          f"on {facts['most_helpful_review']['game']}",
                          facts["most_helpful_review"].get("excerpt", ""))
        achievement_card("😂", "Funniest Review on Record",
                          f"{facts['most_funny_review']['funny_votes']:,} funny votes",
                          f"on {facts['most_funny_review']['game']}",
                          facts["most_funny_review"].get("excerpt", ""))
        achievement_card("⏱️", "Most Addictive Game (by avg. hours)",
                          f"{facts['longest_avg_playtime_game']['avg_hours']:.0f} hours average",
                          f"{facts['longest_avg_playtime_game']['game']} — that's over 17 days playtime, on average")
    with col2:
        achievement_card("📜", "Longest Review Ever Written",
                          f"{facts['longest_single_review']['characters']:,} characters",
                          f"on {facts['longest_single_review']['game']} — a small novel",
                          facts["longest_single_review"].get("excerpt", ""))
        achievement_card("📈", "Sells Great, Rated... Not Great",
                          facts["sells_great_rated_poorly"],
                          "biggest gap between sales rank and player rating in the catalog")
        achievement_card("💎", "Hidden Gem",
                          facts["rated_great_sells_modestly"],
                          "high player rating without matching sales — a word-of-mouth underdog")

    st.caption(
        "Hover any card with a review record to read an excerpt. Excerpts are trimmed previews of "
        "player-written reviews from the source dataset, shown for illustration."
    )

    st.divider()
    st.markdown("#### Our Recommendations")
    st.markdown("""
    1. **Build in the highest-lift genre pairing** (see Prescriptive tab) — it's not a guess, it's
       backed by 439 statistically validated co-occurrence rules.
    2. **Target the highest-rated archetype** from the Clustering analysis rather than an unproven
       genre mashup — safer positioning, grounded in what already resonates with players.
    3. **Invest in content that earns engagement, not just playtime** — engagement concentrates:
       funny and helpful votes track each other closely (R² ≈ 0.74), so a review that earns one kind
       of attention earns the others. A few reviews carry the visibility that feeds Steam's discovery
       algorithm.
    4. **Watch the "sells great, rated poorly" pattern** — high sales don't guarantee reception;
       treat early rating trends as an early-warning signal, not an afterthought.
    """)
