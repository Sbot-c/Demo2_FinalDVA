import pickle
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_resource
def _load_models(export_dir):
    with open(export_dir / "classifier.pkl", "rb") as f:
        clf_bundle = pickle.load(f)
    with open(export_dir / "regressor.pkl", "rb") as f:
        reg_bundle = pickle.load(f)
    with open(export_dir / "kmeans.pkl", "rb") as f:
        km_bundle = pickle.load(f)
    return clf_bundle, reg_bundle, km_bundle


@st.cache_data
def _load_games(export_dir):
    return pd.read_csv(export_dir / "games_with_clusters.csv")


def render(export_dir):
    st.subheader("🎛️ What-If Analysis Simulator")
    st.caption(
        "Live predictions from the exact trained models used in the Predictive tab. "
        "Sliders are clamped to roughly the training data's real range — predictions outside "
        "that range extrapolate and are less reliable."
    )

    clf_bundle, reg_bundle, km_bundle = _load_models(export_dir)
    games_df = _load_games(export_dir)

    sim_review, sim_game = st.tabs(["📝 Simulate a Review", "🎮 Simulate a Game's Archetype"])

    # ---------------------------------------------------- Review-level simulator ----
    with sim_review:
        st.markdown("Adjust a hypothetical review's characteristics and see the classifier / "
                     "regressor react live.")
        c1, c2 = st.columns(2)
        with c1:
            hours = st.slider("Hours played", 0, 500, 50, step=5)
            helpful = st.slider("Helpful votes", 0, 200, 5, step=1)
            rec_assumed = st.radio("Assume review is:", ["Recommended", "Not Recommended"], horizontal=True)
        with c2:
            length = st.slider("Review length (characters)", 10, 3000, 300, step=10)
            words = st.slider("Word count", 2, 500, 50, step=2)
            excl = st.slider("Exclamation marks", 0, 20, 1, step=1)
            caps = st.slider("Caps ratio", 0.0, 0.5, 0.05, step=0.01)

        clf = clf_bundle["model"]
        clf_features = clf_bundle["features"]
        funny_median = clf_bundle["feature_medians"].get("funny_n", 1.0)

        row_clf = pd.DataFrame([{
            "hours_n": hours, "funny_n": funny_median, "helpful_n": helpful,
            "review_len": length, "word_count": words, "exclaim_count": excl, "caps_ratio": caps,
        }])[clf_features]
        proba_rec = clf.predict_proba(row_clf.values)[0, 1]

        reg = reg_bundle["model"]
        reg_features = reg_bundle["features"]
        row_reg = pd.DataFrame([{
            "hours_n": hours, "helpful_n": helpful, "review_len": length, "word_count": words,
            "exclaim_count": excl, "caps_ratio": caps, "rec_bin": 1 if rec_assumed == "Recommended" else 0,
        }])[reg_features]
        pred_funny = float(np.expm1(reg.predict(row_reg.values)[0]))

        st.divider()
        r1, r2 = st.columns(2)
        with r1:
            st.metric("P(review is Recommended)", f"{proba_rec:.1%}")
            st.progress(min(max(proba_rec, 0.0), 1.0))
        with r2:
            st.metric("Predicted funny votes", f"{pred_funny:.1f}")

    # ---------------------------------------------------- Game-level simulator ----
    with sim_game:
        lookup_mode = st.radio("Mode:", ["Look up an existing game", "Build a hypothetical game"], horizontal=True)

        if lookup_mode == "Look up an existing game":
            game_name = st.selectbox("Choose a game", sorted(games_df["name"].dropna().unique()))
            row = games_df[games_df["name"] == game_name].iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Assigned cluster", int(row["cluster"]))
            c2.metric("Rating score", f"{row['rating_score']:.0f}/6")
            c3.metric("Review count", f"{int(row['review_count']):,}")
            st.caption(f"Genres: {row['genre_filtered']}")

        else:
            st.caption("Pick genres and rough popularity/reception for a hypothetical new title — "
                       "see which existing archetype it would land closest to.")
            top_genres = km_bundle["top_genres"]
            chosen_genres = st.multiselect("Genres", top_genres, default=top_genres[:2])
            rating_score_in = st.slider("Assumed rating score (0=Very Negative … 6=Overwhelmingly Positive)",
                                         0, 6, 5)
            review_count_in = st.slider("Assumed review count", 0, 50000, 2000, step=100)

            if st.button("Predict archetype", type="primary"):
                mlb = km_bundle["mlb"]
                scaler = km_bundle["scaler"]
                pca = km_bundle["pca"]
                km = km_bundle["model"]

                genre_vec = mlb.transform([chosen_genres])
                X_new = np.hstack([genre_vec, [[rating_score_in]], [[np.log1p(review_count_in)]]])
                X_new_scaled = scaler.transform(X_new)
                X_new_pca = pca.transform(X_new_scaled)
                predicted_cluster = int(km.predict(X_new_pca)[0])

                st.success(f"This hypothetical game would land in **Cluster {predicted_cluster}**")
                profile = games_df[games_df["cluster"] == predicted_cluster]
                st.caption(
                    f"That archetype currently contains {len(profile)} games with an average rating "
                    f"of {profile['rating_score'].mean():.1f}/6."
                )
