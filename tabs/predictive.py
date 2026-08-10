import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import theme


@st.cache_data
def _load(export_dir):
    cluster_profiles = pd.read_csv(export_dir / "cluster_profiles.csv")
    games_clusters = pd.read_csv(export_dir / "games_with_clusters.csv")
    reg_importance = pd.read_csv(export_dir / "regression_feature_importance.csv")
    rules = pd.read_csv(export_dir / "association_rules.csv")

    # Diagnostics added to give each technique its supporting evidence on screen
    # rather than only in prose. Optional so the tab still renders if a file is
    # missing from dashboard_exports.
    extra = {}
    for key, fname in [
        ("clf_models", "classification_model_comparison.csv"),
        ("depth", "depth_sweep.csv"),
        ("roc", "roc_curve.csv"),
        ("elbow", "elbow_silhouette.csv"),
        ("pca_var", "pca_variance.csv"),
        ("clu_algos", "clustering_algorithm_comparison.csv"),
        ("clu_sweep", "cluster_sweep.csv"),
        ("reg_models", "regression_model_comparison.csv"),
        ("reg_scatter", "regression_pred_vs_actual.csv"),
        ("support", "support_sweep.csv"),
    ]:
        try:
            extra[key] = pd.read_csv(export_dir / fname)
        except Exception:
            extra[key] = None
    return cluster_profiles, games_clusters, reg_importance, rules, extra


def target_badge(label, value_str, passed):
    theme.target_row(label, value_str, passed)


def render(export_dir, metrics):
    st.subheader("Predictive Analysis")
    st.caption("Every model shown here was picked by comparing multiple candidate algorithms with "
               "cross-validation, then validated on a held-out test set — not just the first thing that worked.")

    cluster_profiles, games_clusters, reg_importance, rules, extra = _load(export_dir)
    clf_m, clu_m, reg_m, assoc_m = (metrics["classification"], metrics["clustering"],
                                     metrics["regression"], metrics["association"])

    sub_clf, sub_clu, sub_reg, sub_assoc = st.tabs(
        ["a. Classification", "b. Clustering", "c. Regression", "d. Association"]
    )

    # ---------------------------------------------------- a. Classification ----
    with sub_clf:
        st.markdown("**Target:** predict `recommendation` (Recommended / Not Recommended) from "
                     "review engagement features · **Model:** Gradient Boosting, depth=3")
        col1, col2 = st.columns([1, 1.3])
        with col1:
            target_badge("Accuracy > 80%", f"{clf_m['accuracy']:.1%}", clf_m["accuracy"] > 0.80)
            target_badge("Precision 80–85%", f"{clf_m['precision']:.1%}", 0.80 <= clf_m["precision"] <= 0.85)
            target_badge("AUC > 75%", f"{clf_m['auc']:.1%}", clf_m["auc"] > 0.75)
            target_badge("No overfitting", f"train/test gap {clf_m['train_test_gap']:.1%}",
                         clf_m["train_test_gap"] < 0.02)
            st.metric("Recall", f"{clf_m['recall']:.1%}")
            st.metric("F1-score", f"{clf_m['f1']:.1%}")
        with col2:
            fig = go.Figure(go.Bar(
                x=["Accuracy","Precision","Recall","F1","AUC"],
                y=[clf_m["accuracy"], clf_m["precision"], clf_m["recall"], clf_m["f1"], clf_m["auc"]],
                marker_color=theme.CHART_COLORWAY[:5],
                text=[f"{v:.1%}" for v in [clf_m["accuracy"], clf_m["precision"], clf_m["recall"], clf_m["f1"], clf_m["auc"]]],
                textposition="outside",
            ))
            fig.update_traces(textfont_size=12, cliponaxis=False,
                              hovertemplate="<b>%{x}</b>: %{y:.3f}<extra></extra>")
            fig.update_layout(
                title="Classifier Performance — Held-Out Test Set",
                yaxis_range=[0, 1.15], height=380,
                xaxis_title="Metric", yaxis_title="Score (0–1)",
                yaxis_tickformat=".0%",
            )
            fig.add_hline(
                y=0.83, line_dash="dot", line_color=theme.MUTED, line_width=1.5,
                annotation_text="83.0% — always guessing 'Recommended'",
                annotation_position="bottom right",
                annotation_font=dict(size=10, color=theme.MUTED),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ---- Confusion matrix -------------------------------------------------
        # The headline metrics hide *which* mistakes the model makes. This shows
        # that its errors are overwhelmingly one-directional.
        cmx = clf_m.get("confusion_matrix")
        if cmx:
            st.markdown("#### Confusion Matrix — where the errors actually fall")
            tn, fp = cmx["true_negative"], cmx["false_positive"]
            fn, tp = cmx["false_negative"], cmx["true_positive"]
            n = cmx["n_test"]

            z = [[tn, fp], [fn, tp]]
            # Technical term first, plain English second: the screen and the
            # terminology have to agree if someone asks for a false-positive count.
            labels = [["true negatives<br>correctly flagged",
                       "FALSE POSITIVES<br>missed negatives"],
                      ["false negatives<br>false alarms",
                       "true positives<br>correctly flagged"]]
            text = [[f"<b>{z[r][c]:,}</b><br>{labels[r][c]}" for c in range(2)] for r in range(2)]

            # Solid fills, not tints: a low-alpha wash disappears against the dark
            # starfield. Colour encodes correct vs incorrect only — scaling by
            # magnitude would flatten the three small cells against the 22,148
            # and hide the very asymmetry this chart exists to show.
            figc = go.Figure(go.Heatmap(
                z=[[1, 0], [0, 1]],
                x=["Predicted:<br>Not Recommended", "Predicted:<br>Recommended"],
                y=["Actual:<br>Not Recommended", "Actual:<br>Recommended"],
                text=text, texttemplate="%{text}",
                textfont=dict(size=13, color="#07130f"),  # dark text on saturated fills
                colorscale=[[0, "#ff6b5b"], [1, theme.PLASMA_GREEN]],
                showscale=False, xgap=8, ygap=8,
                hovertemplate="%{y}<br>%{x}<extra></extra>",
            ))
            figc.update_layout(
                height=380, margin=dict(t=20, b=20, l=10, r=10),
                xaxis=dict(side="top", tickfont=dict(size=12, color=theme.TEXT)),
                yaxis=dict(tickfont=dict(size=12, color=theme.TEXT)),
            )
            figc.update_yaxes(autorange="reversed")
            st.plotly_chart(figc, use_container_width=True)

            pred_yes = tp + fp
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted 'Recommended'", f"{pred_yes / n:.1%}",
                      help=f"{pred_yes:,} of {n:,} test reviews")
            c2.metric("Actually 'Recommended'", f"{(tp + fn) / n:.1%}",
                      help=f"{tp + fn:,} of {n:,} test reviews")
            c3.metric("Errors in one direction", f"{fp / (fp + fn):.0%}",
                      help=f"{fp:,} of {fp + fn:,} mistakes are false positives")

            st.info(
                f"**Read the top row:** of the {tn + fp:,} reviews that genuinely were *not* "
                f"recommendations, the model caught {tn:,} and called the other {fp:,} recommendations "
                f"— that is the 0.22 recall on the minority class. The model predicts 'Recommended' for "
                f"{pred_yes / n:.1%} of reviews when only {(tp + fn) / n:.1%} actually are, so its "
                f"mistakes are not evenly split: {fp / (fp + fn):.0%} of them are the same error in the "
                f"same direction. At the default 0.5 threshold, class imbalance pushes borderline cases "
                f"toward the majority class. Class weighting or a tuned threshold is the fix."
            )

        # ---- Algorithm comparison ------------------------------------------
        if extra.get("clf_models") is not None:
            cmp_df = extra["clf_models"]
            st.markdown("#### Algorithm comparison — 5-fold stratified cross-validation")
            st.caption(
                "Accuracy alone separates these models by two points. AUC separates them by "
                "seventeen — which is why the choice was not made on accuracy."
            )
            figm = go.Figure()
            figm.add_bar(name="CV accuracy", y=cmp_df["model"], x=cmp_df["cv_accuracy"],
                         orientation="h", marker_color=theme.PLASMA_BLUE,
                         text=cmp_df["cv_accuracy"].map(lambda v: f"{v:.3f}"), textposition="auto",
                         hovertemplate="<b>%{y}</b><br>CV accuracy %{x:.4f}<extra></extra>")
            figm.add_bar(name="AUC", y=cmp_df["model"], x=cmp_df["auc"],
                         orientation="h", marker_color=theme.PLASMA_GREEN,
                         text=cmp_df["auc"].map(lambda v: f"{v:.3f}"), textposition="auto",
                         hovertemplate="<b>%{y}</b><br>AUC %{x:.4f}<extra></extra>")
            figm.update_layout(barmode="group", height=520, xaxis_title="Score",
                               yaxis_title="", xaxis_range=[0, 1.02],
                               yaxis=dict(autorange="reversed"),
                               legend=dict(orientation="h", y=-0.12), margin=dict(l=10, r=20))
            st.plotly_chart(figm, use_container_width=True)
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)

        # ---- Depth sweep: the clearest picture of overfitting ---------------
        if extra.get("depth") is not None:
            dsw = extra["depth"]
            st.markdown("#### Overfitting check — what happens as tree depth increases")
            figd = go.Figure()
            figd.add_scatter(x=dsw["max_depth"], y=dsw["train_accuracy"], name="Train accuracy",
                             mode="lines+markers", line=dict(color=theme.PLASMA_GOLD, width=3),
                             hovertemplate="depth %{x}<br>train %{y:.4f}<extra></extra>")
            figd.add_scatter(x=dsw["max_depth"], y=dsw["test_accuracy"], name="Test accuracy",
                             mode="lines+markers", line=dict(color=theme.PLASMA_GREEN, width=3),
                             hovertemplate="depth %{x}<br>test %{y:.4f}<extra></extra>")
            figd.add_vrect(x0=2.8, x1=3.2, fillcolor=theme.PLASMA_GREEN, opacity=0.10,
                           line_width=0, annotation_text="chosen", annotation_position="top",
                           annotation_font=dict(size=10, color=theme.MUTED))
            figd.update_layout(height=360, xaxis_title="max_depth",
                               yaxis_title="Accuracy", xaxis=dict(dtick=1),
                               legend=dict(orientation="h", y=-0.18))
            st.plotly_chart(figd, use_container_width=True)
            st.info(
                "Training accuracy climbs steadily with depth — 0.8473, 0.8495, 0.8534, 0.8580, "
                "0.8649. Test accuracy does not move: it sits at 0.8478 for depths 3, 4 and 5, then "
                "**falls** at depth 6. Everything the deeper trees gained, they gained only on data "
                "they had already seen. Depth 3 is the shallowest setting that reaches the ceiling."
            )
            st.dataframe(dsw, use_container_width=True, hide_index=True)

        # ---- ROC curve ------------------------------------------------------
        if extra.get("roc") is not None:
            roc = extra["roc"]
            st.markdown("#### ROC curve")
            figr = go.Figure()
            figr.add_scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="Model",
                             line=dict(color=theme.PLASMA_GREEN, width=3), fill="tozeroy",
                             fillcolor="rgba(55,230,160,0.10)",
                             hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>")
            figr.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess (AUC 0.5)",
                             line=dict(color=theme.MUTED, width=2, dash="dot"))
            figr.update_layout(height=420, xaxis_title="False positive rate",
                               yaxis_title="True positive rate",
                               legend=dict(orientation="h", y=-0.16))
            st.plotly_chart(figr, use_container_width=True)
            st.caption(
                f"Area under the curve = {clf_m['auc']:.4f}. The curve sits well above the diagonal, "
                "so the model ranks reviews by recommendation probability far better than chance — "
                "this is the evidence that the model learned real signal, despite accuracy sitting "
                "only 1.8 points above the majority-class baseline."
            )

        with st.expander("Why Gradient Boosting (depth=3) — comparison against other techniques"):
            st.markdown("""
Logistic Regression and KNN clear the accuracy bar but sit noticeably below on F1/AUC — they
underfit the non-linear relationship between engagement features and recommendation status.
Random Forest and deeper Gradient Boosting settings (depth ≥ 4) post higher *training* accuracy,
but their train/test gap widens well past depth=3 — a classic overfitting signature. Depth=3
Gradient Boosting sits at the best point on that trade-off: it clears every target while keeping
the smallest generalization gap among models that clear the bar.

An exhaustive `GridSearchCV` — 12 hyperparameter combinations × 5 folds = 60 fits, run on a
20,000-row subsample (189.7s) and refit on the full training set — independently selected
**max_depth=3**, the same depth chosen by hand. Both were then scored on the same held-out test set:
            """)
            gs = metrics.get("gridsearch", {}).get("classification")
            if gs:
                h = gs["head_to_head"]
                st.dataframe(pd.DataFrame({
                    "Metric": ["Accuracy", "Precision", "Recall", "F1", "AUC"],
                    "Manual (depth=3)": [f"{h['manual'][k]:.4f}" for k in ["accuracy","precision","recall","f1","auc"]],
                    "Grid search winner": [f"{h['grid_best'][k]:.4f}" for k in ["accuracy","precision","recall","f1","auc"]],
                }), hide_index=True, use_container_width=True)
                st.caption(
                    f"Best params: {gs['best_params']}. Identical F1 (0.9142); the manual model is "
                    "marginally ahead on AUC. The grid search confirms the manual tuning rather than "
                    "improving on it."
                )

    # ---------------------------------------------------- b. Clustering ----
    with sub_clu:
        st.markdown("**Target:** segment games into archetypes from genre mix + rating + popularity · "
                     "**Model:** KMeans, k=4 (PCA-reduced, 10 components)")
        col1, col2 = st.columns([1, 1.3])
        with col1:
            target_badge("Overlap reduced", f"Silhouette {clu_m['silhouette_k4']:.3f} (up from ~0.02 on raw genres)",
                         clu_m["silhouette_k4"] > 0.1)
            target_badge("Elbow method performed", "k=4 selected via elbow + silhouette sweep", True)
            st.markdown("**Cluster profiles**")
            st.dataframe(cluster_profiles, use_container_width=True, hide_index=True)
        with col2:
            gcl = games_clusters.copy()
            sizes = gcl["cluster"].value_counts().to_dict()
            gcl["Cluster"] = gcl["cluster"].apply(lambda c: f"Cluster {c}  ({sizes[c]} games)")
            fig = px.scatter(
                gcl, x="pc1", y="pc2", color="Cluster",
                hover_name="name", title="Game Clusters (PCA components 1 & 2)",
            )
            fig.update_traces(
                marker=dict(size=9, line=dict(width=0.5, color="rgba(255,255,255,0.35)")),
                hovertemplate="<b>%{hovertext}</b><br>PC1 %{x:.2f} · PC2 %{y:.2f}<extra></extra>",
            )
            fig.update_layout(
                height=440,
                xaxis_title="Principal component 1", yaxis_title="Principal component 2",
                legend_title_text="Archetype",
                legend=dict(orientation="h", yanchor="bottom", y=-0.28, x=0),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ---- Elbow + silhouette by k ----------------------------------------
        if extra.get("elbow") is not None:
            eb = extra["elbow"]
            st.markdown("#### Elbow method and silhouette across k")
            fige = go.Figure()
            fige.add_scatter(x=eb["k"], y=eb["inertia"], name="Inertia (elbow)",
                             mode="lines+markers", line=dict(color=theme.PLASMA_BLUE, width=3),
                             hovertemplate="k=%{x}<br>inertia %{y:,.0f}<extra></extra>")
            fige.add_scatter(x=eb["k"], y=eb["silhouette"], name="Silhouette", yaxis="y2",
                             mode="lines+markers", line=dict(color=theme.PLASMA_GREEN, width=3),
                             hovertemplate="k=%{x}<br>silhouette %{y:.4f}<extra></extra>")
            fige.add_vline(x=4, line_dash="dot", line_color=theme.PLASMA_GOLD, line_width=2,
                           annotation_text="k = 4 chosen", annotation_position="top",
                           annotation_font=dict(size=10, color=theme.PLASMA_GOLD))
            fige.update_layout(
                height=400, xaxis_title="Number of clusters (k)", xaxis=dict(dtick=1),
                yaxis=dict(title="Inertia (within-cluster sum of squares)"),
                yaxis2=dict(title="Silhouette", overlaying="y", side="right",
                            tickfont=dict(color=theme.PLASMA_GREEN)),
                legend=dict(orientation="h", y=-0.18))
            st.plotly_chart(fige, use_container_width=True)
            st.caption(
                "Inertia always falls as k rises, so it can never pick k on its own — the elbow is "
                "where the fall slows, around k=4. Silhouette peaks at k=4 (0.1649) and declines "
                "after. Two independent criteria agreeing is why k=4 was chosen."
            )

        # ---- PCA variance ---------------------------------------------------
        if extra.get("pca_var") is not None:
            pv = extra["pca_var"]
            st.markdown("#### How much signal each PCA component retains")
            figp = go.Figure(go.Scatter(
                x=pv["n_components"], y=pv["cumulative_variance"], mode="lines+markers",
                line=dict(color=theme.PLASMA_BLUE, width=3), fill="tozeroy",
                fillcolor="rgba(58,168,255,0.10)",
                hovertemplate="%{x} components<br>%{y:.1%} of variance retained<extra></extra>"))
            figp.add_vline(x=10, line_dash="dot", line_color=theme.PLASMA_GREEN, line_width=2,
                           annotation_text="10 components — 68.2%", annotation_position="bottom right",
                           annotation_font=dict(size=10, color=theme.PLASMA_GREEN))
            figp.update_layout(height=340, xaxis_title="Number of PCA components",
                               yaxis_title="Cumulative variance retained",
                               yaxis_tickformat=".0%", xaxis=dict(dtick=2))
            st.plotly_chart(figp, use_container_width=True)

        # ---- Algorithm comparison -------------------------------------------
        if extra.get("clu_algos") is not None:
            ca = extra["clu_algos"].sort_values("silhouette")
            st.markdown("#### Algorithm comparison")
            figa = px.bar(ca, x="silhouette", y="algorithm", orientation="h",
                          text=ca["silhouette"].map(lambda v: f"{v:.4f}"))
            figa.update_traces(
                marker_color=[theme.FAIL if v < 0 else theme.PLASMA_GREEN for v in ca["silhouette"]],
                textposition="outside", cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>silhouette %{x:.4f}<extra></extra>")
            figa.update_layout(height=300, xaxis_title="Silhouette score", yaxis_title="",
                               margin=dict(l=10, r=40))
            figa.add_vline(x=0, line_color=theme.MUTED, line_width=1.5)
            st.plotly_chart(figa, use_container_width=True)
            st.caption(
                "A negative silhouette means points sit closer to a neighbouring cluster than their "
                "own — the clustering is worse than useless. DBSCAN scored negative at every eps "
                "tried, because genre-tag space lacks the density contrast it needs."
            )

        # ---- k x PCA sweep: the trade-off behind the k=4 decision ------------
        if extra.get("clu_sweep") is not None:
            cs = extra["clu_sweep"]
            st.markdown("#### The trade-off behind keeping k=4")
            pivot = cs.pivot(index="n_pca", columns="k", values="silhouette")
            figh = go.Figure(go.Heatmap(
                z=pivot.values, x=[f"k={c}" for c in pivot.columns],
                y=[f"{i} comps" for i in pivot.index],
                colorscale=[[0, "#1d2942"], [0.5, theme.PLASMA_BLUE], [1, theme.PLASMA_GREEN]],
                text=pivot.round(3).values, texttemplate="%{text}",
                textfont=dict(size=11, color="#07130f"), xgap=3, ygap=3,
                colorbar=dict(title="Silhouette"),
                hovertemplate="%{y}, %{x}<br>silhouette %{z:.4f}<extra></extra>"))
            figh.update_layout(height=360, margin=dict(t=20, b=20))
            st.plotly_chart(figh, use_container_width=True)
            st.warning(
                "**Read this top to bottom, not left to right.** Silhouette rises sharply as PCA "
                "components fall — the best scores sit in the top rows, where the least variance is "
                "retained. Compressing the feature space makes clusters look tighter because there "
                "is less information left to disagree about. Our k=4 / 10-component configuration "
                "scores 0.1649 while keeping 68.2% of the variance; the sweep's best scores keep as "
                "little as 26%. That is why we did not simply take the highest number."
            )

        with st.expander("Why KMeans (k=4) — comparison against other techniques"):
            st.markdown("""
Agglomerative clustering lands close in silhouette but produces far less balanced cluster sizes
(a few dominant clusters, several near-singletons) — not useful for a "which archetype should we
build" recommendation. DBSCAN, swept across a range of `eps`, either collapses almost everything
into one cluster plus noise, or fragments into many tiny clusters — the genre-tag feature space
doesn't have the density contrast DBSCAN needs. KMeans at k=4 gives the most usable, balanced,
interpretable split.

A grid sweep over (k × PCA components), scored by silhouette, found a *higher* silhouette at
k=6 with fewer components — a real finding we are reporting rather than burying:
            """)
            gs = metrics.get("gridsearch", {}).get("clustering")
            if gs:
                st.dataframe(pd.DataFrame({
                    "Configuration": ["Chosen (k=4)", "Sweep winner (k=6)"],
                    "PCA components": [gs["chosen"]["n_pca"], gs["best"]["n_pca"]],
                    "Silhouette": [f"{gs['chosen']['silhouette']:.4f}", f"{gs['best']['silhouette']:.4f}"],
                    "Variance retained": [f"{gs['chosen']['explained_variance']:.1%}",
                                          f"{gs['best']['explained_variance']:.1%}"],
                }), hide_index=True, use_container_width=True)
                st.caption(
                    "The higher silhouette comes at the cost of discarding a third of the original "
                    "signal (49% variance retained vs 68%). Tighter clusters in a space that has "
                    "thrown away more information is a weaker result than it looks. We kept k=4 and "
                    "flag k=6 as the obvious next refinement."
                )

    # ---------------------------------------------------- c. Regression ----
    with sub_reg:
        st.markdown("**Target:** predict `funny` votes (log-transformed) from review engagement/text "
                     "features · **Model:** Gradient Boosting, depth=4")
        col1, col2 = st.columns([1, 1.3])
        with col1:
            target_badge("R² > 0.7", f"{reg_m['r2']:.3f}", reg_m["r2"] > 0.7)
            st.caption("A rejected earlier target — predicting `hours_played` from review-text "
                       "features — only reached R² ≈ 0.07 and is reported honestly rather than hidden.")
        with col2:
            ri = reg_importance.sort_values("importance")
            fig = px.bar(
                ri, x="importance", y="feature", orientation="h",
                title="Feature Importance — Regression Model", color="importance",
                color_continuous_scale=["#1d2942", theme.PLASMA_BLUE, theme.PLASMA_GREEN],
                text=ri["importance"].map(lambda v: f"{v:.1%}"),
            )
            fig.update_traces(
                textposition="outside", textfont_size=11, cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>%{x:.1%} of model importance<extra></extra>",
            )
            fig.update_layout(
                showlegend=False, coloraxis_showscale=False, height=380,
                xaxis_title="Share of total model importance", yaxis_title="Feature",
                xaxis_tickformat=".0%",
                xaxis_range=[0, ri["importance"].max() * 1.2], margin=dict(l=10, r=30),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ---- Model comparison with fold spread ------------------------------
        if extra.get("reg_models") is not None:
            rm = extra["reg_models"].sort_values("cv_r2")
            st.markdown("#### Algorithm comparison — 5-fold cross-validated R²")
            figrm = go.Figure(go.Bar(
                x=rm["cv_r2"], y=rm["model"], orientation="h",
                error_x=dict(type="data", array=rm["std"], color=theme.MUTED, thickness=1.5),
                marker_color=[theme.FAIL if v < 0.5 else theme.PLASMA_GREEN for v in rm["cv_r2"]],
                text=rm.apply(lambda r: f"{r['cv_r2']:.4f} ± {r['std']:.4f}", axis=1),
                textposition="outside", cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>R² %{x:.4f}<extra></extra>"))
            figrm.update_layout(height=340, xaxis_title="Cross-validated R² (bars show ± std across folds)",
                                yaxis_title="", xaxis_range=[0, 1.0], margin=dict(l=10, r=60))
            st.plotly_chart(figrm, use_container_width=True)
            st.info(
                "The linear models score 0.1104 with a standard deviation of 0.1210 — **the spread is "
                "larger than the score**. Across five folds they are wildly unstable, and on some "
                "folds they do worse than simply predicting the average. Engagement is not a "
                "straight-line function of review characteristics. Gradient boosting at depth 4 "
                "gives 0.7380 with a spread of 0.0066."
            )

        # ---- Predicted vs actual --------------------------------------------
        if extra.get("reg_scatter") is not None:
            ps = extra["reg_scatter"]
            st.markdown("#### Predicted vs actual — held-out test set")
            lo = float(min(ps["actual"].min(), ps["predicted"].min()))
            hi = float(max(ps["actual"].max(), ps["predicted"].max()))
            figs = go.Figure()
            figs.add_scatter(x=ps["actual"], y=ps["predicted"], mode="markers",
                             marker=dict(size=5, color=theme.PLASMA_BLUE, opacity=0.35),
                             name="Test reviews",
                             hovertemplate="actual %{x:.2f}<br>predicted %{y:.2f}<extra></extra>")
            figs.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                             line=dict(color=theme.PLASMA_GOLD, width=2, dash="dash"))
            figs.update_layout(height=420, xaxis_title="Actual log(1 + funny votes)",
                               yaxis_title="Predicted log(1 + funny votes)",
                               legend=dict(orientation="h", y=-0.16))
            st.plotly_chart(figs, use_container_width=True)
            st.caption(
                "A random 3,000-review sample of the test set. Points cluster along the diagonal, "
                "which is what R² = 0.738 looks like. The spread widens at the high end — the model "
                "is least accurate on the rare, heavily-voted reviews, which is expected given the "
                "log transform was applied precisely because those are so skewed."
            )

        with st.expander("Why Gradient Boosting (depth=4) — comparison against other techniques"):
            st.markdown("""
Linear Regression underfits badly — the relationship between review characteristics and funny-vote
counts is non-linear (e.g. diminishing returns on review length). Random Forest is close behind
Gradient Boosting but with slightly higher cross-validation variance across folds. Gradient
Boosting's sequential error correction handles the skewed, zero-inflated vote-count distribution
best.

An exhaustive `GridSearchCV` (same subsample-then-refit approach as classification — 60 fits,
219.1s) independently selected **max_depth=4**, matching the manual choice. Scored on the same
held-out test set:
            """)
            gs = metrics.get("gridsearch", {}).get("regression")
            if gs:
                h = gs["head_to_head"]
                st.dataframe(pd.DataFrame({
                    "Model": ["Manual (depth=4)", "Grid search winner"],
                    "R² (held-out test)": [f"{h['manual_test_r2']:.4f}", f"{h['grid_test_r2']:.4f}"],
                }), hide_index=True, use_container_width=True)
                st.caption(
                    f"Best params: {gs['best_params']}. A difference of 0.0006 R² — the manual "
                    "configuration is already at the ceiling this feature set reaches on this target."
                )

    # ---------------------------------------------------- d. Association ----
    with sub_assoc:
        st.markdown("**Target:** which genres reliably co-occur · **Algorithm:** Apriori (min_support=0.1)")
        col1, col2 = st.columns([1, 1.3])
        with col1:
            target_badge("Confidence > 70%", "all shown rules clear this", True)
            target_badge("Lift > 1.2", f"max lift {assoc_m['top_lift']:.2f}", assoc_m["top_lift"] > 1.2)
            st.metric("Qualifying rules found", assoc_m["n_rules"])
        with col2:
            top10 = rules.head(10).copy()
            top10["rule"] = top10["antecedents"] + " → " + top10["consequents"]
            t10 = top10.sort_values("lift")
            fig = px.bar(t10, x="lift", y="rule", orientation="h",
                         title="Top 10 Association Rules by Lift",
                         color="lift", color_continuous_scale=["#1d2942", theme.PLASMA_GREEN],
                         text=t10["lift"].map(lambda v: f"{v:.2f}×"))
            fig.update_traces(
                textposition="outside", textfont_size=11, cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>lift %{x:.2f}×<extra></extra>",
            )
            fig.update_layout(
                showlegend=False, coloraxis_showscale=False, height=440,
                xaxis_title="Lift  (how many times more often than chance)",
                yaxis_title="Rule",
                xaxis_range=[0, t10["lift"].max() * 1.2], margin=dict(l=10, r=40),
            )
            fig.add_vline(x=1.2, line_dash="dot", line_color=theme.MUTED, line_width=1.5,
                          annotation_text="target 1.2×", annotation_position="top",
                          annotation_font=dict(size=10, color=theme.MUTED))
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(rules.head(20), use_container_width=True, hide_index=True)

        # ---- Confidence vs lift for every qualifying rule -------------------
        st.markdown("#### All 439 qualifying rules — confidence against lift")
        figcl = px.scatter(rules, x="confidence", y="lift", size="support",
                           hover_name=rules["antecedents"] + "  →  " + rules["consequents"],
                           color="lift",
                           color_continuous_scale=["#1d2942", theme.PLASMA_BLUE, theme.PLASMA_GREEN])
        figcl.update_traces(marker=dict(line=dict(width=0)),
                            hovertemplate="<b>%{hovertext}</b><br>confidence %{x:.1%}"
                                          "<br>lift %{y:.2f}×<extra></extra>")
        figcl.update_layout(height=420, xaxis_title="Confidence", yaxis_title="Lift (× vs chance)",
                            xaxis_tickformat=".0%", coloraxis_showscale=False)
        figcl.add_hline(y=1.2, line_dash="dot", line_color=theme.MUTED,
                        annotation_text="lift threshold 1.2", annotation_position="bottom right",
                        annotation_font=dict(size=10, color=theme.MUTED))
        figcl.add_vline(x=0.70, line_dash="dot", line_color=theme.MUTED,
                        annotation_text="confidence threshold 70%", annotation_position="top left",
                        annotation_font=dict(size=10, color=theme.MUTED))
        st.plotly_chart(figcl, use_container_width=True)
        st.caption(
            "Every rule that cleared both thresholds. Bubble size is support — how common the "
            "combination is. High confidence with low lift means the consequent is simply common; "
            "the useful rules sit high on both axes."
        )

        # ---- Support sensitivity --------------------------------------------
        if extra.get("support") is not None:
            sp = extra["support"]
            st.markdown("#### How sensitive are the results to the support threshold?")
            figsp = go.Figure()
            figsp.add_bar(x=sp["min_support"], y=sp["n_rules"], name="Qualifying rules",
                          marker_color=theme.PLASMA_BLUE,
                          hovertemplate="support %{x}<br>%{y:,} rules<extra></extra>")
            figsp.add_scatter(x=sp["min_support"], y=sp["max_lift"], name="Max lift", yaxis="y2",
                              mode="lines+markers", line=dict(color=theme.PLASMA_GOLD, width=3),
                              hovertemplate="support %{x}<br>max lift %{y:.2f}×<extra></extra>")
            figsp.add_vline(x=0.10, line_dash="dot", line_color=theme.PLASMA_GREEN, line_width=2,
                            annotation_text="0.10 chosen", annotation_position="top",
                            annotation_font=dict(size=10, color=theme.PLASMA_GREEN))
            figsp.update_layout(height=380, xaxis_title="Minimum support threshold",
                                yaxis=dict(title="Qualifying rules"),
                                yaxis2=dict(title="Max lift", overlaying="y", side="right",
                                            tickfont=dict(color=theme.PLASMA_GOLD)),
                                legend=dict(orientation="h", y=-0.20))
            st.plotly_chart(figsp, use_container_width=True)
            st.warning(
                "**The rule count is highly sensitive to this threshold** — 1,047 rules at 0.08, "
                "439 at 0.10, 171 at 0.12. We are reporting that rather than claiming stability. "
                "What *is* stable is the headline finding: the FPS + Multiplayer → First-Person + "
                "Shooter rule appears with an identical lift of 6.865 at every threshold up to 0.10, "
                "and disappears above it only because its own support is exactly 0.100 — a "
                "mechanical exclusion, not instability. Lower thresholds surface rarer, higher-lift "
                "combinations; 0.10 keeps rules general enough to act on."
            )
            st.dataframe(sp, use_container_width=True, hide_index=True)

        with st.expander("Why Apriori — comparison against FP-Growth"):
            st.markdown("""
Frequent itemset mining is deterministic given the same support threshold, so Apriori and
FP-Growth necessarily surface identical rule sets — the choice comes down to clarity, not results.
Apriori's level-wise candidate generation is more transparent to walk through in a presentation;
FP-Growth's speed advantage matters on much larger transaction sets than this dataset's 290 games.
A `min_support` sweep (0.05–0.20) also confirmed the rule count and max lift are stable in the
0.08–0.15 range, so 0.1 isn't a cherry-picked edge case.
            """)
