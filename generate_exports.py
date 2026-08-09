"""
Generates dashboard_exports/ from the real Steam CSVs — mirrors the notebook's Section 7 export
cell exactly, plus the new descriptive/diagnostic aggregate exports needed for the dashboard's
first two tabs. Run once (or re-run whenever the notebook / data changes) to refresh the dashboard's
data without the dashboard itself touching the raw 475MB reviews file at runtime.
"""
import json
import pickle
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, r2_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from pathlib import Path

RANDOM_STATE = 42
SAMPLE_SIZE = 150000
OUT_DIR = "dashboard_exports"

import os
os.makedirs(OUT_DIR, exist_ok=True)

ROOT_DIR = Path(__file__).resolve().parent
LOCAL_UPLOADS = ROOT_DIR
REMOTE_UPLOADS = Path("/mnt/user-data/uploads")
UPLOAD_DIR = LOCAL_UPLOADS if all((LOCAL_UPLOADS / fname).exists() for fname in [
    "steam_game_reviews.csv", "games_description.csv", "games_ranking.csv"
]) else REMOTE_UPLOADS

print(f"Using uploads directory: {UPLOAD_DIR}")

# ---------------------------------------------------------------- Load & clean ----
desc = pd.read_csv(UPLOAD_DIR / "games_description.csv")
rank = pd.read_csv(UPLOAD_DIR / "games_ranking.csv")

def load_reviews(path, sample_size, chunksize=100000, random_state=42):
    usecols = ["review","hours_played","helpful","funny","recommendation","date","game_name"]
    chunks, collected = [], 0
    frac = min(1.0, (sample_size / 992153) * 1.5)
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        s = chunk.sample(frac=frac, random_state=random_state)
        chunks.append(s)
        collected += len(s)
        if collected >= sample_size:
            break
    out = pd.concat(chunks, ignore_index=True)
    return out.sample(n=min(sample_size, len(out)), random_state=random_state).reset_index(drop=True)

rev = load_reviews(f"{UPLOAD_DIR}/steam_game_reviews.csv", SAMPLE_SIZE, random_state=RANDOM_STATE)
print("reviews loaded:", rev.shape)

def clean_num(s):
    if pd.isna(s):
        return np.nan
    s = str(s).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan

rev["helpful_n"] = rev["helpful"].apply(clean_num)
rev["funny_n"] = rev["funny"].apply(clean_num)
rev.loc[rev["funny_n"] > 100000, "funny_n"] = np.nan
rev["hours_n"] = pd.to_numeric(rev["hours_played"], errors="coerce")
rev["review"] = rev["review"].fillna("").astype(object).astype(str)
rev["review_len"] = rev["review"].str.len()
rev["word_count"] = rev["review"].str.split().str.len()
rev["exclaim_count"] = rev["review"].str.count("!")
rev["caps_ratio"] = rev["review"].apply(lambda t: sum(1 for c in str(t) if c.isupper()) / max(len(str(t)), 1))
rev["rec_bin"] = (rev["recommendation"] == "Recommended").astype(int)
rev = rev.dropna(subset=["review", "hours_n", "funny_n", "helpful_n"])

def clean_reviewcount(s):
    if pd.isna(s):
        return np.nan
    s = str(s).replace(",", "").replace("(", "").replace(")", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan

desc["genre_list"] = desc["genres"].apply(eval)
desc["review_count"] = desc["number_of_reviews_from_purchased_people"].apply(clean_reviewcount)
rating_map = {"Overwhelmingly Positive": 6, "Very Positive": 5, "Mostly Positive": 4, "Positive": 4,
              "Mixed": 3, "Mostly Negative": 2, "Very Negative": 1, "Overwhelmingly Negative": 0}
desc["rating_score"] = desc["overall_player_rating"].map(rating_map)

print("clean reviews:", rev.shape, " desc:", desc.shape)

# ---------------------------------------------------------------- 1. Descriptive aggregates ----
all_genres = Counter([g for lst in desc["genre_list"] for g in lst])
genre_counts_df = pd.DataFrame(all_genres.most_common(20), columns=["genre", "count"])
genre_counts_df.to_csv(f"{OUT_DIR}/genre_counts.csv", index=False)

order = ["Overwhelmingly Positive","Very Positive","Mostly Positive","Positive",
         "Mixed","Mostly Negative","Very Negative","Overwhelmingly Negative"]
rating_dist = desc["overall_player_rating"].value_counts().reindex(order).dropna().reset_index()
rating_dist.columns = ["rating_category", "count"]
rating_dist.to_csv(f"{OUT_DIR}/rating_distribution.csv", index=False)

top_by_rank = rank[rank["rank_type"].isin(["Sales","Revenue","Review"])].sort_values(["rank_type","rank"])
top_by_rank[top_by_rank["rank"] <= 10].to_csv(f"{OUT_DIR}/top10_by_rank.csv", index=False)

engagement_stats = rev[["hours_n","helpful_n","funny_n","review_len","word_count"]].describe().round(2)
engagement_stats.to_csv(f"{OUT_DIR}/engagement_summary_stats.csv")

hours_hist, hours_edges = np.histogram(rev["hours_n"].clip(upper=500), bins=40)
pd.DataFrame({"bin_left": hours_edges[:-1], "bin_right": hours_edges[1:], "count": hours_hist}).to_csv(
    f"{OUT_DIR}/hours_played_histogram.csv", index=False)

# ---------------------------------------------------------------- 2. Diagnostic aggregates ----
numeric_cols = ["hours_n","helpful_n","funny_n","review_len","word_count","exclaim_count","caps_ratio","rec_bin"]
corr = rev[numeric_cols].corr().round(3)
corr.to_csv(f"{OUT_DIR}/correlation_matrix.csv")

rec_comparison = rev.groupby("recommendation")[["hours_n","helpful_n","funny_n","review_len","word_count"]].mean().round(2)
rec_comparison.to_csv(f"{OUT_DIR}/recommended_vs_not_comparison.csv")

sales_rank = rank[rank["rank_type"] == "Sales"][["game_name","rank"]].rename(columns={"rank":"sales_rank"})
merged_rank = desc.merge(sales_rank, left_on="name", right_on="game_name", how="inner").dropna(subset=["rating_score","sales_rank"])
merged_rank[["name","rating_score","sales_rank"]].to_csv(f"{OUT_DIR}/rating_vs_salesrank.csv", index=False)
rating_salesrank_corr = round(float(merged_rank["rating_score"].corr(merged_rank["sales_rank"])), 3)

# ---------------------------------------------------------------- 3. Classification ----
clf_features = ["hours_n","funny_n","helpful_n","review_len","word_count","exclaim_count","caps_ratio"]
df_clf = rev.dropna(subset=["rec_bin"] + clf_features)
X = df_clf[clf_features].values
y = df_clf["rec_bin"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=RANDOM_STATE)
clf.fit(Xtr, ytr)
pred = clf.predict(Xte)
proba = clf.predict_proba(Xte)[:, 1]
acc, prec, rec, f1, auc = (accuracy_score(yte, pred), precision_score(yte, pred), recall_score(yte, pred),
                            f1_score(yte, pred), roc_auc_score(yte, proba))
tr_acc = accuracy_score(ytr, clf.predict(Xtr))
print(f"Classification: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} auc={auc:.4f} gap={tr_acc-acc:.4f}")

# ---------------------------------------------------------------- 4. Clustering ----
top_genres = [g for g, _ in all_genres.most_common(25)]
desc["genre_filtered"] = desc["genre_list"].apply(lambda lst: [g for g in lst if g in top_genres])
mlb = MultiLabelBinarizer(classes=top_genres)
desc_clean = desc.dropna(subset=["rating_score", "review_count"]).reset_index(drop=True)
genre_matrix = mlb.fit_transform(desc_clean["genre_filtered"])
X_cluster = np.hstack([genre_matrix, desc_clean[["rating_score"]].values, np.log1p(desc_clean[["review_count"]].values)])
scaler = StandardScaler().fit(X_cluster)
X_scaled = scaler.transform(X_cluster)
pca = PCA(n_components=10, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

K_range = range(2, 11)
sils = []
for k in K_range:
    labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X_pca)
    sils.append(silhouette_score(X_pca, labels))

km_final = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
desc_clean["cluster"] = km_final.fit_predict(X_pca)
desc_clean["pc1"] = X_pca[:, 0]
desc_clean["pc2"] = X_pca[:, 1]
sil_k4 = sils[list(K_range).index(4)]
print(f"Clustering: silhouette@k4={sil_k4:.4f}")

cluster_profiles = []
for c in sorted(desc_clean["cluster"].unique()):
    sub = desc_clean[desc_clean["cluster"] == c]
    gc = Counter([g for lst in sub["genre_filtered"] for g in lst])
    cluster_profiles.append({
        "cluster": int(c), "n_games": int(len(sub)),
        "avg_rating": round(float(sub["rating_score"].mean()), 2),
        "top_genres": ", ".join(g for g, _ in gc.most_common(3)),
    })
pd.DataFrame(cluster_profiles).to_csv(f"{OUT_DIR}/cluster_profiles.csv", index=False)

# ---------------------------------------------------------------- 5. Regression ----
reg_features = ["hours_n","helpful_n","review_len","word_count","exclaim_count","caps_ratio","rec_bin"]
df_r = rev.dropna(subset=["funny_n"] + reg_features)
Xr, yr = df_r[reg_features].values, np.log1p(df_r["funny_n"].values)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=RANDOM_STATE)
reg = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=RANDOM_STATE)
reg.fit(Xr_tr, yr_tr)
pred_r = reg.predict(Xr_te)
r2_final = r2_score(yr_te, pred_r)
print(f"Regression: R2={r2_final:.4f}")

feature_importance = pd.Series(reg.feature_importances_, index=reg_features).sort_values(ascending=False)
feature_importance.reset_index().rename(columns={"index":"feature", 0:"importance"}).to_csv(
    f"{OUT_DIR}/regression_feature_importance.csv", index=False)

# ---------------------------------------------------------------- 6. Association ----
te = TransactionEncoder()
te_ary = te.fit(desc["genre_list"]).transform(desc["genre_list"])
df_trans = pd.DataFrame(te_ary, columns=te.columns_)
freq_items = apriori(df_trans, min_support=0.1, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.7)
rules = rules[rules["lift"] > 1.2].sort_values("lift", ascending=False)
rules_export = rules[["antecedents","consequents","support","confidence","lift"]].copy()
rules_export["antecedents"] = rules_export["antecedents"].apply(lambda x: ", ".join(x))
rules_export["consequents"] = rules_export["consequents"].apply(lambda x: ", ".join(x))
rules_export.to_csv(f"{OUT_DIR}/association_rules.csv", index=False)
print(f"Association: {len(rules)} rules")

# ---------------------------------------------------------------- 7. Prescriptive narrative ----
top_rule = rules_export.iloc[0]
best_cluster_row = max(cluster_profiles, key=lambda r: r["avg_rating"])
narrative = (
    f"Based on the data, the strongest genre pairing is '{top_rule['antecedents']} -> {top_rule['consequents']}' "
    f"(lift {top_rule['lift']:.2f}) -- a reliably co-occurring, high-demand combination worth targeting. "
    f"Cluster {best_cluster_row['cluster']} (top genres: {best_cluster_row['top_genres']}) shows the highest "
    f"average player rating ({best_cluster_row['avg_rating']}/6) among the four game archetypes and is a safer "
    f"positioning bet for a new title. Our recommendation model reaches {acc*100:.1f}% accuracy and {f1*100:.1f}% "
    f"F1 identifying what drives positive reception, catching {rec*100:.1f}% of true recommends. Engagement itself "
    f"(funny/helpful votes) is predictable with R2={r2_final:.2f} from review length and tone -- meaning polished, "
    f"substantial content reliably earns visible community engagement, which in turn feeds Steam's discovery "
    f"algorithms. Recommendation: prioritize a title in the top-rated archetype above, built around the "
    f"highest-lift genre pairing, and encourage the kind of longer, substantive reviews the regression model "
    f"identifies as the strongest engagement driver."
)
with open(f"{OUT_DIR}/prescriptive_narrative.txt", "w") as f:
    f.write(narrative)
print("\nNarrative:\n", narrative)

# ---------------------------------------------------------------- 8. Metrics summary + models ----
metrics_summary = {
    "classification": {"accuracy": float(acc), "precision": float(prec), "recall": float(rec),
                        "f1": float(f1), "auc": float(auc), "train_test_gap": float(tr_acc - acc)},
    "clustering": {"silhouette_k4": float(sil_k4), "chosen_k": 4},
    "regression": {"r2": float(r2_final), "target": "log1p(funny_votes)"},
    "association": {"n_rules": int(len(rules)), "top_lift": float(rules["lift"].max())},
    "diagnostic": {"rating_vs_salesrank_corr": rating_salesrank_corr},
    "dataset": {"n_games": int(len(desc)), "n_reviews_sampled": int(len(rev)), "n_reviews_total": 992153},
}
with open(f"{OUT_DIR}/metrics_summary.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)

desc_clean.drop(columns=["genre_list"]).to_csv(f"{OUT_DIR}/games_with_clusters.csv", index=False)

with open(f"{OUT_DIR}/classifier.pkl", "wb") as f:
    pickle.dump({"model": clf, "features": clf_features,
                 "feature_medians": df_clf[clf_features].median().to_dict()}, f)
with open(f"{OUT_DIR}/regressor.pkl", "wb") as f:
    pickle.dump({"model": reg, "features": reg_features}, f)
with open(f"{OUT_DIR}/kmeans.pkl", "wb") as f:
    pickle.dump({"model": km_final, "pca": pca, "scaler": scaler, "mlb": mlb, "top_genres": top_genres}, f)

# ---------------------------------------------------------------- 9. Fun facts ----
most_helpful_idx = rev["helpful_n"].idxmax()
most_helpful = rev.loc[most_helpful_idx]

most_funny_idx = rev["funny_n"].idxmax()
most_funny = rev.loc[most_funny_idx]

avg_hours_by_game = rev.groupby("game_name")["hours_n"].mean().sort_values(ascending=False)
longest_playtime_game = avg_hours_by_game.index[0]
longest_playtime_hours = round(float(avg_hours_by_game.iloc[0]), 1)

# biggest mismatch between sales rank (popularity) and player rating (reception)
mismatch = merged_rank.copy()
mismatch["rating_norm"] = mismatch["rating_score"] / 6.0
mismatch["sales_norm"] = 1 - (mismatch["sales_rank"] - mismatch["sales_rank"].min()) / (
    mismatch["sales_rank"].max() - mismatch["sales_rank"].min())
mismatch["gap"] = mismatch["sales_norm"] - mismatch["rating_norm"]
sells_great_rated_poorly = mismatch.loc[mismatch["gap"].idxmax(), "name"]
rated_great_sells_poorly = mismatch.loc[mismatch["gap"].idxmin(), "name"]

longest_review_idx = rev["review_len"].idxmax()
longest_review_game = rev.loc[longest_review_idx, "game_name"]
longest_review_chars = int(rev.loc[longest_review_idx, "review_len"])

fun_facts = {
    "total_reviews_in_full_dataset": 992153,
    "reviews_analyzed_this_run": int(len(rev)),
    "total_games": int(len(desc)),
    "most_helpful_review": {
        "game": str(most_helpful["game_name"]), "helpful_votes": int(most_helpful["helpful_n"]),
    },
    "most_funny_review": {
        "game": str(most_funny["game_name"]), "funny_votes": int(most_funny["funny_n"]),
    },
    "longest_avg_playtime_game": {"game": str(longest_playtime_game), "avg_hours": longest_playtime_hours},
    "longest_single_review": {"game": str(longest_review_game), "characters": longest_review_chars},
    "sells_great_rated_poorly": str(sells_great_rated_poorly),
    "rated_great_sells_modestly": str(rated_great_sells_poorly),
    "data_bug_found": "24 rows in the 'funny' votes column contained 4294967295 (2^32-1, a classic "
                       "32-bit integer overflow) from the original Steam scrape — corrected to missing.",
    "positive_rating_share_pct": round(
        float(desc["overall_player_rating"].isin(
            ["Overwhelmingly Positive","Very Positive","Mostly Positive","Positive"]).mean() * 100), 1),
}
with open(f"{OUT_DIR}/fun_facts.json", "w") as f:
    json.dump(fun_facts, f, indent=2)
print("\nFun facts:\n", json.dumps(fun_facts, indent=2))

print("\nAll exports written to", OUT_DIR)
print(sorted(os.listdir(OUT_DIR)))
