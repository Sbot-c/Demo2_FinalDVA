# Steam Games Analytics Dashboard

Six-tab Streamlit dashboard — Descriptive, Diagnostic, Predictive, Prescriptive, What-If Simulator,
Fun Facts — built on a validated notebook analysis of 290 Steam games and ~136k sampled reviews.

The dashboard never touches the raw ~475 MB reviews CSV at runtime. It reads pre-computed artifacts
from `dashboard_exports/`, which is what keeps it fast enough to demo live and small enough to host.

## Deploy to Streamlit Community Cloud

1. Push this folder to a new GitHub repo (see "What's in the repo" below — the raw CSV must stay out).
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **New app**.
3. Pick the repo, set the branch, set the main file to `app.py`, click **Deploy**.

First build takes a few minutes while dependencies install. After that, every `git push` redeploys
automatically.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at <http://localhost:8501>.

## What's in the repo

```
steam-dashboard/
├── app.py                    # entrypoint: hero header + tab layout
├── theme.py                  # palette, starfield CSS, logo hero, Plotly template
├── assets/logo.svg           # logo mark — drop in a logo.png here to replace it
├── tabs/                     # one module per tab
├── dashboard_exports/        # pre-computed data + trained models (1.7 MB)
├── generate_exports.py       # regenerates dashboard_exports/ from the raw CSVs
├── games_description.csv     # source data (1 MB)
├── games_ranking.csv         # source data (28 KB)
├── requirements.txt
├── .streamlit/config.toml    # dark base theme
└── .gitignore
```

Total: about 2.8 MB.

**`steam_game_reviews.csv` is deliberately not in this repo.** It's ~475 MB, and GitHub rejects any
single file over 100 MB, so a push containing it will fail. Keep it on your machine next to the repo
folder; only `generate_exports.py` needs it.

## Swapping the logo

`theme.py` looks for `assets/logo.png` first and falls back to `assets/logo.svg`. Drop your own PNG
in as `assets/logo.png` and it appears inside the rotating halo with no code change. Square images
around 512×512 with a transparent background work best.

## The theme

Dark starry background, a glowing central mark ringed by a rotating green/blue/gold gradient halo,
translucent panels with neon edges, Orbitron/Rajdhani/JetBrains Mono type.

It is built entirely from CSS and inline SVG — no video files, no image assets, no JavaScript. That
matters for three reasons: the repo stays tiny, there's nothing to buffer or repaint on a page that
re-runs on every widget interaction, and there's no third-party media with licensing attached. The
starfield is seeded, so the sky is identical on every rerun instead of reshuffling when someone
moves a slider. All motion is disabled automatically for visitors with "reduce motion" enabled.

To retheme, change the hex values at the top of `theme.py` — the palette, the charts, and the
panels all read from there.

## Regenerating the data

```bash
python generate_exports.py
```

Reads the three source CSVs and rewrites `dashboard_exports/`. Update the `UPLOAD_DIR` constant at
the top of the script to point at wherever you keep `steam_game_reviews.csv`.

**Pin note:** the models in `dashboard_exports/*.pkl` were pickled with scikit-learn 1.8.0, so
`requirements.txt` pins that exact version. If you regenerate the exports under a different
scikit-learn, update the pin to match — loading a pickle across versions raises
`InconsistentVersionWarning` and can silently misbehave.
