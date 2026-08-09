"""
Visual theme for the Steam Games Analytics Dashboard.

Everything here is pure CSS + SVG — no video files, no external media, no JS.
That keeps the repo small, the page fast, and the deploy free of copyright
questions. All motion is disabled automatically for visitors who have
"reduce motion" turned on in their OS.
"""

import base64
import random
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------------------------------------------------------------- palette ----
# Named once here, used everywhere. Change a value and the whole dashboard moves.
VOID = "#05070f"        # deepest background
DEEP = "#0a1020"        # panel base
EDGE = "#1d2942"        # panel border
TEXT = "#e8ecf5"        # primary text
MUTED = "#8b97b0"       # captions, secondary text

PLASMA_GREEN = "#37e6a0"
PLASMA_BLUE = "#3aa8ff"
PLASMA_GOLD = "#f2c14e"

PASS = PLASMA_GREEN
FAIL = "#ff6b5b"

CHART_COLORWAY = [PLASMA_GREEN, PLASMA_BLUE, PLASMA_GOLD, "#c77dff", "#ff7a59", "#4dd4d4"]


# ---------------------------------------------------------------- starfield ----
def _starfield_layer(count, seed, max_radius, max_alpha):
    """Build one parallax layer of stars as CSS radial-gradients.

    Seeded, so the sky is identical on every rerun and every machine — a
    starfield that reshuffles whenever someone moves a slider reads as a bug.
    """
    rng = random.Random(seed)
    dots = []
    for _ in range(count):
        x = round(rng.uniform(0, 100), 2)
        y = round(rng.uniform(0, 100), 2)
        r = round(rng.uniform(0.6, max_radius), 2)
        a = round(rng.uniform(max_alpha * 0.35, max_alpha), 2)
        dots.append(
            f"radial-gradient({r}px {r}px at {x}% {y}%, rgba(255,255,255,{a}) 0%, rgba(255,255,255,0) 60%)"
        )
    return ",\n    ".join(dots)


def _background_css():
    far = _starfield_layer(38, seed=11, max_radius=1.1, max_alpha=0.55)
    mid = _starfield_layer(20, seed=22, max_radius=1.5, max_alpha=0.75)
    near = _starfield_layer(10, seed=33, max_radius=2.1, max_alpha=0.95)

    # Two coloured nebulae sit under the stars and never move — they anchor the
    # composition so the drifting layers read as depth rather than noise.
    nebula = (
        f"radial-gradient(900px 620px at 12% 8%, rgba(58,168,255,0.16) 0%, rgba(58,168,255,0) 70%),\n    "
        f"radial-gradient(780px 560px at 88% 82%, rgba(55,230,160,0.13) 0%, rgba(55,230,160,0) 70%),\n    "
        f"radial-gradient(600px 400px at 70% 12%, rgba(242,193,78,0.07) 0%, rgba(242,193,78,0) 70%)"
    )

    return f"""
.stApp {{
    background-color: {VOID};
    background-image:
    {near},
    {mid},
    {far},
    {nebula};
    background-size: 1400px 1400px, 900px 900px, 600px 600px, 100% 100%, 100% 100%, 100% 100%;
    background-repeat: repeat, repeat, repeat, no-repeat, no-repeat, no-repeat;
}}

"""


# ---------------------------------------------------------------- css ----
def inject_css():
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

{_background_css()}

/* Streamlit's own header bar would sit as an opaque strip over the sky. */
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbar"] {{ right: 1rem; }}

html, body, [class*="css"] {{
    font-family: 'Rajdhani', system-ui, sans-serif;
    color: {TEXT};
}}

h1, h2, h3, h4 {{
    font-family: 'Orbitron', system-ui, sans-serif !important;
    color: {TEXT} !important;
    letter-spacing: 0.04em;
}}

/* ---- hero ---- */
.hero {{
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 26px 0 14px 0;
}}

.logo-shell {{
    position: relative;
    width: 148px;
    height: 148px;
    display: grid;
    place-items: center;
}}

/* The signature element: a gradient ring that rotates around the mark.
   Drawn as a conic-gradient disc, then hollowed out with a mask. */
.halo {{
    position: absolute;
    inset: 0;
    border-radius: 50%;
    background: conic-gradient(
        from 0deg,
        {PLASMA_GREEN}, {PLASMA_BLUE}, {PLASMA_GOLD}, {PLASMA_GREEN}
    );
    -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 5px), #000 calc(100% - 5px));
    mask: radial-gradient(farthest-side, transparent calc(100% - 5px), #000 calc(100% - 5px));
    animation: spin 9s linear infinite;
    will-change: transform;
    filter: drop-shadow(0 0 14px rgba(55,230,160,0.45));
}}

.halo-soft {{
    position: absolute;
    inset: -22px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(58,168,255,0.22) 0%, rgba(58,168,255,0) 68%);
    animation: breathe 6s ease-in-out infinite;
    will-change: transform, opacity;
}}

@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
@keyframes breathe {{
    0%, 100% {{ opacity: 0.55; transform: scale(1); }}
    50%      {{ opacity: 1;    transform: scale(1.06); }}
}}

.logo-mark {{
    width: 96px;
    height: 96px;
    z-index: 1;
    filter: drop-shadow(0 0 12px rgba(55,230,160,0.55));
}}

.hero-title {{
    font-family: 'Orbitron', sans-serif;
    font-weight: 900;
    font-size: 2.05rem;
    letter-spacing: 0.13em;
    margin-top: 20px;
    background: linear-gradient(92deg, {PLASMA_GREEN}, {PLASMA_BLUE} 45%, {PLASMA_GOLD});
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}}

.hero-sub {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: {MUTED};
    margin-top: 6px;
    text-align: center;
}}

.hero-status {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    letter-spacing: 0.1em;
    margin-top: 14px;
    padding: 5px 16px;
    border-radius: 999px;
    border: 1px solid {EDGE};
    background: rgba(10,16,32,0.6);
    color: {MUTED};
}}
.hero-status b {{ color: {PLASMA_GREEN}; }}

/* ---- panels ---- */
.panel {{
    background: rgba(10,16,32,0.62);
    border: 1px solid {EDGE};
    border-radius: 14px;
    padding: 20px 24px;
}}

.target-row {{
    background: rgba(10,16,32,0.62);
    border: 1px solid {EDGE};
    border-left: 3px solid var(--accent, {PASS});
    border-radius: 8px;
    padding: 9px 15px;
    margin-bottom: 7px;
    font-size: 0.95rem;
}}

.ach-card {{
    background: rgba(10,16,32,0.62);
    border: 1px solid {EDGE};
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 12px;
    transition: border-color 0.25s ease, transform 0.25s ease;
}}
.ach-card:hover {{ border-color: {PLASMA_GREEN}; transform: translateY(-2px); }}
.ach-eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    color: {PLASMA_GOLD};
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-top: 6px;
}}
.ach-title {{ font-family: 'Orbitron', sans-serif; font-size: 1.02rem; margin-top: 3px; }}
.ach-value {{ color: {PLASMA_GREEN}; font-weight: 700; font-size: 1.35rem; }}
.ach-sub {{ color: {MUTED}; font-size: 0.86rem; }}

/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background: rgba(10,16,32,0.55);
    border: 1px solid {EDGE};
    border-radius: 12px;
    padding: 6px;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'Orbitron', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    border-radius: 8px;
    color: {MUTED};
}}
.stTabs [aria-selected="true"] {{
    background: rgba(55,230,160,0.12);
    color: {PLASMA_GREEN} !important;
    box-shadow: inset 0 0 0 1px rgba(55,230,160,0.35);
}}

/* ---- metrics & data ---- */
[data-testid="stMetric"] {{
    background: rgba(10,16,32,0.62);
    border: 1px solid {EDGE};
    border-radius: 12px;
    padding: 14px 16px;
}}
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace;
    color: {PLASMA_GREEN};
}}
[data-testid="stMetricLabel"] {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {MUTED};
}}

hr {{ border-color: {EDGE}; }}

/* Accessibility floor: honour the OS "reduce motion" setting. */
@media (prefers-reduced-motion: reduce) {{
    .halo, .halo-soft {{ animation: none !important; }}
    .ach-card {{ transition: none; }}
}}

/* Mobile: the hero shrinks rather than overflowing. */
@media (max-width: 640px) {{
    .logo-shell {{ width: 108px; height: 108px; }}
    .logo-mark {{ width: 70px; height: 70px; }}
    .hero-title {{ font-size: 1.4rem; letter-spacing: 0.08em; }}
}}
</style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- logo ----
_ASSETS = Path(__file__).parent / "assets"


def _logo_markup():
    """Use assets/logo.png if the user has dropped one in, else the bundled SVG.

    Kept swappable on purpose: drop your own logo.png into assets/ and it
    replaces the mark with no code change.
    """
    png = _ASSETS / "logo.png"
    if png.exists():
        b64 = base64.b64encode(png.read_bytes()).decode()
        return f'<img class="logo-mark" src="data:image/png;base64,{b64}" alt="Dashboard logo">'

    svg = _ASSETS / "logo.svg"
    if svg.exists():
        b64 = base64.b64encode(svg.read_bytes()).decode()
        return f'<img class="logo-mark" src="data:image/svg+xml;base64,{b64}" alt="Dashboard logo">'

    return '<div class="logo-mark" style="font-size:3.4rem;line-height:96px;">🎮</div>'


def hero(title, subtitle, status_html=""):
    st.markdown(
        f"""
<div class="hero">
  <div class="logo-shell">
    <div class="halo-soft"></div>
    <div class="halo"></div>
    {_logo_markup()}
  </div>
  <div class="hero-title">{title}</div>
  <div class="hero-sub">{subtitle}</div>
  {f'<div class="hero-status">{status_html}</div>' if status_html else ''}
</div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- plotly ----
def register_plotly_template():
    """Register and default to a transparent, space-palette Plotly template.

    Registering it as the default means every existing px./go. call in tabs/
    picks it up with no changes at the call site.
    """
    pio.templates["space"] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Rajdhani, sans-serif", color=TEXT, size=13),
            title=dict(font=dict(family="Orbitron, sans-serif", size=15, color=TEXT)),
            colorway=CHART_COLORWAY,
            xaxis=dict(gridcolor="rgba(29,41,66,0.85)", zerolinecolor="rgba(29,41,66,0.85)",
                       linecolor=EDGE, tickfont=dict(color=MUTED)),
            yaxis=dict(gridcolor="rgba(29,41,66,0.85)", zerolinecolor="rgba(29,41,66,0.85)",
                       linecolor=EDGE, tickfont=dict(color=MUTED)),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
            margin=dict(t=52, b=42, l=42, r=24),
        )
    )
    pio.templates.default = "space"


def target_row(label, value_str, passed):
    """One pass/fail line for a stated model target."""
    accent = PASS if passed else FAIL
    icon = "PASS" if passed else "MISS"
    st.markdown(
        f'<div class="target-row" style="--accent:{accent};">'
        f'<span style="font-family:JetBrains Mono,monospace;font-size:0.7rem;'
        f'letter-spacing:0.12em;color:{accent};">{icon}</span>&nbsp;&nbsp;'
        f'<b>{label}:</b> {value_str}</div>',
        unsafe_allow_html=True,
    )
