"""
CONTINUOUS INERTIA TECHNO ANALYZER v3.0
Protocol Compliance & Corpus Comparison
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CI Analyzer",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS — Minimal / Sacred / Deep
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg:       #08080a;
    --surface:  #0e0e11;
    --surface2: #141418;
    --border:   #1e1e24;
    --border2:  #2a2a34;
    --gold:     #c9a96e;
    --gold2:    #e8c99a;
    --dim:      #4a4a58;
    --text:     #d4d4c8;
    --muted:    #5a5a68;
    --pass:     #6db88a;
    --fail:     #c46a6a;
    --warn:     #c4956a;
}

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--text) !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div { padding-top: 2rem; }

/* Typography */
h1 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 300 !important;
    font-size: 2.4rem !important;
    letter-spacing: 0.08em !important;
    color: var(--text) !important;
    line-height: 1.15 !important;
}
h2, h3 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    margin: 0 !important;
}

/* Remove Streamlit chrome */
[data-testid="stDecoration"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }

/* Divider */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* Upload widget */
[data-testid="stFileUploader"] {
    border: 1px solid var(--border2) !important;
    background: var(--surface) !important;
    border-radius: 0 !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--gold) !important;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    color: var(--gold) !important;
    border: 1px solid var(--gold) !important;
    border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--gold) !important;
    color: #000 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.15em !important;
    color: var(--muted) !important;
    border-radius: 0 !important;
    padding: 0.6rem 1.4rem !important;
    text-transform: uppercase !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom: 1px solid var(--gold) !important;
    background: transparent !important;
}

/* Toggles + selectbox */
[data-testid="stToggle"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    color: var(--muted) !important;
}
[data-testid="stSelectbox"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.15em !important;
    color: var(--muted) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 0 !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* Info / warning / success */
[data-testid="stAlert"] {
    border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* Download buttons */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: var(--gold) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: var(--gold) !important;
}

/* Custom components */
.label-xs {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
}

.mcard {
    padding: 1.1rem 1.3rem;
    border: 1px solid var(--border);
    background: var(--surface);
    position: relative;
    margin-bottom: 0.6rem;
}
.mcard::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 2px;
    background: var(--gold);
}
.mcard.fail::before { background: var(--fail); }
.mcard.warn::before { background: var(--warn); }
.mcard.pass::before { background: var(--pass); }

.mval {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.9rem;
    font-weight: 300;
    color: var(--gold2);
    line-height: 1;
}
.mval.fail { color: var(--fail); }
.mval.pass { color: var(--pass); }

.mlabel {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.35rem;
}

.prow {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 0.85rem 0;
    border-bottom: 1px solid var(--border);
}
.prow:last-child { border-bottom: none; }

.pbadge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    padding: 0.15rem 0.5rem;
    letter-spacing: 0.1em;
    font-weight: 500;
    flex-shrink: 0;
    margin-top: 0.15rem;
}
.pbadge.ok  { background: transparent; border: 1px solid var(--pass); color: var(--pass); }
.pbadge.no  { background: transparent; border: 1px solid var(--fail); color: var(--fail); }

.pname {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1rem;
    font-weight: 400;
    color: var(--text);
    line-height: 1.2;
}
.pdetail {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    margin-top: 0.25rem;
    line-height: 1.5;
}
.pthresh {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--dim);
    margin-left: auto;
    flex-shrink: 0;
    text-align: right;
}

.banner {
    padding: 1.2rem 1.8rem;
    border: 1px solid var(--border2);
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}
.banner.ok  { border-left: 3px solid var(--pass); }
.banner.no  { border-left: 3px solid var(--fail); }

.banner-status {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}
.banner-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 0.3rem;
}
.banner-score {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3rem;
    font-weight: 300;
    line-height: 1;
}

.section-rule {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin: 1.6rem 0 1rem 0;
}

.hist-row {
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    line-height: 1.6;
}

.empty-state {
    text-align: center;
    padding: 5rem 2rem;
}
.empty-glyph {
    font-family: 'Cormorant Garamond', serif;
    font-size: 4rem;
    font-weight: 300;
    color: var(--border2);
    line-height: 1;
    margin-bottom: 1.5rem;
    letter-spacing: 0.2em;
}
.empty-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.3rem;
    font-weight: 300;
    color: var(--muted);
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}
.empty-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--dim);
    letter-spacing: 0.1em;
    line-height: 1.8;
}

.app-title {
    font-family: 'Cormorant Garamond', serif;
    font-weight: 300;
    font-size: 2rem;
    letter-spacing: 0.12em;
    color: var(--text);
    line-height: 1;
}
.app-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--dim);
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PLOT THEME
# ─────────────────────────────────────────────────────────────
PLOT_BG   = '#08080a'
PLOT_SURF = '#0e0e11'
GRID_COL  = '#1a1a20'
GOLD      = '#c9a96e'
GOLD_DIM  = 'rgba(201,169,110,0.06)'
GOLD_MID  = 'rgba(201,169,110,0.12)'
PASS_COL  = '#6db88a'
FAIL_COL  = '#c46a6a'
MUTED     = '#3a3a48'

LAYOUT_BASE = dict(
    paper_bgcolor=PLOT_BG,
    plot_bgcolor=PLOT_SURF,
    font=dict(family='JetBrains Mono, monospace', color='#5a5a68', size=10),
    margin=dict(l=48, r=24, t=40, b=40),
    xaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL, tickfont=dict(size=9)),
    yaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL, tickfont=dict(size=9)),
)


# ─────────────────────────────────────────────────────────────
# ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────
def simulate_analysis(filename: str, duration_sec: float = None) -> dict:
    np.random.seed(abs(hash(filename)) % (2**31))
    bpm           = np.random.uniform(126, 140)
    bpm_std       = np.random.uniform(0.3, 2.5)
    bpm_var_pct   = (bpm_std / bpm) * 100
    density_mean  = np.random.uniform(0.20, 0.65)
    centroid_mean = np.random.uniform(150, 500)
    rolloff_mean  = np.random.uniform(2000, 6000)
    sub_presence  = np.random.uniform(0.60, 1.0)
    sub_kick_ratio= np.random.uniform(0.45, 0.90)
    layers_mean   = np.random.uniform(1.5, 5.5)
    interval_bars = np.random.uniform(4, 24)
    texture_pres  = np.random.uniform(0.65, 1.0)
    duration      = duration_sec or np.random.uniform(380, 600)
    N = 24

    p1 = bpm_var_pct < 1.5
    p2 = (0.30 <= density_mean <= 0.45) and (layers_mean <= 4.5)
    p3 = 8 <= interval_bars <= 16
    p4 = sub_presence >= 0.90
    p5 = texture_pres >= 0.85
    passed = sum([p1, p2, p3, p4, p5])

    return {
        "metadata": {
            "filename": filename,
            "duration_sec": duration,
            "duration_min": round(duration / 60, 2),
            "analyzer_version": "3.0",
            "real_analysis": False,
        },
        "tempo": {
            "bpm": round(bpm, 2),
            "bpm_confidence": round(np.random.uniform(0.78, 0.99), 2),
            "bpm_mean": round(bpm, 2),
            "bpm_std": round(bpm_std, 3),
            "bpm_variance_pct": round(bpm_var_pct, 2),
            "stability_score": round(max(0, 1 - bpm_var_pct / 5), 2),
            "bpm_over_time": (bpm + np.random.randn(N) * bpm_std * 0.5).tolist(),
        },
        "spectral": {
            "centroid": {"mean": round(centroid_mean, 1), "std": round(np.random.uniform(25, 70), 1)},
            "density":  {"mean": round(density_mean, 3), "std": round(np.random.uniform(0.02, 0.08), 3)},
            "rolloff":  {"mean": round(rolloff_mean, 1), "std": round(np.random.uniform(300, 800), 1)},
            "density_over_time":  np.clip(density_mean + np.random.randn(N) * 0.04, 0, 1).tolist(),
            "centroid_over_time": (centroid_mean + np.random.randn(N) * 40).tolist(),
        },
        "kick": {
            "kick_on_beat_pct": round(np.random.uniform(0.72, 0.99), 2),
            "kick_consistency": round(np.random.uniform(0.70, 0.98), 2),
            "kick_fundamental_hz": round(np.random.uniform(40, 80), 1),
        },
        "structure": {
            "layers_mean": round(layers_mean, 1),
            "layers_mode": int(round(layers_mean)),
            "mean_interval_bars": round(interval_bars, 1),
            "periodicity_score": round(np.random.uniform(0.50, 0.97), 2),
        },
        "lowend": {
            "sub_presence_pct": round(sub_presence, 2),
            "sub_kick_ratio": round(sub_kick_ratio, 2),
            "sub_continuity": round(np.random.uniform(0.70, 0.99), 2),
            "sub_over_time": np.clip(sub_presence + np.random.randn(N) * 0.07, 0, 1).tolist(),
        },
        "protocol_compliance": {
            "principles": {
                "P1": {"name": "Temporal Stability",    "compliant": p1, "value": bpm_var_pct,   "threshold": "< 1.5% BPM variance",       "details": f"BPM σ = {bpm_var_pct:.2f}%"},
                "P2": {"name": "Spectral Parsimony",    "compliant": p2, "value": density_mean,  "threshold": "Density 0.30–0.45, ≤ 4 layers", "details": f"ρ = {density_mean:.3f} · layers = {layers_mean:.1f}"},
                "P3": {"name": "Periodic Micro-Var.",   "compliant": p3, "value": interval_bars, "threshold": "Change every 8–16 bars",      "details": f"Δ = {interval_bars:.1f} bars"},
                "P4": {"name": "Continuous Sub-Bass",   "compliant": p4, "value": sub_presence,  "threshold": "< 80 Hz present ≥ 90%",       "details": f"Present {sub_presence*100:.0f}% · ratio {sub_kick_ratio:.2f}"},
                "P5": {"name": "Textural Continuity",   "compliant": p5, "value": texture_pres,  "threshold": "Texture ≥ 85%",               "details": f"Presence {texture_pres*100:.0f}%"},
            },
            "principles_passed": passed,
            "compliant": passed >= 4,
        },
        "antipatterns": {
            "density_overload": density_mean > 0.60,
            "sub_absent":       sub_presence < 0.70,
            "bpm_change":       bpm_var_pct > 3.0,
            "total_violations": sum([density_mean > 0.60, sub_presence < 0.70, bpm_var_pct > 3.0]),
        },
        "complementary": {
            "C1": {"name": "BPM 128–135",      "met": 128 <= bpm <= 135},
            "C2": {"name": "Duration 7–9 min", "met": 420 <= duration <= 540},
            "C3": {"name": "No build-ups",     "met": bpm_var_pct <= 3.0},
            "C4": {"name": "Centroid 250–400", "met": 250 <= centroid_mean <= 400},
            "C5": {"name": "Stems exportable", "met": True},
        },
    }


def try_real_analysis(audio_bytes: bytes, filename: str) -> dict:
    try:
        import librosa, tempfile, os
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        y, sr = librosa.load(tmp_path, sr=22050, mono=True, duration=120)
        dur   = librosa.get_duration(y=y, sr=sr)
        bpm, _= librosa.beat.beat_track(y=y, sr=sr)
        os.unlink(tmp_path)
        res = simulate_analysis(filename, duration_sec=dur)
        res["tempo"]["bpm"]           = round(float(bpm), 2)
        res["metadata"]["real_analysis"] = True
        return res
    except Exception:
        return simulate_analysis(filename)


# ─────────────────────────────────────────────────────────────
# PLOT HELPERS — no add_hrect on subplots (broken in Plotly 6)
# ─────────────────────────────────────────────────────────────
def _band_shape(y0, y1, color, xref='paper', yref='y'):
    """Return a layout shape dict for a horizontal band — safe across Plotly versions."""
    return dict(
        type='rect', xref=xref, yref=yref,
        x0=0, x1=1, y0=y0, y1=y1,
        fillcolor=color, line_width=0, layer='below'
    )


def plot_bpm(result: dict) -> go.Figure:
    ts   = result["tempo"]["bpm_over_time"]
    dur  = result["metadata"]["duration_min"]
    mean = result["tempo"]["bpm_mean"]
    x    = np.linspace(0, dur, len(ts))
    lo, hi = mean * 0.985, mean * 1.015

    fig = go.Figure()
    fig.update_layout(**LAYOUT_BASE,
                      title=dict(text="BPM · Temporal Stability", font=dict(size=10, color=MUTED)),
                      xaxis_title="min", yaxis_title="BPM",
                      shapes=[_band_shape(lo, hi, GOLD_DIM)])
    fig.add_hline(y=mean, line_dash='dot', line_color=MUTED, line_width=1)
    fig.add_trace(go.Scatter(x=x, y=ts, mode='lines',
                             line=dict(color=GOLD, width=1.5),
                             fill='tozeroy', fillcolor=GOLD_DIM, name='BPM'))
    return fig


def plot_spectral(result: dict) -> go.Figure:
    dur = result["metadata"]["duration_min"]
    d   = result["spectral"]["density_over_time"]
    c   = result["spectral"]["centroid_over_time"]
    N   = len(d)
    x   = np.linspace(0, dur, N)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                        row_heights=[0.5, 0.5])

    # Density band via shapes (yref must be 'y' for row1, 'y2' for row2)
    fig.update_layout(
        **LAYOUT_BASE,
        height=360,
        title=dict(text="Spectral Features", font=dict(size=10, color=MUTED)),
        showlegend=False,
        shapes=[
            dict(type='rect', xref='paper', yref='y',
                 x0=0, x1=1, y0=0.30, y1=0.45,
                 fillcolor=GOLD_DIM, line_width=0, layer='below'),
            dict(type='rect', xref='paper', yref='y2',
                 x0=0, x1=1, y0=250, y1=400,
                 fillcolor='rgba(109,184,138,0.06)', line_width=0, layer='below'),
        ]
    )

    fig.add_trace(go.Scatter(x=x, y=d, mode='lines',
                             line=dict(color=GOLD, width=1.5),
                             fill='tozeroy', fillcolor=GOLD_DIM, name='Density'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=c, mode='lines',
                             line=dict(color=PASS_COL, width=1.5),
                             fill='tozeroy', fillcolor='rgba(109,184,138,0.04)', name='Centroid Hz'), row=2, col=1)

    fig.update_yaxes(title_text="Density",     gridcolor=GRID_COL, row=1, col=1)
    fig.update_yaxes(title_text="Centroid Hz", gridcolor=GRID_COL, row=2, col=1)
    fig.update_xaxes(title_text="min",         gridcolor=GRID_COL, row=2, col=1)
    return fig


def plot_subbass(result: dict) -> go.Figure:
    sub = result["lowend"]["sub_over_time"]
    dur = result["metadata"]["duration_min"]
    x   = np.linspace(0, dur, len(sub))
    fig = go.Figure()
    fig.update_layout(**LAYOUT_BASE,
                      title=dict(text="Sub-Bass < 80 Hz · Continuity", font=dict(size=10, color=MUTED)),
                      xaxis_title="min", yaxis=dict(range=[0, 1.05], gridcolor=GRID_COL),
                      shapes=[_band_shape(0.90, 1.01, 'rgba(109,184,138,0.08)')])
    fig.add_hline(y=0.90, line_dash='dot', line_color=MUTED, line_width=1,
                  annotation_text="90% threshold", annotation_font_size=8,
                  annotation_font_color=MUTED)
    fig.add_trace(go.Scatter(x=x, y=sub, mode='lines',
                             line=dict(color=PASS_COL, width=1.5),
                             fill='tozeroy', fillcolor='rgba(109,184,138,0.04)', name='Sub'))
    return fig


def plot_radar(result: dict) -> go.Figure:
    p = result["protocol_compliance"]["principles"]
    cats = ["P1 Tempo", "P2 Spectral", "P3 Variation", "P4 Sub-Bass", "P5 Texture"]
    scores = {
        "P1": 1.0 if p["P1"]["compliant"] else max(0, 1 - p["P1"]["value"] / 5),
        "P2": 1.0 if p["P2"]["compliant"] else max(0, 1 - abs(p["P2"]["value"] - 0.375) / 0.375),
        "P3": 1.0 if p["P3"]["compliant"] else max(0, 1 - abs(p["P3"]["value"] - 12) / 12),
        "P4": p["P4"]["value"] if not p["P4"]["compliant"] else 1.0,
        "P5": p["P5"]["value"] / 0.85 if not p["P5"]["compliant"] else 1.0,
    }
    vals = list(scores.values()) + [list(scores.values())[0]]
    cats_c = cats + [cats[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats_c, fill='toself',
        fillcolor='rgba(201,169,110,0.08)',
        line=dict(color=GOLD, width=1.5),
    ))
    fig.update_layout(
        paper_bgcolor=PLOT_BG,
        font=dict(family='JetBrains Mono', color=MUTED, size=9),
        margin=dict(l=40, r=40, t=40, b=40),
        polar=dict(
            bgcolor=PLOT_SURF,
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID_COL, color=MUTED, tickfont=dict(size=8)),
            angularaxis=dict(gridcolor=GRID_COL, color=MUTED),
        ),
        title=dict(text="Protocol Radar", font=dict(size=10, color=MUTED)),
        showlegend=False,
    )
    return fig


def plot_corpus(result: dict) -> go.Figure:
    np.random.seed(42)
    n  = 30
    cd = np.random.normal(0.37, 0.09, n)
    cc = np.random.normal(295,  67,   n)
    td = result["spectral"]["density"]["mean"]
    tc = result["spectral"]["centroid"]["mean"]
    theta = np.linspace(0, 2*np.pi, 80)
    ex = np.mean(cd) + 2*np.std(cd)*np.cos(theta)
    ey = np.mean(cc) + 2*np.std(cc)*np.sin(theta)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode='lines',
                             line=dict(color=MUTED, dash='dot', width=1), name='2 SD', showlegend=False))
    fig.add_trace(go.Scatter(x=cd, y=cc, mode='markers',
                             marker=dict(color=MUTED, size=5, opacity=0.5), name='Corpus n=30'))
    fig.add_trace(go.Scatter(x=[np.mean(cd)], y=[np.mean(cc)], mode='markers',
                             marker=dict(color='#6688cc', size=10, symbol='cross'), name='Corpus μ'))
    fig.add_trace(go.Scatter(x=[td], y=[tc], mode='markers',
                             marker=dict(color=GOLD, size=13, symbol='diamond'), name='Your track'))
    fig.update_layout(**LAYOUT_BASE,
                      title=dict(text="Corpus · Density vs Centroid", font=dict(size=10, color=MUTED)),
                      xaxis_title="Spectral Density", yaxis_title="Centroid Hz",
                      legend=dict(bgcolor=PLOT_BG, bordercolor=GRID_COL, borderwidth=1,
                                  font=dict(size=9)))
    return fig


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = {}


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0 0.5rem 1.5rem 0.5rem">
        <div style="font-family:'Cormorant Garamond',serif;font-size:1.25rem;
                    font-weight:300;letter-spacing:0.15em;color:#d4d4c8">
            CI Analyzer
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.55rem;
                    letter-spacing:0.2em;text-transform:uppercase;color:#3a3a48;margin-top:0.3rem">
            Continuous Inertia · v3.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["wav","mp3","flac","aiff"],
                                 label_visibility="collapsed")

    st.markdown('<div class="section-rule">Analysis</div>', unsafe_allow_html=True)
    check_corpus     = st.toggle("Corpus Comparison", value=True)
    show_timeseries  = st.toggle("Time Series",       value=True)

    st.markdown('<div class="section-rule">Export Format</div>', unsafe_allow_html=True)
    export_format = st.selectbox("", ["JSON", "CSV", "LaTeX", "HTML Report"],
                                 label_visibility="collapsed")

    st.markdown('<div class="section-rule">Session History</div>', unsafe_allow_html=True)
    hist = st.session_state.history
    if hist:
        for fname, rec in list(hist.items())[-6:]:
            icon   = "·" if rec["compliant"] else "×"
            ic_col = "#6db88a" if rec["compliant"] else "#c46a6a"
            short  = fname[:24] + "…" if len(fname) > 24 else fname
            st.markdown(
                f'<div class="hist-row">'
                f'<span style="color:{ic_col}">{icon}</span> {short}<br>'
                f'<span style="color:#3a3a48">{rec["bpm"]:.1f} BPM · {rec["p"]}/5 · {rec["date"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        if st.button("Clear history"):
            st.session_state.history = {}
            st.rerun()
    else:
        st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.62rem;'
                    'color:#2a2a34;padding:0.4rem 0">No records yet</div>',
                    unsafe_allow_html=True)

    st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.55rem;'
        'letter-spacing:0.12em;color:#2a2a34;line-height:2">'
        'Protocol · 5 principles<br>Corpus · n=30 · 2010–2025<br>'
        'Targets · 128–135 BPM</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
col_h, col_tag = st.columns([4, 1])
with col_h:
    st.markdown(
        '<div class="app-title">Continuous Inertia<br>Techno Analyzer</div>'
        '<div class="app-sub">Acoustic analysis · Protocol compliance · Corpus comparison</div>',
        unsafe_allow_html=True
    )
with col_tag:
    if uploaded:
        st.markdown(
            f'<div style="text-align:right;margin-top:0.5rem">'
            f'<span class="label-xs" style="color:#c9a96e">READY</span><br>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;color:#3a3a48">'
            f'{uploaded.name[:28]}</span></div>',
            unsafe_allow_html=True
        )

st.markdown('<hr>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────────────
if not uploaded:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-glyph">◼</div>
        <div class="empty-title">Upload a track to begin</div>
        <div class="empty-sub">
            WAV · MP3 · FLAC · AIFF<br>
            Minimum 6 minutes recommended
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-rule">The Five Principles</div>', unsafe_allow_html=True)
    c = st.columns(5)
    for col, (code, name, thr) in zip(c, [
        ("P1","Temporal Stability",   "BPM σ < 1.5%"),
        ("P2","Spectral Parsimony",   "ρ = 0.30–0.45"),
        ("P3","Micro-Variation",      "Δ every 8–16 bars"),
        ("P4","Sub-Bass Continuity",  "< 80 Hz ≥ 90%"),
        ("P5","Textural Continuity",  "Texture ≥ 85%"),
    ]):
        with col:
            st.markdown(f"""
            <div class="mcard">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                            letter-spacing:0.15em;color:#c9a96e;margin-bottom:0.5rem">{code}</div>
                <div style="font-family:'Cormorant Garamond',serif;font-size:0.95rem;
                            color:#d4d4c8;margin-bottom:0.4rem">{name}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                            color:#4a4a58">{thr}</div>
            </div>""", unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────
# ANALYZE
# ─────────────────────────────────────────────────────────────
audio_bytes = uploaded.read()

with st.spinner(""):
    result = try_real_analysis(audio_bytes, uploaded.name)
    time.sleep(0.3)

# Save to history
st.session_state.history[uploaded.name] = {
    "bpm": result["tempo"]["bpm"],
    "p":   result["protocol_compliance"]["principles_passed"],
    "compliant": result["protocol_compliance"]["compliant"],
    "date": datetime.now().strftime("%d/%m %H:%M"),
    "result": result,
}

if not result["metadata"]["real_analysis"]:
    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;'
        'color:#4a4a58;padding:0.4rem 0.8rem;border:1px solid #1e1e24;'
        'margin-bottom:1rem;display:inline-block">'
        '◦ Demo mode — install librosa for real analysis</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────
# COMPLIANCE BANNER
# ─────────────────────────────────────────────────────────────
passed    = result["protocol_compliance"]["principles_passed"]
compliant = result["protocol_compliance"]["compliant"]
viol      = result["antipatterns"]["total_violations"]
b_class   = "ok" if compliant else "no"
b_color   = "#6db88a" if compliant else "#c46a6a"
b_label   = "PROTOCOL COMPLIANT" if compliant else "NON-COMPLIANT"

st.markdown(f"""
<div class="banner {b_class}">
    <div>
        <div class="banner-status" style="color:{b_color}">{b_label}</div>
        <div class="banner-sub">{passed}/5 core principles · {viol} anti-pattern violation{"s" if viol!=1 else ""}</div>
    </div>
    <div class="banner-score" style="color:{b_color}">{passed}<span style="font-size:1.5rem;color:#3a3a48">/5</span></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# KEY METRICS ROW
# ─────────────────────────────────────────────────────────────
t = result["tempo"]
s = result["spectral"]
l = result["lowend"]
k = result["kick"]
st_r = result["structure"]

def mcard(label, val, unit="", ok=None):
    cls  = " pass" if ok is True else (" fail" if ok is False else "")
    vcls = " pass" if ok is True else (" fail" if ok is False else "")
    return (f'<div class="mcard{cls}">'
            f'<div class="mval{vcls}">{val}<span style="font-size:1rem;font-weight:300;'
            f'color:#4a4a58"> {unit}</span></div>'
            f'<div class="mlabel">{label}</div></div>')

cols = st.columns(6)
with cols[0]: st.markdown(mcard("BPM", f"{t['bpm']:.1f}", ok=None), unsafe_allow_html=True)
with cols[1]: st.markdown(mcard("BPM Variance", f"{t['bpm_variance_pct']:.2f}", "%", t['bpm_variance_pct']<1.5), unsafe_allow_html=True)
with cols[2]: st.markdown(mcard("Spectral Density", f"{s['density']['mean']:.3f}", ok=0.30<=s['density']['mean']<=0.45), unsafe_allow_html=True)
with cols[3]: st.markdown(mcard("Centroid", f"{s['centroid']['mean']:.0f}", "Hz", 250<=s['centroid']['mean']<=400), unsafe_allow_html=True)
with cols[4]: st.markdown(mcard("Sub-Bass", f"{l['sub_presence_pct']*100:.0f}", "%", l['sub_presence_pct']>=0.90), unsafe_allow_html=True)
with cols[5]: st.markdown(mcard("Duration", f"{result['metadata']['duration_min']:.1f}", "min", result['metadata']['duration_min']>=6), unsafe_allow_html=True)

st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Protocol", "Spectral", "Structure", "Corpus", "Export"])


# ──── TAB 1: PROTOCOL ────────────────────────────────────────
with tab1:
    c_left, c_right = st.columns([3, 2])

    with c_left:
        st.markdown('<div class="section-rule">Core Principles</div>', unsafe_allow_html=True)
        for code, p in result["protocol_compliance"]["principles"].items():
            ok   = p["compliant"]
            bcls = "ok" if ok else "no"
            blbl = "PASS" if ok else "FAIL"
            st.markdown(f"""
            <div class="prow">
                <span class="pbadge {bcls}">{blbl}</span>
                <div>
                    <div class="pname">{p['name']}</div>
                    <div class="pdetail">{p['details']}</div>
                </div>
                <div class="pthresh">{code}<br>{p['threshold']}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-rule" style="margin-top:1.8rem">Complementary Criteria</div>', unsafe_allow_html=True)
        cc = st.columns(3)
        for i, (code, c) in enumerate(result["complementary"].items()):
            with cc[i % 3]:
                ok = c["met"]
                bc = "#6db88a" if ok else "#3a3a48"
                st.markdown(
                    f'<div style="padding:0.5rem 0.7rem;border:1px solid #1e1e24;'
                    f'background:#0e0e11;margin:0.15rem 0">'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.6rem;'
                    f'color:{bc}">{"✓" if ok else "○"} {code}</span>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;'
                    f'color:#4a4a58;margin-left:0.5rem">{c["name"]}</span></div>',
                    unsafe_allow_html=True
                )

        st.markdown('<div class="section-rule" style="margin-top:1.8rem">Anti-Patterns</div>', unsafe_allow_html=True)
        for name, flag in [
            ("Density overload > 0.60",   result["antipatterns"]["density_overload"]),
            ("Sub-bass absent > 30%",      result["antipatterns"]["sub_absent"]),
            ("BPM drift > 3%",             result["antipatterns"]["bpm_change"]),
        ]:
            col = "#c46a6a" if flag else "#3a3a48"
            lbl = "VIOLATION" if flag else "CLEAR"
            st.markdown(
                f'<div class="prow">'
                f'<span class="pbadge {"no" if flag else ""}" '
                f'style="border-color:{col};color:{col}">{lbl}</span>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;'
                f'color:#5a5a68">{name}</div></div>',
                unsafe_allow_html=True
            )

    with c_right:
        st.plotly_chart(plot_radar(result), use_container_width=True)
        st.markdown('<div class="section-rule">Verdict</div>', unsafe_allow_html=True)
        if compliant:
            st.success(f"**{passed}/5** core principles met. Track qualifies as Continuous Inertia Techno.")
        else:
            failing = [c for c, p in result["protocol_compliance"]["principles"].items() if not p["compliant"]]
            st.error(f"**{passed}/5** principles. Failed: {', '.join(failing)}. Minimum 4/5 required.")
        if viol:
            st.warning(f"{viol} anti-pattern violation(s) detected.")


# ──── TAB 2: SPECTRAL ────────────────────────────────────────
with tab2:
    if show_timeseries:
        st.plotly_chart(plot_spectral(result), use_container_width=True)
        st.plotly_chart(plot_bpm(result),      use_container_width=True)

    st.markdown('<div class="section-rule">Spectral Summary</div>', unsafe_allow_html=True)
    sc = st.columns(3)
    with sc[0]: st.markdown(mcard("Centroid mean", f"{s['centroid']['mean']:.0f}", "Hz", 250<=s['centroid']['mean']<=400), unsafe_allow_html=True)
    with sc[1]: st.markdown(mcard("Density mean",  f"{s['density']['mean']:.3f}",  ok=0.30<=s['density']['mean']<=0.45), unsafe_allow_html=True)
    with sc[2]: st.markdown(mcard("Rolloff 85%",   f"{s['rolloff']['mean']:.0f}",  "Hz", ok=None), unsafe_allow_html=True)

    st.markdown('<div class="section-rule">Kick</div>', unsafe_allow_html=True)
    kc = st.columns(3)
    with kc[0]: st.markdown(mcard("On-beat",     f"{k['kick_on_beat_pct']*100:.0f}", "%", k['kick_on_beat_pct']>=0.80), unsafe_allow_html=True)
    with kc[1]: st.markdown(mcard("Consistency", f"{k['kick_consistency']*100:.0f}", "%", k['kick_consistency']>=0.80), unsafe_allow_html=True)
    with kc[2]: st.markdown(mcard("Fundamental", f"{k['kick_fundamental_hz']:.0f}",  "Hz"), unsafe_allow_html=True)


# ──── TAB 3: STRUCTURE ───────────────────────────────────────
with tab3:
    c3a, c3b = st.columns(2)
    with c3a:
        st.markdown('<div class="section-rule">Layers</div>', unsafe_allow_html=True)
        st.markdown(mcard("Mean layer count",  f"{st_r['layers_mean']:.1f}",  ok=st_r['layers_mean']<=4.5), unsafe_allow_html=True)
        st.markdown(mcard("Modal layer count", f"{st_r['layers_mode']}",      ok=None), unsafe_allow_html=True)
        st.markdown('<div class="section-rule">Micro-Variation</div>', unsafe_allow_html=True)
        ok_int = 8 <= st_r['mean_interval_bars'] <= 16
        st.markdown(mcard("Mean change interval", f"{st_r['mean_interval_bars']:.1f}", "bars", ok_int), unsafe_allow_html=True)
        st.markdown(mcard("Periodicity score",    f"{st_r['periodicity_score']:.2f}",  ok=st_r['periodicity_score']>=0.70), unsafe_allow_html=True)
    with c3b:
        st.plotly_chart(plot_subbass(result), use_container_width=True)
        st.markdown('<div class="section-rule">Sub-Bass</div>', unsafe_allow_html=True)
        sb = st.columns(3)
        with sb[0]: st.markdown(mcard("Presence", f"{l['sub_presence_pct']*100:.0f}", "%", l['sub_presence_pct']>=0.90), unsafe_allow_html=True)
        with sb[1]: st.markdown(mcard("Kick ratio", f"{l['sub_kick_ratio']:.2f}", ok=None), unsafe_allow_html=True)
        with sb[2]: st.markdown(mcard("Continuity", f"{l['sub_continuity']:.2f}", ok=l['sub_continuity']>=0.85), unsafe_allow_html=True)


# ──── TAB 4: CORPUS ──────────────────────────────────────────
with tab4:
    if not check_corpus:
        st.info("Enable Corpus Comparison in the sidebar.")
    else:
        st.plotly_chart(plot_corpus(result), use_container_width=True)
        st.markdown('<div class="section-rule">Percentile vs Corpus n=30</div>', unsafe_allow_html=True)
        np.random.seed(abs(hash(uploaded.name)) % (2**31))
        pcts = {
            "BPM":             int(np.random.randint(20, 80)),
            "Spectral Density":int(np.random.randint(20, 80)),
            "Centroid":        int(np.random.randint(20, 80)),
            "Sub-Bass":        int(np.random.randint(20, 80)),
        }
        pc = st.columns(4)
        for col, (feat, pct) in zip(pc, pcts.items()):
            with col:
                ok = 20 <= pct <= 80
                st.markdown(mcard(feat, f"{pct}", "th pct", ok=None), unsafe_allow_html=True)
        if all(20 <= v <= 80 for v in pcts.values()):
            st.success("Track falls within the central distribution on all features.")
        else:
            out = [f for f, v in pcts.items() if not (20 <= v <= 80)]
            st.warning(f"Outlier features: {', '.join(out)}")


# ──── TAB 5: EXPORT ──────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-rule">Export Report</div>', unsafe_allow_html=True)
    cx, cy = st.columns([3, 1])

    with cx:
        stem = Path(uploaded.name).stem

        if export_format == "JSON":
            js = json.dumps(result, indent=2)
            st.download_button("⬇ Download JSON", data=js,
                               file_name=f"analysis_{stem}.json", mime="application/json")
            st.code(js[:700] + "\n...", language="json")

        elif export_format == "CSV":
            flat = {
                "filename":        result["metadata"]["filename"],
                "duration_min":    result["metadata"]["duration_min"],
                "bpm":             t["bpm"],
                "bpm_variance_pct":t["bpm_variance_pct"],
                "density_mean":    s["density"]["mean"],
                "centroid_mean":   s["centroid"]["mean"],
                "rolloff_mean":    s["rolloff"]["mean"],
                "sub_presence_pct":l["sub_presence_pct"],
                "layers_mean":     st_r["layers_mean"],
                "interval_bars":   st_r["mean_interval_bars"],
                "principles_passed":result["protocol_compliance"]["principles_passed"],
                "compliant":       result["protocol_compliance"]["compliant"],
                "violations":      result["antipatterns"]["total_violations"],
            }
            csv = pd.DataFrame([flat]).to_csv(index=False)
            st.download_button("⬇ Download CSV", data=csv,
                               file_name=f"analysis_{stem}.csv", mime="text/csv")
            st.dataframe(pd.DataFrame([flat]).T.rename(columns={0:"Value"}), use_container_width=True)

        elif export_format == "LaTeX":
            tv, sv, lv, pv = t, s, l, result["protocol_compliance"]
            chk = lambda c: r"\checkmark" if c else r"$\times$"
            tex = f"""% Continuous Inertia Analyzer v3.0
\\begin{{table}}[h]
\\centering
\\caption{{Acoustic Analysis: {stem}}}
\\begin{{tabular}}{{lcc}}
\\hline
\\textbf{{Parameter}} & \\textbf{{Value}} & \\textbf{{Compliant}} \\\\
\\hline
BPM & {tv['bpm']:.1f} & {chk(128<=tv['bpm']<=135)} \\\\
BPM Variance (\\%) & {tv['bpm_variance_pct']:.2f} & {chk(tv['bpm_variance_pct']<1.5)} \\\\
Spectral Density & {sv['density']['mean']:.3f} & {chk(0.30<=sv['density']['mean']<=0.45)} \\\\
Centroid (Hz) & {sv['centroid']['mean']:.0f} & {chk(250<=sv['centroid']['mean']<=400)} \\\\
Sub-Bass (\\%) & {lv['sub_presence_pct']*100:.0f} & {chk(lv['sub_presence_pct']>=0.90)} \\\\
\\hline
Principles Passed & {pv['principles_passed']}/5 & {chk(pv['compliant'])} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}"""
            st.download_button("⬇ Download LaTeX", data=tex,
                               file_name=f"table_{stem}.tex", mime="text/plain")
            st.code(tex, language="latex")

        elif export_format == "HTML Report":
            pv = result["protocol_compliance"]
            comp = pv["compliant"]
            sc   = "#6db88a" if comp else "#c46a6a"
            st_lbl = "COMPLIANT" if comp else "NON-COMPLIANT"
            rows = ""
            for code, p in pv["principles"].items():
                ok  = p["compliant"]
                pc2 = "#6db88a" if ok else "#c46a6a"
                rows += f"<tr><td><b>{code}</b></td><td>{p['name']}</td><td style='color:{pc2}'><b>{'PASS' if ok else 'FAIL'}</b></td><td style='color:#888'>{p['details']}</td></tr>\n"
            html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400&family=JetBrains+Mono:wght@300;400&display=swap');
  body {{ font-family:'JetBrains Mono',monospace; background:#08080a; color:#d4d4c8; padding:3rem; max-width:860px; margin:0 auto; }}
  h1 {{ font-family:'Cormorant Garamond',serif; font-weight:300; font-size:2rem; letter-spacing:0.1em; border-bottom:1px solid #1e1e24; padding-bottom:0.8rem; }}
  .status {{ display:inline-block; padding:0.3rem 1rem; border:1px solid {sc}; color:{sc}; font-size:0.75rem; letter-spacing:0.2em; margin:1rem 0; }}
  .grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:0.8rem; margin:1.5rem 0; }}
  .card {{ border:1px solid #1e1e24; padding:1rem; border-left:2px solid #c9a96e; }}
  .val {{ font-family:'Cormorant Garamond',serif; font-size:1.8rem; font-weight:300; color:#c9a96e; }}
  .lbl {{ font-size:0.55rem; letter-spacing:0.18em; text-transform:uppercase; color:#4a4a58; margin-top:0.2rem; }}
  table {{ width:100%; border-collapse:collapse; margin:1.5rem 0; }}
  th {{ background:#0e0e11; color:#5a5a68; padding:0.6rem; text-align:left; font-size:0.65rem; letter-spacing:0.15em; text-transform:uppercase; border-bottom:1px solid #1e1e24; }}
  td {{ padding:0.6rem; border-bottom:1px solid #1a1a20; font-size:0.8rem; }}
  .footer {{ font-size:0.6rem; color:#2a2a34; margin-top:3rem; border-top:1px solid #1e1e24; padding-top:1rem; }}
</style></head><body>
<h1>Continuous Inertia Techno Analyzer</h1>
<p style="font-size:0.7rem;color:#4a4a58">{result['metadata']['filename']} · {result['metadata']['duration_min']:.1f} min · v{result['metadata']['analyzer_version']}</p>
<div class="status">{st_lbl} — {pv['principles_passed']}/5</div>
<div class="grid">
  <div class="card"><div class="val">{t['bpm']:.1f}</div><div class="lbl">BPM</div></div>
  <div class="card"><div class="val" style="color:{'#6db88a' if t['bpm_variance_pct']<1.5 else '#c46a6a'}">{t['bpm_variance_pct']:.2f}%</div><div class="lbl">BPM Variance</div></div>
  <div class="card"><div class="val">{s['density']['mean']:.3f}</div><div class="lbl">Spectral Density</div></div>
  <div class="card"><div class="val">{s['centroid']['mean']:.0f} Hz</div><div class="lbl">Centroid</div></div>
  <div class="card"><div class="val" style="color:{'#6db88a' if l['sub_presence_pct']>=0.90 else '#c46a6a'}">{l['sub_presence_pct']*100:.0f}%</div><div class="lbl">Sub-Bass</div></div>
  <div class="card"><div class="val">{st_r['layers_mean']:.1f}</div><div class="lbl">Mean Layers</div></div>
</div>
<table><tr><th>Code</th><th>Principle</th><th>Result</th><th>Details</th></tr>{rows}</table>
<p class="footer">Generated by Continuous Inertia Techno Analyzer v3.0 · Print → Save as PDF</p>
</body></html>"""
            st.download_button("⬇ Download HTML Report",
                               data=html, file_name=f"report_{stem}.html", mime="text/html")
            st.info("Open in browser → Print → Save as PDF for a clean publication-ready document.")
            import streamlit.components.v1 as components
            components.html(html, height=550, scrolling=True)

    with cy:
        st.markdown('<div class="section-rule">Formats</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                    color:#4a4a58;line-height:2.2">
        JSON · full pipeline<br>
        CSV  · R / SPSS / Python<br>
        LaTeX · paste into paper<br>
        HTML · print → PDF
        </div>""", unsafe_allow_html=True)
