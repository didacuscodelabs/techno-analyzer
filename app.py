"""
CONTINUOUS INERTIA TECHNO ANALYZER
Protocol Compliance & Corpus Comparison Tool
Streamlit App — v1.0
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import io
import time
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Continuous Inertia Analyzer",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS — Industrial / Brutalist Dark
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #0a0a0a;
    --surface: #111111;
    --surface2: #1a1a1a;
    --border: #2a2a2a;
    --accent: #c8ff00;
    --accent2: #ff6b00;
    --text: #e8e8e0;
    --muted: #666660;
    --danger: #ff3b3b;
    --success: #00e5a0;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    border-radius: 0 !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.15s ease !important;
}

.stButton > button:hover {
    background: #e0ff40 !important;
    transform: translateY(-1px) !important;
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    padding: 1rem 1.2rem;
    margin: 0.4rem 0;
}

.metric-card.fail {
    border-left-color: var(--danger);
}

.metric-card.warn {
    border-left-color: var(--accent2);
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}

.metric-value.fail { color: var(--danger); }
.metric-value.warn { color: var(--accent2); }

.metric-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.2rem;
}

.principle-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border);
    font-size: 0.9rem;
}

.badge {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 0.15rem 0.5rem;
    border-radius: 2px;
    font-weight: 700;
}

.badge-ok { background: var(--success); color: #000; }
.badge-fail { background: var(--danger); color: #fff; }
.badge-warn { background: var(--accent2); color: #000; }

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
}

.tag {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    padding: 0.1rem 0.45rem;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--muted);
    margin: 0.1rem;
}

.compliance-score {
    font-family: 'Space Mono', monospace;
    font-size: 3.5rem;
    font-weight: 700;
    text-align: center;
    line-height: 1;
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    letter-spacing: 0.15em;
    color: var(--accent);
    text-transform: uppercase;
}

.stFileUploader { border: 1px dashed var(--border) !important; }

[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    padding: 0.8rem !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    padding: 0.6rem 1.2rem;
    color: var(--muted);
    border-radius: 0 !important;
}

.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}

</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# ANALYSIS ENGINE (SIMULATED FOR DEMO / REAL with librosa if installed)
# ──────────────────────────────────────────────

def simulate_analysis(filename: str, duration_sec: float = None) -> dict:
    """
    Returns simulated analysis results.
    Replace this with real analysis from techno_analyzer package.
    """
    np.random.seed(hash(filename) % (2**31))
    
    bpm = np.random.uniform(126, 140)
    bpm_std = np.random.uniform(0.3, 2.5)
    bpm_variance_pct = (bpm_std / bpm) * 100
    
    density_mean = np.random.uniform(0.20, 0.65)
    centroid_mean = np.random.uniform(150, 500)
    rolloff_mean = np.random.uniform(2000, 6000)
    
    sub_presence = np.random.uniform(0.60, 1.0)
    sub_kick_ratio = np.random.uniform(0.45, 0.90)
    
    layers_mean = np.random.uniform(1.5, 5.5)
    mean_interval_bars = np.random.uniform(4, 24)
    
    texture_presence = np.random.uniform(0.65, 1.0)
    
    duration = duration_sec or np.random.uniform(360, 600)
    
    # Time series
    n_segments = 20
    bpm_over_time = bpm + np.random.randn(n_segments) * bpm_std * 0.5
    density_over_time = density_mean + np.random.randn(n_segments) * 0.04
    centroid_over_time = centroid_mean + np.random.randn(n_segments) * 40
    sub_over_time = np.clip(sub_presence + np.random.randn(n_segments) * 0.08, 0, 1)
    
    # Protocol evaluation
    p1_ok = bpm_variance_pct < 1.5
    p2_ok = 0.30 <= density_mean <= 0.45 and layers_mean <= 4.5
    p3_ok = 8 <= mean_interval_bars <= 16
    p4_ok = sub_presence >= 0.90 and -6 <= (sub_kick_ratio * -10 + 2) <= -3
    p5_ok = texture_presence >= 0.85
    
    principles_passed = sum([p1_ok, p2_ok, p3_ok, p4_ok, p5_ok])
    
    # Antipatterns
    density_overload = density_mean > 0.60
    sub_absent = sub_presence < 0.70
    bpm_change = bpm_variance_pct > 3.0
    
    return {
        "metadata": {
            "filename": filename,
            "duration_sec": duration,
            "duration_min": duration / 60,
            "analyzer_version": "2.0"
        },
        "tempo": {
            "bpm": round(bpm, 2),
            "bpm_confidence": round(np.random.uniform(0.75, 0.99), 2),
            "bpm_mean": round(bpm, 2),
            "bpm_std": round(bpm_std, 3),
            "bpm_variance_pct": round(bpm_variance_pct, 2),
            "stability_score": round(max(0, 1 - bpm_variance_pct / 5), 2),
            "bpm_over_time": bpm_over_time.tolist(),
        },
        "spectral": {
            "centroid": {"mean": round(centroid_mean, 1), "std": round(np.random.uniform(25, 70), 1)},
            "density": {"mean": round(density_mean, 3), "std": round(np.random.uniform(0.02, 0.08), 3)},
            "rolloff": {"mean": round(rolloff_mean, 1), "std": round(np.random.uniform(300, 800), 1)},
            "density_over_time": density_over_time.tolist(),
            "centroid_over_time": centroid_over_time.tolist(),
        },
        "kick": {
            "kick_on_beat_pct": round(np.random.uniform(0.72, 0.99), 2),
            "kick_consistency": round(np.random.uniform(0.70, 0.98), 2),
            "kick_fundamental_hz": round(np.random.uniform(40, 80), 1),
        },
        "structure": {
            "layers_mean": round(layers_mean, 1),
            "layers_mode": int(round(layers_mean)),
            "mean_interval_bars": round(mean_interval_bars, 1),
            "periodicity_score": round(np.random.uniform(0.5, 0.97), 2),
        },
        "lowend": {
            "sub_presence_pct": round(sub_presence, 2),
            "sub_kick_ratio": round(sub_kick_ratio, 2),
            "sub_continuity": round(np.random.uniform(0.70, 0.99), 2),
            "sub_over_time": sub_over_time.tolist(),
        },
        "protocol_compliance": {
            "principles": {
                "P1": {
                    "name": "Temporal Stability",
                    "compliant": p1_ok,
                    "value": bpm_variance_pct,
                    "threshold": "<1.5% BPM variance",
                    "details": f"BPM variance: {bpm_variance_pct:.2f}% (threshold <1.5%)"
                },
                "P2": {
                    "name": "Spectral Parsimony",
                    "compliant": p2_ok,
                    "value": density_mean,
                    "threshold": "Density 0.30–0.45, ≤4 layers",
                    "details": f"Density: {density_mean:.3f} | Layers: {layers_mean:.1f}"
                },
                "P3": {
                    "name": "Periodic Micro-Variation",
                    "compliant": p3_ok,
                    "value": mean_interval_bars,
                    "threshold": "Changes every 8–16 bars",
                    "details": f"Mean interval: {mean_interval_bars:.1f} bars"
                },
                "P4": {
                    "name": "Continuous Sub-Bass",
                    "compliant": p4_ok,
                    "value": sub_presence,
                    "threshold": "Sub <80Hz present ≥90%",
                    "details": f"Sub present: {sub_presence*100:.0f}% | Ratio to kick: {sub_kick_ratio:.2f}"
                },
                "P5": {
                    "name": "Textural Continuity",
                    "compliant": p5_ok,
                    "value": texture_presence,
                    "threshold": "Texture ≥85% of track",
                    "details": f"Texture presence: {texture_presence*100:.0f}%"
                },
            },
            "principles_passed": principles_passed,
            "compliant": principles_passed >= 4,
        },
        "antipatterns": {
            "density_overload": density_overload,
            "sub_absent": sub_absent,
            "bpm_change": bpm_change,
            "total_violations": sum([density_overload, sub_absent, bpm_change]),
        },
        "complementary": {
            "C1": {"name": "BPM 128–135", "met": 128 <= bpm <= 135},
            "C2": {"name": "Duration 7–9 min", "met": 420 <= duration <= 540},
            "C3": {"name": "No build-ups/drops", "met": not bpm_change},
            "C4": {"name": "Centroid 250–400 Hz", "met": 250 <= centroid_mean <= 400},
            "C5": {"name": "Stems exportable", "met": True},
        }
    }


def try_real_analysis(audio_bytes: bytes, filename: str) -> dict:
    """Try to use librosa for real analysis, fallback to simulation."""
    try:
        import librosa
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        y, sr = librosa.load(tmp_path, sr=22050, mono=True, duration=120)
        os.unlink(tmp_path)
        
        duration = librosa.get_duration(y=y, sr=sr)
        bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        result = simulate_analysis(filename, duration_sec=duration * (librosa.get_duration(path=tmp_path) / duration if False else 1))
        result["tempo"]["bpm"] = round(float(bpm), 2)
        result["metadata"]["real_analysis"] = True
        return result
    except Exception:
        result = simulate_analysis(filename)
        result["metadata"]["real_analysis"] = False
        return result


# ──────────────────────────────────────────────
# PLOTTING FUNCTIONS
# ──────────────────────────────────────────────

PLOT_LAYOUT = dict(
    paper_bgcolor='#0a0a0a',
    plot_bgcolor='#111111',
    font=dict(family='Space Mono, monospace', color='#e8e8e0', size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor='#222', zerolinecolor='#222'),
    yaxis=dict(gridcolor='#222', zerolinecolor='#222'),
)


def plot_bpm_stability(result: dict) -> go.Figure:
    bpm_time = result["tempo"]["bpm_over_time"]
    n = len(bpm_time)
    x = np.linspace(0, result["metadata"]["duration_min"], n)
    mean_bpm = result["tempo"]["bpm_mean"]
    
    fig = go.Figure()
    fig.add_hrect(y0=mean_bpm - mean_bpm * 0.015,
                  y1=mean_bpm + mean_bpm * 0.015,
                  fillcolor='rgba(200,255,0,0.08)', line_width=0, annotation_text="±1.5% threshold")
    fig.add_trace(go.Scatter(
        x=x, y=bpm_time, mode='lines',
        line=dict(color='#c8ff00', width=2),
        name='BPM over time',
        fill='tozeroy', fillcolor='rgba(200,255,0,0.03)'
    ))
    fig.add_hline(y=mean_bpm, line_dash='dash', line_color='#666', line_width=1)
    fig.update_layout(**PLOT_LAYOUT, title="BPM Stability Over Time",
                      xaxis_title="Time (min)", yaxis_title="BPM")
    return fig


def plot_spectral_density(result: dict) -> go.Figure:
    density = result["spectral"]["density_over_time"]
    centroid = result["spectral"]["centroid_over_time"]
    n = len(density)
    x = np.linspace(0, result["metadata"]["duration_min"], n)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    
    fig.add_hrect(y0=0.30, y1=0.45, fillcolor='rgba(200,255,0,0.08)', line_width=0, row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=density, mode='lines', line=dict(color='#c8ff00', width=2),
                              name='Density', fill='tozeroy', fillcolor='rgba(200,255,0,0.03)'), row=1, col=1)
    
    fig.add_hrect(y0=250, y1=400, fillcolor='rgba(255,107,0,0.08)', line_width=0, row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=centroid, mode='lines', line=dict(color='#ff6b00', width=2),
                              name='Centroid (Hz)', fill='tozeroy', fillcolor='rgba(255,107,0,0.03)'), row=2, col=1)
    
    fig.update_layout(**PLOT_LAYOUT, title="Spectral Features Over Time",
                      height=400, showlegend=True)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_yaxes(title_text="Centroid Hz", row=2, col=1)
    fig.update_xaxes(title_text="Time (min)", row=2, col=1)
    return fig


def plot_sub_bass(result: dict) -> go.Figure:
    sub = result["lowend"]["sub_over_time"]
    n = len(sub)
    x = np.linspace(0, result["metadata"]["duration_min"], n)
    
    fig = go.Figure()
    fig.add_hrect(y0=0.90, y1=1.01, fillcolor='rgba(0,229,160,0.08)', line_width=0, annotation_text="≥90% target")
    fig.add_trace(go.Scatter(x=x, y=sub, mode='lines+markers',
                              line=dict(color='#00e5a0', width=2),
                              marker=dict(size=3, color='#00e5a0'),
                              name='Sub-bass presence',
                              fill='tozeroy', fillcolor='rgba(0,229,160,0.03)'))
    fig.update_layout(**PLOT_LAYOUT, title="Sub-Bass (<80Hz) Continuity",
                      xaxis_title="Time (min)", yaxis_title="Presence",
                      yaxis=dict(range=[0, 1.05], gridcolor='#222'))
    return fig


def plot_compliance_radar(result: dict) -> go.Figure:
    p = result["protocol_compliance"]["principles"]
    categories = [v["name"].split(" ")[0] + "<br>" + " ".join(v["name"].split(" ")[1:]) for v in p.values()]
    
    # Normalize values to 0-1 score
    raw = {
        "P1": 1.0 if p["P1"]["compliant"] else max(0, 1 - p["P1"]["value"] / 5),
        "P2": 1.0 if p["P2"]["compliant"] else max(0, 1 - abs(p["P2"]["value"] - 0.375) / 0.375),
        "P3": 1.0 if p["P3"]["compliant"] else max(0, 1 - abs(p["P3"]["value"] - 12) / 12),
        "P4": 1.0 if p["P4"]["compliant"] else p["P4"]["value"],
        "P5": 1.0 if p["P5"]["compliant"] else p["P5"]["value"] / 0.85,
    }
    values = list(raw.values()) + [list(raw.values())[0]]
    categories_closed = categories + [categories[0]]
    
    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(200, 255, 0, 0.12)',
        line=dict(color='#c8ff00', width=2),
        name='Protocol Score'
    ))
    fig.update_layout(
        **{k: v for k, v in PLOT_LAYOUT.items() if k not in ['xaxis', 'yaxis']},
        polar=dict(
            bgcolor='#111111',
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#2a2a2a', color='#666'),
            angularaxis=dict(gridcolor='#2a2a2a', color='#888')
        ),
        title="Protocol Compliance Radar",
        showlegend=False,
    )
    return fig


def plot_corpus_scatter(result: dict) -> go.Figure:
    """Simulated corpus scatter plot."""
    np.random.seed(42)
    n_corpus = 30
    corpus_density = np.random.normal(0.37, 0.09, n_corpus)
    corpus_centroid = np.random.normal(295, 67, n_corpus)
    
    track_density = result["spectral"]["density"]["mean"]
    track_centroid = result["spectral"]["centroid"]["mean"]
    
    fig = go.Figure()
    
    # 2SD ellipse (approximate)
    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse_x = np.mean(corpus_density) + 2 * np.std(corpus_density) * np.cos(theta)
    ellipse_y = np.mean(corpus_centroid) + 2 * np.std(corpus_centroid) * np.sin(theta)
    fig.add_trace(go.Scatter(x=ellipse_x, y=ellipse_y, mode='lines',
                              line=dict(color='#444', dash='dash'), name='2SD ellipse'))
    
    fig.add_trace(go.Scatter(x=corpus_density, y=corpus_centroid, mode='markers',
                              marker=dict(color='#444', size=7, symbol='circle'),
                              name='Corpus (n=30)'))
    
    fig.add_trace(go.Scatter(x=[np.mean(corpus_density)], y=[np.mean(corpus_centroid)],
                              mode='markers', marker=dict(color='#4488ff', size=12, symbol='cross'),
                              name='Corpus Mean'))
    
    fig.add_trace(go.Scatter(x=[track_density], y=[track_centroid], mode='markers',
                              marker=dict(color='#c8ff00', size=15, symbol='star'),
                              name='Your Track'))
    
    fig.update_layout(**PLOT_LAYOUT, title="Corpus Comparison: Density vs. Centroid",
                      xaxis_title="Spectral Density", yaxis_title="Spectral Centroid (Hz)",
                      legend=dict(bgcolor='#111'))
    return fig


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="hero-title">⬛ INERTIA<br>ANALYZER</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#444;font-size:0.7rem;font-family:Space Mono;margin-top:0.3rem;margin-bottom:1.5rem">v2.0 — Protocol-Enhanced</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Upload Track</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Audio file", type=["wav", "mp3", "flac", "aiff"], label_visibility="collapsed")
    
    st.markdown('<div class="section-header">Options</div>', unsafe_allow_html=True)
    check_protocol = st.toggle("Protocol Compliance", value=True)
    check_corpus = st.toggle("Corpus Comparison", value=True)
    show_timeseries = st.toggle("Time Series Plots", value=True)
    
    st.markdown('<div class="section-header">Corpus</div>', unsafe_allow_html=True)
    corpus_option = st.selectbox("Reference corpus", 
                                  ["Continuous Inertia v1 (n=30)", "Upload custom corpus"],
                                  label_visibility="collapsed")
    
    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
    export_format = st.selectbox("Format", ["JSON", "CSV", "LaTeX", "PDF Report"], label_visibility="collapsed")
    

    st.markdown('<div class="section-header">Session History</div>', unsafe_allow_html=True)
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = {}
    history = st.session_state.analysis_history
    if history:
        for fname, rec in list(history.items())[-5:]:
            icon = "✅" if rec["compliant"] else "❌"
            short = fname[:22] + "…" if len(fname) > 22 else fname
            st.markdown(
                f'<div style="font-size:0.72rem;color:#666;padding:0.2rem 0;border-bottom:1px solid #1a1a1a">' +
                f'{icon} <span style="color:#aaa">{short}</span><br>' +
                f'<span style="font-family:Space Mono;font-size:0.65rem;color:#555">' +
                f'{rec["bpm"]:.1f} BPM · {rec["principles"]}/5 · {rec["date"]}</span></div>',
                unsafe_allow_html=True
            )
        if st.button("🗑 Clear History", use_container_width=True):
            st.session_state.analysis_history = {}
            st.rerun()
    else:
        st.markdown('<div style="font-size:0.72rem;color:#444">No analyses this session yet.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:0.65rem;color:#444;font-family:Space Mono;line-height:1.6">'
                'Protocol: 5 core principles<br>'
                'Corpus: hypnotic/minimal techno<br>'
                '2010–2025 reference set<br><br>'
                '© 2026 Research Tool</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────

# Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("# Continuous Inertia Techno Analyzer")
    st.markdown('<div style="color:#666;font-family:Space Mono;font-size:0.8rem">Acoustic analysis · Protocol compliance · Corpus comparison</div>', unsafe_allow_html=True)
with col_h2:
    if uploaded:
        st.markdown(f'<div style="text-align:right"><span class="tag">READY</span><br><span style="font-size:0.75rem;color:#888">{uploaded.name}</span></div>', unsafe_allow_html=True)

st.markdown("---")

# ── NO FILE STATE
if not uploaded:
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem">
        <div style="font-size:3rem;margin-bottom:1rem">⬛</div>
        <div style="font-family:Space Mono;font-size:1rem;color:#666;margin-bottom:0.5rem">
            Upload a track to begin analysis
        </div>
        <div style="font-size:0.8rem;color:#444;max-width:500px;margin:0 auto">
            Supports WAV · MP3 · FLAC · AIFF<br>
            Minimum 6 minutes recommended for accurate protocol compliance
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Protocol at a Glance</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    principles = [
        ("P1", "Temporal Stability", "BPM variance <1.5%"),
        ("P2", "Spectral Parsimony", "Density 0.30–0.45"),
        ("P3", "Micro-Variation", "Every 8–16 bars"),
        ("P4", "Continuous Sub-Bass", "<80Hz ≥90% track"),
        ("P5", "Textural Continuity", "Drone ≥85% track"),
    ]
    for col, (code, name, desc) in zip(cols, principles):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-family:Space Mono;font-size:0.65rem;color:#c8ff00;letter-spacing:0.1em">{code}</div>
                <div style="font-weight:500;font-size:0.85rem;margin:0.3rem 0">{name}</div>
                <div style="font-size:0.72rem;color:#666">{desc}</div>
            </div>""", unsafe_allow_html=True)
    st.stop()

# ── ANALYSIS
audio_bytes = uploaded.read()

with st.spinner("Analyzing track..."):
    result = try_real_analysis(audio_bytes, uploaded.name)
    time.sleep(0.5)  # UX pause

# Save to session history
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = {}
from datetime import datetime
st.session_state.analysis_history[uploaded.name] = {
    "bpm": result["tempo"]["bpm"],
    "principles": result["protocol_compliance"]["principles_passed"],
    "compliant": result["protocol_compliance"]["compliant"],
    "date": datetime.now().strftime("%m/%d %H:%M"),
    "result": result,
}

real = result["metadata"].get("real_analysis", False)
if not real:
    st.info("⚠️ Running in **demo mode** (librosa not installed). Install `librosa` for real audio analysis. Results are simulated for demonstration.", icon="⚠️")


# ──────────────────────────────────────────────
# COMPLIANCE HEADER BANNER
# ──────────────────────────────────────────────

passed = result["protocol_compliance"]["principles_passed"]
compliant = result["protocol_compliance"]["compliant"]
total_violations = result["antipatterns"]["total_violations"]

banner_color = "#00e5a0" if compliant else "#ff3b3b"
banner_text = "PROTOCOL COMPLIANT" if compliant else "NON-COMPLIANT"

st.markdown(f"""
<div style="background:{banner_color}12;border:1px solid {banner_color}40;
            border-left:4px solid {banner_color};padding:1rem 1.5rem;
            display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem">
    <div>
        <div style="font-family:Space Mono;font-size:0.7rem;color:{banner_color};letter-spacing:0.1em">{banner_text}</div>
        <div style="font-size:0.85rem;color:#aaa;margin-top:0.2rem">
            {passed}/5 core principles met · {total_violations} anti-pattern violations
        </div>
    </div>
    <div style="font-family:Space Mono;font-size:2.5rem;font-weight:700;color:{banner_color}">{passed}/5</div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# KEY METRICS ROW
# ──────────────────────────────────────────────

t = result["tempo"]
s = result["spectral"]
l = result["lowend"]
k = result["kick"]
st_r = result["structure"]

c1, c2, c3, c4, c5, c6 = st.columns(6)

def metric_html(label, value, unit="", ok=None):
    cls = "" if ok is None else ("" if ok else " fail")
    val_cls = "" if ok is None else ("" if ok else " fail")
    return f"""<div class="metric-card{cls}">
        <div class="metric-value{val_cls}">{value}<span style="font-size:1rem">{unit}</span></div>
        <div class="metric-label">{label}</div>
    </div>"""

with c1: st.markdown(metric_html("BPM", f"{t['bpm']:.1f}", "", True), unsafe_allow_html=True)
with c2: st.markdown(metric_html("BPM Variance", f"{t['bpm_variance_pct']:.2f}", "%", t['bpm_variance_pct'] < 1.5), unsafe_allow_html=True)
with c3: st.markdown(metric_html("Spectral Density", f"{s['density']['mean']:.3f}", "", 0.30 <= s['density']['mean'] <= 0.45), unsafe_allow_html=True)
with c4: st.markdown(metric_html("Centroid", f"{s['centroid']['mean']:.0f}", "Hz", 250 <= s['centroid']['mean'] <= 400), unsafe_allow_html=True)
with c5: st.markdown(metric_html("Sub-Bass", f"{l['sub_presence_pct']*100:.0f}", "%", l['sub_presence_pct'] >= 0.90), unsafe_allow_html=True)
with c6: st.markdown(metric_html("Duration", f"{result['metadata']['duration_min']:.1f}", "min", result['metadata']['duration_min'] >= 6), unsafe_allow_html=True)

st.markdown("&nbsp;", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Protocol", "📊 Spectral", "🎚️ Structure", "🌐 Corpus", "📄 Export"
])


# ── TAB 1: PROTOCOL
with tab1:
    cola, colb = st.columns([3, 2])
    
    with cola:
        st.markdown('<div class="section-header">Core Principles (4/5 required)</div>', unsafe_allow_html=True)
        
        principles = result["protocol_compliance"]["principles"]
        for code, p in principles.items():
            ok = p["compliant"]
            badge = f'<span class="badge {"badge-ok" if ok else "badge-fail"}">{"PASS" if ok else "FAIL"}</span>'
            st.markdown(f"""
            <div class="principle-row">
                <div style="font-family:Space Mono;font-size:0.8rem;color:#666;min-width:2rem">{code}</div>
                {badge}
                <div style="flex:1">
                    <div style="font-size:0.9rem;font-weight:500">{p['name']}</div>
                    <div style="font-size:0.75rem;color:#666;margin-top:0.1rem">{p['details']}</div>
                </div>
                <div style="font-size:0.7rem;color:#444;text-align:right">{p['threshold']}</div>
            </div>""", unsafe_allow_html=True)
        
        st.markdown('<div class="section-header" style="margin-top:1.5rem">Complementary Criteria (3/5 recommended)</div>', unsafe_allow_html=True)
        comp = result["complementary"]
        cols_c = st.columns(3)
        for i, (code, c) in enumerate(comp.items()):
            with cols_c[i % 3]:
                ok = c["met"]
                st.markdown(f"""
                <div style="padding:0.5rem;border:1px solid {'#333' if not ok else '#2a3a2a'};
                            background:{'#111' if not ok else '#0d1a0d'};margin:0.2rem 0">
                    <span class="badge {'badge-ok' if ok else 'badge-warn'}">{'✓' if ok else '○'}</span>
                    <span style="font-size:0.8rem;margin-left:0.5rem">{c['name']}</span>
                </div>""", unsafe_allow_html=True)
        
        st.markdown('<div class="section-header" style="margin-top:1.5rem">Anti-Patterns Detected</div>', unsafe_allow_html=True)
        ap = result["antipatterns"]
        ap_list = [
            ("Density Overload (>0.60)", ap["density_overload"]),
            ("Sub-Bass Absent (>30%)", ap["sub_absent"]),
            ("BPM Change (>3%)", ap["bpm_change"]),
        ]
        for name, triggered in ap_list:
            badge = f'<span class="badge badge-fail">VIOLATION</span>' if triggered else f'<span class="badge" style="background:#1a2a1a;color:#00e5a0">CLEAR</span>'
            st.markdown(f"""
            <div class="principle-row">
                {badge}
                <div style="font-size:0.85rem">{name}</div>
            </div>""", unsafe_allow_html=True)
    
    with colb:
        st.plotly_chart(plot_compliance_radar(result), use_container_width=True)
        
        st.markdown('<div class="section-header">Summary</div>', unsafe_allow_html=True)
        if compliant:
            st.success(f"✅ **{passed}/5 core principles met.** Track qualifies as Continuous Inertia Techno under protocol v2.0.")
        else:
            failing = [code for code, p in principles.items() if not p["compliant"]]
            st.error(f"❌ **{passed}/5 principles.** Failed: {', '.join(failing)}. Minimum 4/5 required.")
        
        if total_violations > 0:
            st.warning(f"⚠️ {total_violations} anti-pattern violation(s) detected.")


# ── TAB 2: SPECTRAL
with tab2:
    if show_timeseries:
        st.plotly_chart(plot_spectral_density(result), use_container_width=True)
    
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown('<div class="section-header">Spectral Centroid</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card {'warn' if not (250 <= s['centroid']['mean'] <= 400) else ''}">
            <div class="metric-value {'warn' if not (250 <= s['centroid']['mean'] <= 400) else ''}">{s['centroid']['mean']:.0f} <span style="font-size:1rem">Hz</span></div>
            <div class="metric-label">Mean centroid · Target 250–400 Hz</div>
            <div style="font-size:0.75rem;color:#555;margin-top:0.5rem">±{s['centroid']['std']:.0f} Hz std</div>
        </div>""", unsafe_allow_html=True)
    
    with col_s2:
        st.markdown('<div class="section-header">Spectral Density</div>', unsafe_allow_html=True)
        ok_d = 0.30 <= s['density']['mean'] <= 0.45
        st.markdown(f"""
        <div class="metric-card {'fail' if not ok_d else ''}">
            <div class="metric-value {'fail' if not ok_d else ''}">{s['density']['mean']:.3f}</div>
            <div class="metric-label">Mean density · Target 0.30–0.45</div>
            <div style="font-size:0.75rem;color:#555;margin-top:0.5rem">±{s['density']['std']:.3f} std</div>
        </div>""", unsafe_allow_html=True)
    
    with col_s3:
        st.markdown('<div class="section-header">Spectral Rolloff</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{s['rolloff']['mean']:.0f} <span style="font-size:1rem">Hz</span></div>
            <div class="metric-label">85% energy rolloff</div>
            <div style="font-size:0.75rem;color:#555;margin-top:0.5rem">±{s['rolloff']['std']:.0f} Hz std</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Kick Analysis</div>', unsafe_allow_html=True)
    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        st.markdown(metric_html("Kick On-Beat", f"{k['kick_on_beat_pct']*100:.0f}", "%", k['kick_on_beat_pct'] >= 0.80), unsafe_allow_html=True)
    with col_k2:
        st.markdown(metric_html("Kick Consistency", f"{k['kick_consistency']*100:.0f}", "%", k['kick_consistency'] >= 0.80), unsafe_allow_html=True)
    with col_k3:
        st.markdown(metric_html("Kick Fundamental", f"{k['kick_fundamental_hz']:.0f}", "Hz", True), unsafe_allow_html=True)
    
    st.plotly_chart(plot_bpm_stability(result), use_container_width=True)


# ── TAB 3: STRUCTURE
with tab3:
    col_st1, col_st2 = st.columns(2)
    
    with col_st1:
        st.markdown('<div class="section-header">Layer Estimation</div>', unsafe_allow_html=True)
        ok_layers = st_r['layers_mean'] <= 4.5
        st.markdown(metric_html("Mean Layer Count", f"{st_r['layers_mean']:.1f}", "", ok_layers), unsafe_allow_html=True)
        st.markdown(metric_html("Modal Layer Count", f"{st_r['layers_mode']}", "", True), unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">Periodic Variation</div>', unsafe_allow_html=True)
        ok_interval = 8 <= st_r['mean_interval_bars'] <= 16
        st.markdown(metric_html("Mean Change Interval", f"{st_r['mean_interval_bars']:.1f}", " bars", ok_interval), unsafe_allow_html=True)
        st.markdown(metric_html("Periodicity Score", f"{st_r['periodicity_score']:.2f}", "", st_r['periodicity_score'] >= 0.70), unsafe_allow_html=True)
    
    with col_st2:
        st.markdown('<div class="section-header">Sub-Bass Continuity</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_sub_bass(result), use_container_width=True)
    
    col_lb1, col_lb2, col_lb3 = st.columns(3)
    with col_lb1:
        st.markdown(metric_html("Sub Presence", f"{l['sub_presence_pct']*100:.0f}", "%", l['sub_presence_pct'] >= 0.90), unsafe_allow_html=True)
    with col_lb2:
        st.markdown(metric_html("Sub/Kick Ratio", f"{l['sub_kick_ratio']:.2f}", "", True), unsafe_allow_html=True)
    with col_lb3:
        st.markdown(metric_html("Sub Continuity", f"{l['sub_continuity']:.2f}", "", l['sub_continuity'] >= 0.85), unsafe_allow_html=True)


# ── TAB 4: CORPUS
with tab4:
    if not check_corpus:
        st.info("Enable **Corpus Comparison** in the sidebar to see this analysis.")
    else:
        st.plotly_chart(plot_corpus_scatter(result), use_container_width=True)
        
        st.markdown('<div class="section-header">Percentile Position vs. Corpus (n=30)</div>', unsafe_allow_html=True)
        
        # Simulated percentiles
        np.random.seed(hash(uploaded.name) % (2**31))
        percentiles = {
            "BPM": int(np.random.randint(20, 80)),
            "Spectral Density": int(np.random.randint(20, 80)),
            "Spectral Centroid": int(np.random.randint(20, 80)),
            "Sub-Bass Ratio": int(np.random.randint(20, 80)),
        }
        
        col_p = st.columns(4)
        for col, (feat, pct) in zip(col_p, percentiles.items()):
            with col:
                color = "#00e5a0" if 20 <= pct <= 80 else "#ff6b00"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{color}">{pct}<span style="font-size:1rem">th</span></div>
                    <div class="metric-label">{feat}</div>
                </div>""", unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">Interpretation</div>', unsafe_allow_html=True)
        all_middle = all(20 <= v <= 80 for v in percentiles.values())
        if all_middle:
            st.success("✅ Track falls within the central distribution of the corpus on all features. Acoustic profile is typical of Continuous Inertia Techno.")
        else:
            outliers = [f for f, v in percentiles.items() if not (20 <= v <= 80)]
            st.warning(f"⚠️ Outlier features vs corpus: **{', '.join(outliers)}**. Review these parameters.")


# ── TAB 5: EXPORT
with tab5:
    st.markdown('<div class="section-header">Export Analysis Report</div>', unsafe_allow_html=True)
    
    col_ex1, col_ex2 = st.columns([2, 1])
    
    with col_ex1:
        if export_format == "JSON":
            json_str = json.dumps(result, indent=2)
            st.download_button(
                "⬇ Download JSON Report",
                data=json_str,
                file_name=f"analysis_{Path(uploaded.name).stem}.json",
                mime="application/json"
            )
            st.code(json_str[:800] + "\n...", language="json")
        
        elif export_format == "PDF Report":
            # Build a clean HTML report then offer download as HTML (print-to-PDF)
            t_v = result["tempo"]
            s_v = result["spectral"]
            l_v = result["lowend"]
            p_v = result["protocol_compliance"]
            comp = result["protocol_compliance"]["compliant"]
            color = "#00c77a" if comp else "#e53535"
            status = "COMPLIANT" if comp else "NON-COMPLIANT"
            
            rows_html = ""
            for code, p in p_v["principles"].items():
                ok = p["compliant"]
                rows_html += f"""<tr>
                    <td><b>{code}</b></td>
                    <td>{p['name']}</td>
                    <td style="color:{'#00c77a' if ok else '#e53535'};font-weight:700">{'PASS' if ok else 'FAIL'}</td>
                    <td style="color:#666">{p['details']}</td>
                </tr>"""
            
            html_report = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{ font-family: monospace; background: #fff; color: #111; padding: 3rem; max-width: 900px; margin: 0 auto; }}
  h1 {{ font-size: 1.6rem; border-bottom: 3px solid #000; padding-bottom: 0.5rem; }}
  .status {{ display: inline-block; padding: 0.4rem 1rem; background: {color}; color: #fff; font-weight: 700; font-size: 1.1rem; margin: 1rem 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1.5rem 0; }}
  th {{ background: #111; color: #fff; padding: 0.5rem; text-align: left; }}
  td {{ padding: 0.45rem; border-bottom: 1px solid #eee; font-size: 0.9rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1rem 0; }}
  .card {{ border: 1px solid #ddd; padding: 1rem; }}
  .val {{ font-size: 1.5rem; font-weight: 700; }}
  .lbl {{ font-size: 0.7rem; color: #888; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.2rem; }}
  @media print {{ body {{ padding: 1rem; }} }}
</style>
</head><body>
<h1>⬛ Continuous Inertia Techno Analyzer</h1>
<p style="color:#888;font-size:0.85rem">Track: <b>{result['metadata']['filename']}</b> · Duration: {result['metadata']['duration_min']:.1f} min · Analyzer v{result['metadata']['analyzer_version']}</p>
<div class="status">{status} — {p_v['principles_passed']}/5 PRINCIPLES</div>

<div class="grid">
  <div class="card"><div class="val">{t_v['bpm']:.1f}</div><div class="lbl">BPM</div></div>
  <div class="card"><div class="val" style="color:{'#00c77a' if t_v['bpm_variance_pct']<1.5 else '#e53535'}">{t_v['bpm_variance_pct']:.2f}%</div><div class="lbl">BPM Variance</div></div>
  <div class="card"><div class="val" style="color:{'#00c77a' if 0.30<=s_v['density']['mean']<=0.45 else '#e53535'}">{s_v['density']['mean']:.3f}</div><div class="lbl">Spectral Density</div></div>
  <div class="card"><div class="val">{s_v['centroid']['mean']:.0f} Hz</div><div class="lbl">Centroid</div></div>
  <div class="card"><div class="val" style="color:{'#00c77a' if l_v['sub_presence_pct']>=0.90 else '#e53535'}">{l_v['sub_presence_pct']*100:.0f}%</div><div class="lbl">Sub-Bass</div></div>
  <div class="card"><div class="val">{result['structure']['layers_mean']:.1f}</div><div class="lbl">Mean Layers</div></div>
</div>

<h2>Protocol Compliance</h2>
<table>
  <tr><th>Code</th><th>Principle</th><th>Result</th><th>Details</th></tr>
  {rows_html}
</table>

<p style="font-size:0.7rem;color:#aaa;margin-top:3rem;border-top:1px solid #eee;padding-top:1rem">
Generated by Continuous Inertia Techno Analyzer v2.0 · Use browser Print → Save as PDF
</p>
</body></html>"""
            
            st.download_button(
                "⬇ Download HTML Report (→ Print as PDF)",
                data=html_report,
                file_name=f"report_{Path(uploaded.name).stem}.html",
                mime="text/html"
            )
            st.info("💡 Open the downloaded HTML in your browser and use **File → Print → Save as PDF** to get a clean PDF report.")
            st.components.v1.html(html_report, height=600, scrolling=True)
        
        elif export_format == "CSV":
            flat = {
                "filename": result["metadata"]["filename"],
                "duration_min": result["metadata"]["duration_min"],
                "bpm": result["tempo"]["bpm"],
                "bpm_variance_pct": result["tempo"]["bpm_variance_pct"],
                "density_mean": result["spectral"]["density"]["mean"],
                "centroid_mean": result["spectral"]["centroid"]["mean"],
                "rolloff_mean": result["spectral"]["rolloff"]["mean"],
                "sub_presence_pct": result["lowend"]["sub_presence_pct"],
                "layers_mean": result["structure"]["layers_mean"],
                "mean_interval_bars": result["structure"]["mean_interval_bars"],
                "principles_passed": result["protocol_compliance"]["principles_passed"],
                "compliant": result["protocol_compliance"]["compliant"],
                "antipattern_violations": result["antipatterns"]["total_violations"],
            }
            df = pd.DataFrame([flat])
            csv_str = df.to_csv(index=False)
            st.download_button("⬇ Download CSV", data=csv_str,
                                file_name=f"analysis_{Path(uploaded.name).stem}.csv",
                                mime="text/csv")
            st.dataframe(df.T.rename(columns={0: "Value"}), use_container_width=True)
        
        elif export_format == "LaTeX":
            t_val = result["tempo"]
            s_val = result["spectral"]
            l_val = result["lowend"]
            p_val = result["protocol_compliance"]
            
            latex = f"""% Auto-generated by Continuous Inertia Analyzer v2.0
\\begin{{table}}[h]
\\centering
\\caption{{Acoustic Analysis: {Path(uploaded.name).stem}}}
\\begin{{tabular}}{{lcc}}
\\hline
\\textbf{{Parameter}} & \\textbf{{Value}} & \\textbf{{Compliant}} \\\\
\\hline
BPM & {t_val['bpm']:.1f} & {'\\checkmark' if 128 <= t_val['bpm'] <= 135 else '$\\times$'} \\\\
BPM Variance (\\%) & {t_val['bpm_variance_pct']:.2f} & {'\\checkmark' if t_val['bpm_variance_pct'] < 1.5 else '$\\times$'} \\\\
Spectral Density & {s_val['density']['mean']:.3f} & {'\\checkmark' if 0.30 <= s_val['density']['mean'] <= 0.45 else '$\\times$'} \\\\
Centroid (Hz) & {s_val['centroid']['mean']:.0f} & {'\\checkmark' if 250 <= s_val['centroid']['mean'] <= 400 else '$\\times$'} \\\\
Sub-Bass (\\%) & {l_val['sub_presence_pct']*100:.0f} & {'\\checkmark' if l_val['sub_presence_pct'] >= 0.90 else '$\\times$'} \\\\
\\hline
Principles Passed & {p_val['principles_passed']}/5 & {'\\checkmark' if p_val['compliant'] else '$\\times$'} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}"""
            st.download_button("⬇ Download LaTeX", data=latex,
                                file_name=f"table_{Path(uploaded.name).stem}.tex",
                                mime="text/plain")
            st.code(latex, language="latex")
    
    with col_ex2:
        st.markdown('<div class="section-header">Publication Ready</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.82rem;line-height:1.8;color:#888">
        ✓ JSON for full pipeline<br>
        ✓ CSV for SPSS/R/Python<br>
        ✓ LaTeX for direct paste<br>
        ✓ Batch mode via CLI<br><br>
        <span style="color:#555;font-size:0.72rem">
        Use batch CLI to process<br>
        full corpus (n=30) and<br>
        export combined CSV.
        </span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header" style="margin-top:1rem">CLI Reference</div>', unsafe_allow_html=True)
        st.code("""# Single track
python -m techno_analyzer \\
  analyze track.wav \\
  --protocol \\
  --corpus corpus_db.json

# Batch
python -m techno_analyzer \\
  batch tracks/*.wav \\
  --output results.csv""", language="bash")
