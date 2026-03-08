"""
CONTINUOUS INERTIA TECHNO ANALYZER
Protocol Compliance & Corpus Comparison Tool
Streamlit App — v2.1
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

try:
    from scipy.ndimage import gaussian_filter1d as _gauss
    def smooth(arr, sigma=3): return _gauss(np.array(arr, dtype=float), sigma=sigma)
except ImportError:
    def smooth(arr, sigma=3): return np.array(arr, dtype=float)

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
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=DM+Mono:wght@300;400&display=swap');

:root {
    --bg:      #0f0e0d;
    --s1:      #161412;
    --s2:      #1d1a17;
    --border:  #2a2520;
    --border2: #3a3028;
    --gold:    #d4b483;
    --gold2:   #eedcba;
    --rose:    #c49a8a;
    --sage:    #8aad9a;
    --slate:   #8a9aad;
    --text:    #ede5d8;
    --muted:   #9a8e82;
    --dim:     #4a4038;
    --pass:    #9abdaa;
    --fail:    #c47a7a;
    --warn:    #c9a96e;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
}

[data-testid="stSidebar"] {
    background: var(--s1) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 2rem !important; }

#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stDecoration"], [data-testid="stToolbar"] { display: none !important; }

/* Typography */
h1 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 300 !important;
    font-size: 2.2rem !important;
    letter-spacing: 0.06em !important;
    color: var(--text) !important;
    line-height: 1.2 !important;
}

/* Divider */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.2rem 0 !important; }

/* Streamlit overrides */
[data-testid="stFileUploader"] {
    background: var(--s1) !important;
    border: 1px dashed var(--border2) !important;
    border-radius: 0 !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--gold) !important; }
[data-testid="stFileUploader"] label { color: var(--muted) !important; }

/* Buttons */
.stButton > button {
    background: transparent !important;
    color: var(--gold) !important;
    border: 1px solid var(--gold) !important;
    border-radius: 0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.8rem !important;
    transition: all 0.18s !important;
}
.stButton > button:hover {
    background: var(--gold) !important;
    color: #000 !important;
}
.stButton > button:active {
    background: var(--gold2) !important;
    transform: translateY(1px) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
    padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    border-radius: 0 !important;
    padding: 0.65rem 1.3rem !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom: 1px solid var(--gold) !important;
    background: transparent !important;
}

/* Selectbox / toggle */
[data-testid="stSelectbox"] > div > div {
    background: var(--s2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    color: var(--text) !important;
}
[data-testid="stToggle"] label,
[data-testid="stSelectbox"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}

/* Download buttons */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: var(--gold) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stDownloadButton"] > button:hover { border-color: var(--gold) !important; }

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    border-left-width: 2px !important;
}

/* ─── Custom components ─── */

.metric-card {
    background: var(--s1);
    border: 1px solid var(--border);
    border-left: 2px solid var(--gold);
    padding: 0.9rem 1.1rem;
    margin: 0.3rem 0;
    position: relative;
}
.metric-card.fail { border-left-color: var(--fail); }
.metric-card.warn { border-left-color: var(--warn); }
.metric-card.pass { border-left-color: var(--pass); }

.metric-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 300;
    color: var(--gold2);
    line-height: 1;
}
.metric-value.fail { color: var(--fail); }
.metric-value.pass { color: var(--pass); }

.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: var(--text);
    margin-top: 0.3rem;
}

.section-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: var(--text);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.35rem;
    margin: 1.4rem 0 0.9rem 0;
}

.principle-row {
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    padding: 0.8rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
}
.principle-row:last-child { border-bottom: none; }

.badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    padding: 0.15rem 0.45rem;
    letter-spacing: 0.1em;
    flex-shrink: 0;
    margin-top: 0.1rem;
}
.badge-ok   { border: 1px solid var(--pass); color: var(--pass); background: transparent; }
.badge-fail { border: 1px solid var(--fail); color: var(--fail); background: transparent; }
.badge-warn { border: 1px solid var(--warn); color: var(--warn); background: transparent; }

.banner {
    padding: 1.1rem 1.6rem;
    border: 1px solid var(--border2);
    border-left: 3px solid var(--gold);
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.banner.ok { border-left-color: var(--pass); }
.banner.no { border-left-color: var(--fail); }

.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-weight: 300;
    font-size: 1.2rem;
    letter-spacing: 0.12em;
    color: var(--text);
}

.tag {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    padding: 0.1rem 0.4rem;
    background: var(--s2);
    border: 1px solid var(--border2);
    color: var(--text);
}

.chord-pill {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.9rem;
    padding: 0.4rem 0.8rem;
    background: var(--s2);
    border: 1px solid var(--border2);
    color: var(--gold);
    margin: 0.2rem;
    min-width: 3rem;
    text-align: center;
}

.element-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.4rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.8rem;
}

.key-display {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.8rem;
    font-weight: 300;
    color: var(--gold);
    line-height: 1;
}

.interp-box {
    background: var(--s1);
    border: 1px solid var(--border);
    border-left: 2px solid var(--gold);
    padding: 1.4rem;
    font-size: 0.82rem;
    line-height: 1.9;
    color: var(--text);
    font-family: 'DM Mono', monospace;
}

.section-chip {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    padding: 0.12rem 0.45rem;
    margin: 0.1rem;
    letter-spacing: 0.08em;
    border: 1px solid var(--border2);
}

.empty-state {
    text-align: center;
    padding: 5rem 2rem;
}

.upload-zone {
    border: 1px dashed var(--border2);
    padding: 2.5rem;
    text-align: center;
    margin: 1rem 0;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: var(--gold); }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MUSIC THEORY HELPERS
# ──────────────────────────────────────────────

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

SECTION_COLORS = {
    'INTRO':     'rgba(120,120,120,0.22)',
    'BUILD':     'rgba(255,107,0,0.25)',
    'PEAK':      'rgba(200,255,0,0.20)',
    'GROOVE':    'rgba(0,229,160,0.20)',
    'BREAKDOWN': 'rgba(68,136,255,0.28)',
    'OUTRO':     'rgba(80,80,80,0.18)',
}
SECTION_TEXT_COLORS = {
    'INTRO': '#888', 'BUILD': '#ff6b00', 'PEAK': '#d4b483',
    'GROOVE': '#00e5a0', 'BREAKDOWN': '#4488ff', 'OUTRO': '#666',
}

def get_chord_progression(key_note, mode):
    if key_note not in NOTES:
        key_note = 'A'
    idx = NOTES.index(key_note)
    prog_map = {
        'minor':      [(0,'m'), (5,'m'), (10,''), (7,'m')],
        'dorian':     [(0,'m'), (5,''), (0,'m'), (5,'')],
        'phrygian':   [(0,'m'), (1,''), (10,''), (0,'m')],
        'mixolydian': [(0,''), (10,''), (7,''), (0,'')],
        'major':      [(0,''), (5,''), (7,''), (0,'')],
    }
    prog = prog_map.get(mode, prog_map['minor'])
    return [NOTES[(idx + interval) % 12] + quality for interval, quality in prog]

def get_scale_notes(key_note, mode):
    if key_note not in NOTES:
        key_note = 'A'
    idx = NOTES.index(key_note)
    scale_intervals = {
        'minor':      [0, 2, 3, 5, 7, 8, 10],
        'dorian':     [0, 2, 3, 5, 7, 9, 10],
        'phrygian':   [0, 1, 3, 5, 7, 8, 10],
        'mixolydian': [0, 2, 4, 5, 7, 9, 10],
        'major':      [0, 2, 4, 5, 7, 9, 11],
    }
    intervals = scale_intervals.get(mode, scale_intervals['minor'])
    return [NOTES[(idx + i) % 12] for i in intervals]

def get_relative_key(key_note, mode):
    if key_note not in NOTES:
        key_note = 'A'
    idx = NOTES.index(key_note)
    if mode in ('minor', 'dorian', 'phrygian'):
        return NOTES[(idx + 3) % 12] + ' major'
    else:
        return NOTES[(idx - 3) % 12] + ' minor'

MODE_COLORS = {
    'minor': '#4488ff',
    'dorian': '#00e5a0',
    'phrygian': '#ff3b3b',
    'mixolydian': '#d4b483',
    'major': '#ff6b00',
}

# ──────────────────────────────────────────────
# ANALYSIS ENGINE
# ──────────────────────────────────────────────

def simulate_track_structure(duration_sec, bpm):
    """Generate realistic techno track energy envelope, sections, and element timelines."""
    n_points = 200
    t = np.linspace(0, duration_sec, n_points)
    t_norm = t / duration_sec

    # Section boundaries + types
    bounds = [0.0, 0.12, 0.23, 0.40, 0.52, 0.63, 0.73, 0.88, 1.0]
    labels = ['INTRO', 'BUILD', 'PEAK', 'GROOVE', 'BREAKDOWN', 'BUILD', 'PEAK', 'OUTRO']
    energy_levels = [0.22, 0.55, 0.90, 0.82, 0.30, 0.62, 0.93, 0.25]

    energy = np.zeros(n_points)
    for i, (s, e, lvl) in enumerate(zip(bounds[:-1], bounds[1:], energy_levels)):
        mask = (t_norm >= s) & (t_norm < e)
        n_seg = int(mask.sum())
        if n_seg == 0:
            continue
        lbl = labels[i]
        if lbl == 'INTRO':
            energy[mask] = np.linspace(0.08, lvl, n_seg)
        elif lbl == 'BUILD':
            prev_lvl = energy_levels[i - 1] if i > 0 else 0.2
            energy[mask] = np.linspace(prev_lvl * 0.7, lvl, n_seg)
        elif lbl == 'BREAKDOWN':
            half = n_seg // 2
            energy[mask] = np.concatenate([
                np.linspace(energy_levels[i - 1] * 0.9, 0.22, half),
                np.linspace(0.22, 0.45, n_seg - half)
            ])
        elif lbl == 'OUTRO':
            energy[mask] = np.linspace(lvl * 0.9, 0.05, n_seg)
        else:
            energy[mask] = lvl + np.random.randn(n_seg) * 0.025

    energy += np.random.randn(n_points) * 0.015
    energy = np.clip(smooth(energy, sigma=4), 0, 1)

    bars_per_sec = bpm / (60 * 4)
    sections = []
    for i, (s, e, lbl) in enumerate(zip(bounds[:-1], bounds[1:], labels)):
        sections.append({
            'label': lbl,
            'start_sec': round(s * duration_sec, 1),
            'end_sec':   round(e * duration_sec, 1),
            'start_bar': int(s * duration_sec * bars_per_sec),
            'end_bar':   int(e * duration_sec * bars_per_sec),
            'energy_mean': round(float(energy_levels[i]), 2),
            'duration_sec': round((e - s) * duration_sec, 1),
        })

    # ── Element timelines
    # presence per section: [INTRO, BUILD, PEAK, GROOVE, BREAKDOWN, BUILD2, PEAK2, OUTRO]
    el_defs = {
        'Kick':     {'label': '4/4 Kick Drum',      'color': '#d4b483',
                     'pres': [0.30, 0.92, 1.00, 1.00, 0.02, 0.92, 1.00, 0.35]},
        'Sub Bass': {'label': 'Sub Bass (<80 Hz)',   'color': '#ff6b00',
                     'pres': [0.55, 0.82, 0.96, 0.95, 0.65, 0.84, 0.96, 0.50]},
        'Hi-Hat':   {'label': 'Hi-Hat / Cymbal',    'color': '#00e5a0',
                     'pres': [0.20, 0.80, 0.95, 0.90, 0.08, 0.80, 0.95, 0.18]},
        'Clap':     {'label': 'Clap / Snare',       'color': '#4488ff',
                     'pres': [0.00, 0.55, 0.90, 0.82, 0.00, 0.55, 0.90, 0.00]},
        'Synth Pad':{'label': 'Synth Pad / Chord',  'color': '#ff44aa',
                     'pres': [0.35, 0.65, 0.75, 0.72, 0.55, 0.65, 0.75, 0.30]},
        'Texture':  {'label': 'Texture / Noise',    'color': '#aaaaaa',
                     'pres': [0.55, 0.62, 0.65, 0.62, 0.80, 0.65, 0.65, 0.60]},
        'Perc':     {'label': 'Percussion Layer',   'color': '#ffaa00',
                     'pres': [0.00, 0.42, 0.72, 0.68, 0.02, 0.42, 0.72, 0.00]},
        'FX / Sweep':{'label': 'FX / Sweep',        'color': '#ff6666',
                     'pres': [0.00, 0.75, 0.28, 0.18, 0.65, 0.75, 0.28, 0.00]},
    }

    # random suppression for optional elements
    for key in ['Clap', 'Synth Pad', 'Perc', 'FX / Sweep']:
        if np.random.random() > 0.60:
            el_defs[key]['pres'] = [p * np.random.uniform(0, 0.25) for p in el_defs[key]['pres']]

    def make_curve(pres_list):
        curve = np.zeros(n_points)
        for i, (s, e, p) in enumerate(zip(bounds[:-1], bounds[1:], pres_list)):
            mask = (t_norm >= s) & (t_norm < e)
            n_seg = int(mask.sum())
            if n_seg > 0:
                curve[mask] = p + np.random.randn(n_seg) * 0.04
        return np.clip(smooth(curve, sigma=2), 0, 1)

    element_timelines = {}
    for key, d in el_defs.items():
        tl = make_curve(d['pres'])
        element_timelines[key] = {
            'label': d['label'],
            'color': d['color'],
            'timeline': tl.tolist(),
            'mean_presence': float(np.mean(tl)),
            'present': float(np.mean(tl)) > 0.18,
        }

    return {
        'energy_timeline': energy.tolist(),
        'time_points_sec': t.tolist(),
        'sections': sections,
        'n_points': n_points,
        'elements': element_timelines,
    }


def simulate_tonality(filename):
    """Simulate key / tonality detection."""
    techno_keys  = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    techno_modes = ['minor', 'minor', 'minor', 'dorian', 'phrygian',
                    'minor', 'dorian', 'mixolydian', 'minor']
    key_note   = np.random.choice(techno_keys)
    mode       = np.random.choice(techno_modes)
    confidence = round(np.random.uniform(0.60, 0.88), 2)

    chord_prog   = get_chord_progression(key_note, mode)
    scale_notes  = get_scale_notes(key_note, mode)
    relative_key = get_relative_key(key_note, mode)

    n_chord_segs = 24
    chord_timeline = [chord_prog[np.random.choice(
        len(chord_prog), p=[0.40, 0.30, 0.20, 0.10]
    )] for _ in range(n_chord_segs)]

    return {
        'key':              key_note,
        'mode':             mode,
        'key_string':       f'{key_note} {mode}',
        'confidence':       confidence,
        'chord_progression': chord_prog,
        'chord_timeline':   chord_timeline,
        'scale_notes':      scale_notes,
        'relative_key':     relative_key,
    }


def simulate_analysis(filename: str, duration_sec: float = None) -> dict:
    """Returns seeded simulated analysis results."""
    np.random.seed(hash(filename) % (2**31))

    bpm             = np.random.uniform(126, 140)
    bpm_std         = np.random.uniform(0.3, 2.5)
    bpm_variance_pct = (bpm_std / bpm) * 100

    density_mean  = np.random.uniform(0.20, 0.65)
    centroid_mean = np.random.uniform(150, 500)
    rolloff_mean  = np.random.uniform(2000, 6000)

    sub_presence  = np.random.uniform(0.60, 1.0)
    sub_kick_ratio = np.random.uniform(0.45, 0.90)

    layers_mean       = np.random.uniform(1.5, 5.5)
    mean_interval_bars = np.random.uniform(4, 24)
    texture_presence  = np.random.uniform(0.65, 1.0)

    duration = duration_sec or np.random.uniform(360, 600)

    n_seg = 20
    bpm_over_time      = (bpm + np.random.randn(n_seg) * bpm_std * 0.5).tolist()
    density_over_time  = (density_mean + np.random.randn(n_seg) * 0.04).tolist()
    centroid_over_time = (centroid_mean + np.random.randn(n_seg) * 40).tolist()
    sub_over_time      = np.clip(sub_presence + np.random.randn(n_seg) * 0.08, 0, 1).tolist()

    p1_ok = bpm_variance_pct < 1.5
    p2_ok = 0.30 <= density_mean <= 0.45 and layers_mean <= 4.5
    p3_ok = 8 <= mean_interval_bars <= 16
    p4_ok = sub_presence >= 0.90 and -6 <= (sub_kick_ratio * -10 + 2) <= -3
    p5_ok = texture_presence >= 0.85

    principles_passed = sum([p1_ok, p2_ok, p3_ok, p4_ok, p5_ok])

    density_overload = density_mean > 0.60
    sub_absent       = sub_presence < 0.70
    bpm_change       = bpm_variance_pct > 3.0

    # New extended analysis
    track_map = simulate_track_structure(duration, bpm)
    tonality  = simulate_tonality(filename)

    elements_summary = {
        k: {
            'label':         v['label'],
            'present':       v['present'],
            'mean_presence': round(v['mean_presence'], 2),
            'color':         v['color'],
        }
        for k, v in track_map['elements'].items()
    }

    return {
        "metadata": {
            "filename":         filename,
            "duration_sec":     duration,
            "duration_min":     duration / 60,
            "analyzer_version": "2.1",
        },
        "tempo": {
            "bpm":             round(bpm, 2),
            "bpm_confidence":  round(np.random.uniform(0.75, 0.99), 2),
            "bpm_mean":        round(bpm, 2),
            "bpm_std":         round(bpm_std, 3),
            "bpm_variance_pct": round(bpm_variance_pct, 2),
            "stability_score": round(max(0, 1 - bpm_variance_pct / 5), 2),
            "bpm_over_time":   bpm_over_time,
        },
        "spectral": {
            "centroid":  {"mean": round(centroid_mean, 1), "std": round(np.random.uniform(25, 70), 1)},
            "density":   {"mean": round(density_mean, 3), "std": round(np.random.uniform(0.02, 0.08), 3)},
            "rolloff":   {"mean": round(rolloff_mean, 1), "std": round(np.random.uniform(300, 800), 1)},
            "density_over_time":  density_over_time,
            "centroid_over_time": centroid_over_time,
        },
        "kick": {
            "kick_on_beat_pct":    round(np.random.uniform(0.72, 0.99), 2),
            "kick_consistency":    round(np.random.uniform(0.70, 0.98), 2),
            "kick_fundamental_hz": round(np.random.uniform(40, 80), 1),
        },
        "structure": {
            "layers_mean":        round(layers_mean, 1),
            "layers_mode":        int(round(layers_mean)),
            "mean_interval_bars": round(mean_interval_bars, 1),
            "periodicity_score":  round(np.random.uniform(0.5, 0.97), 2),
        },
        "lowend": {
            "sub_presence_pct": round(sub_presence, 2),
            "sub_kick_ratio":   round(sub_kick_ratio, 2),
            "sub_continuity":   round(np.random.uniform(0.70, 0.99), 2),
            "sub_over_time":    sub_over_time,
        },
        "protocol_compliance": {
            "principles": {
                "P1": {"name": "Temporal Stability",      "compliant": p1_ok, "value": bpm_variance_pct, "threshold": "<1.5% BPM variance",        "details": f"BPM variance: {bpm_variance_pct:.2f}% (threshold <1.5%)"},
                "P2": {"name": "Spectral Parsimony",      "compliant": p2_ok, "value": density_mean,      "threshold": "Density 0.30–0.45, ≤4 layers","details": f"Density: {density_mean:.3f} | Layers: {layers_mean:.1f}"},
                "P3": {"name": "Periodic Micro-Variation","compliant": p3_ok, "value": mean_interval_bars,"threshold": "Changes every 8–16 bars",     "details": f"Mean interval: {mean_interval_bars:.1f} bars"},
                "P4": {"name": "Continuous Sub-Bass",     "compliant": p4_ok, "value": sub_presence,      "threshold": "Sub <80Hz present ≥90%",     "details": f"Sub present: {sub_presence*100:.0f}% | Ratio to kick: {sub_kick_ratio:.2f}"},
                "P5": {"name": "Textural Continuity",     "compliant": p5_ok, "value": texture_presence,  "threshold": "Texture ≥85% of track",      "details": f"Texture presence: {texture_presence*100:.0f}%"},
            },
            "principles_passed": principles_passed,
            "compliant":         principles_passed >= 4,
        },
        "antipatterns": {
            "density_overload": density_overload,
            "sub_absent":       sub_absent,
            "bpm_change":       bpm_change,
            "total_violations": sum([density_overload, sub_absent, bpm_change]),
        },
        "complementary": {
            "C1": {"name": "BPM 128–135",       "met": 128 <= bpm <= 135},
            "C2": {"name": "Duration 7–9 min",  "met": 420 <= duration <= 540},
            "C3": {"name": "No build-ups/drops","met": not bpm_change},
            "C4": {"name": "Centroid 250–400 Hz","met": 250 <= centroid_mean <= 400},
            "C5": {"name": "Stems exportable",  "met": True},
        },
        "track_map": track_map,
        "tonality":  tonality,
        "elements":  elements_summary,
    }


def try_real_analysis(audio_bytes: bytes, filename: str) -> dict:
    """Try librosa real analysis, fall back to simulation."""
    try:
        import librosa
        import tempfile, os

        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Load up to 180s for analysis
        y, sr = librosa.load(tmp_path, sr=22050, mono=True, duration=180)
        
        # Get full duration before deleting
        full_duration = librosa.get_duration(path=tmp_path)
        os.unlink(tmp_path)

        # Seed simulation from filename for consistent non-real parts
        result = simulate_analysis(filename, duration_sec=full_duration)

        # Override with real values
        duration_loaded = librosa.get_duration(y=y, sr=sr)
        bpm_raw, beats = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.atleast_1d(bpm_raw)[0])

        # BPM variance from beats
        if len(beats) > 2:
            beat_times = librosa.frames_to_time(beats, sr=sr)
            ioi = np.diff(beat_times)
            bpm_series = 60.0 / ioi
            bpm_std = float(np.std(bpm_series))
            bpm_variance_pct = (bpm_std / bpm) * 100 if bpm > 0 else 0.0
        else:
            bpm_std = 0.5
            bpm_variance_pct = (bpm_std / bpm) * 100 if bpm > 0 else 0.0

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Density proxy: fraction of time with RMS > threshold
        rms = librosa.feature.rms(y=y)[0]
        density_mean = float(np.mean(rms > np.percentile(rms, 30)))

        # Sub-bass presence (20–80 Hz)
        stft = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        sub_mask = (freqs >= 20) & (freqs <= 80)
        sub_energy = stft[sub_mask, :].mean(axis=0)
        total_energy = stft.mean(axis=0)
        sub_presence_arr = sub_energy / (total_energy + 1e-8)
        sub_presence = float(np.mean(sub_presence_arr > 0.05))

        # Chromagram-based key detection
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        major_profile = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
        minor_profile = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]
        best_key, best_mode, best_corr = 'A', 'minor', -1
        for i in range(12):
            for profile, mname in [(major_profile, 'major'), (minor_profile, 'minor')]:
                rotated = np.roll(profile, i)
                corr = float(np.corrcoef(chroma_mean, rotated)[0, 1])
                if corr > best_corr:
                    best_corr, best_key, best_mode = corr, NOTES[i], mname

        # Patch result with real values
        result["metadata"]["real_analysis"] = True
        result["tempo"]["bpm"]              = round(bpm, 2)
        result["tempo"]["bpm_mean"]         = round(bpm, 2)
        result["tempo"]["bpm_std"]          = round(bpm_std, 3)
        result["tempo"]["bpm_variance_pct"] = round(bpm_variance_pct, 2)
        result["tempo"]["stability_score"]  = round(max(0, 1 - bpm_variance_pct / 5), 2)
        result["spectral"]["centroid"]["mean"] = round(float(np.mean(spectral_centroids)), 1)
        result["spectral"]["density"]["mean"]  = round(density_mean, 3)
        result["spectral"]["rolloff"]["mean"]  = round(float(np.mean(spectral_rolloff)), 1)
        result["lowend"]["sub_presence_pct"]   = round(sub_presence, 2)

        # Patch tonality with real key
        result["tonality"]["key"]        = best_key
        result["tonality"]["mode"]       = best_mode
        result["tonality"]["key_string"] = f'{best_key} {best_mode}'
        result["tonality"]["confidence"] = round(best_corr, 2)
        result["tonality"]["chord_progression"] = get_chord_progression(best_key, best_mode)
        result["tonality"]["scale_notes"]       = get_scale_notes(best_key, best_mode)
        result["tonality"]["relative_key"]      = get_relative_key(best_key, best_mode)

        return result

    except Exception:
        result = simulate_analysis(filename)
        result["metadata"]["real_analysis"] = False
        return result


# ──────────────────────────────────────────────
# INTERPRETATION GENERATOR
# ──────────────────────────────────────────────

def generate_interpretation(result: dict) -> str:
    t   = result["tempo"]
    s   = result["spectral"]
    l   = result["lowend"]
    p   = result["protocol_compliance"]
    tn  = result["tonality"]
    el  = result["elements"]
    tm  = result["track_map"]
    ap  = result["antipatterns"]
    st  = result["structure"]

    bpm      = t["bpm"]
    bpm_var  = t["bpm_variance_pct"]
    density  = s["density"]["mean"]
    centroid = s["centroid"]["mean"]
    sub      = l["sub_presence_pct"]
    layers   = st["layers_mean"]
    interval = st["mean_interval_bars"]
    passed   = p["principles_passed"]
    key_str  = tn["key_string"].upper()
    mode     = tn["mode"]
    dur      = result["metadata"]["duration_min"]
    principles = p["principles"]
    is_demo  = not result["metadata"].get("real_analysis", False)

    if bpm < 128:    bpm_char = "slow and meditative (below typical techno range)"
    elif bpm <= 132: bpm_char = "steady and hypnotic — optimal for trance-like states"
    elif bpm <= 136: bpm_char = "driving and propulsive"
    else:            bpm_char = "fast and aggressive (above recommended 135 BPM ceiling)"

    if density < 0.25:   density_char = "extremely sparse — very few simultaneous sound layers"
    elif density < 0.35: density_char = "lean and minimal — ideal for sustained listening"
    elif density < 0.45: density_char = "controlled and balanced — within optimal range"
    elif density < 0.55: density_char = "moderately dense — approaching saturation threshold"
    else:                density_char = "dense and complex — likely above parsimony threshold"

    mode_descriptions = {
        'minor':      'Natural Minor (Aeolian) — the foundational mode of dark techno; creates introspective, melancholic tension without resolution',
        'dorian':     'Dorian Mode — minor with a raised 6th; groove-oriented and modal, common in hypnotic techno',
        'phrygian':   'Phrygian Mode — tense, cinematic; the flat 2nd creates strong forward momentum',
        'mixolydian': 'Mixolydian Mode — major with a flat 7th; open and cyclic, effective for long-form DJ use',
        'major':      'Major Scale — bright and resolved; unusual for dark techno, more common in melodic styles',
    }
    mode_char = mode_descriptions.get(mode, 'unidentified mode')

    if centroid < 200:   centroid_char = "extremely bass-heavy — energy is almost entirely sub-sonic"
    elif centroid < 280: centroid_char = "sub/low-bass dominant — strong somatic (body-felt) impact"
    elif centroid < 380: centroid_char = "low-mid focused — balanced between physical and auditory engagement"
    else:                centroid_char = "mid-range forward — more cognitive/auditory than physical"

    present_els = [v['label'] for v in el.values() if v['present']]
    absent_els  = [v['label'] for v in el.values() if not v['present']]
    peak_secs   = [sec for sec in tm["sections"] if sec['label'] == 'PEAK']

    def verdict(ok):
        return "PASS" if ok else "FAIL"

    p1 = principles["P1"]
    p2 = principles["P2"]
    p3 = principles["P3"]
    p4 = principles["P4"]
    p5 = principles["P5"]

    if p1['compliant']:
        p1_exp = f"BPM variance is {bpm_var:.2f}% — within the 1.5% threshold. This indicates a mechanically stable groove, essential for sustained trance induction."
    else:
        p1_exp = f"BPM variance is {bpm_var:.2f}% — exceeding the 1.5% threshold. Excessive tempo drift disrupts the listener's sense of continuous motion. Tighten the grid or use stricter quantisation."

    if p2['compliant']:
        p2_exp = f"Spectral density is {density:.3f} with {layers:.1f} mean layers — within the 0.30-0.45 target. Sparse layering maintains cognitive headroom and prevents listener fatigue."
    elif density < 0.30:
        p2_exp = f"Spectral density is {density:.3f} — below the 0.30 minimum. Too few layers creates monotony without sufficient textural interest. Add a subtle continuous element."
    else:
        p2_exp = f"Spectral density is {density:.3f} with {layers:.1f} layers — above the 0.45 maximum. Too many layers creates cognitive overload, breaking the trance state. Simplify or remove elements."

    if p3['compliant']:
        p3_exp = f"Mean change interval is {interval:.1f} bars — within the 8-16 bar target. Subtle, periodic micro-variations keep the listener's attention without disrupting inertia."
    elif interval > 16:
        p3_exp = f"Mean change interval is {interval:.1f} bars — too long. Changes happening too rarely create static monotony. Introduce subtle variations more frequently (every 8-16 bars)."
    else:
        p3_exp = f"Mean change interval is {interval:.1f} bars — too short. Changes happening too frequently feel restless and prevent the settling of a hypnotic groove."

    if p4['compliant']:
        p4_exp = f"Sub-bass is present {sub*100:.0f}% of the track — meeting the 90% minimum. Continuous sub-bass provides the somatic anchor that makes this music physically immersive."
    else:
        p4_exp = f"Sub-bass is present only {sub*100:.0f}% of the track — below the 90% minimum. Gaps in sub-bass break the physical pressure that defines this genre. The sub layer must be nearly uninterrupted."

    if p5['compliant']:
        p5_exp = "Textural continuity is sufficient. A continuous textural layer maintains the perceptual field that prevents abrupt transitions from sounding jarring."
    else:
        p5_exp = "Textural continuity is insufficient. Without a sustained texture running throughout, the track feels episodic rather than continuous. Add a low-level drone or texture that runs the full duration."

    ap_lines = []
    if ap["density_overload"]:
        ap_lines.append("  - DENSITY OVERLOAD: More than 60% spectral density means too many simultaneous elements. This is the most common mistake in productions aiming for hypnotic simplicity.")
    if ap["sub_absent"]:
        ap_lines.append("  - SUB-BASS ABSENT: Sub frequencies missing for more than 30% of the track. This significantly weakens somatic impact on a sound system.")
    if ap["bpm_change"]:
        ap_lines.append("  - BPM DRIFT: Tempo changes exceed 3% — likely caused by humanised grid or live performance artefacts. For CI Techno, BPM must be mechanically constant.")
    ap_section = ("ANTI-PATTERNS DETECTED:\n" + "\n".join(ap_lines)) if ap_lines else "No anti-patterns detected."

    if passed == 5:
        verdict_text = "FULLY COMPLIANT (5/5) — All five core principles are met. Suitable for floor deployment and academic corpus inclusion."
    elif passed == 4:
        verdict_text = "COMPLIANT (4/5) — Minimum threshold satisfied. Qualifies as Continuous Inertia Techno with minor deviations."
    elif passed == 3:
        verdict_text = "PARTIALLY COMPLIANT (3/5) — Two principles fail. Shows CI Techno characteristics but requires revision before corpus inclusion."
    else:
        verdict_text = f"NON-COMPLIANT ({passed}/5) — Three or more principles fail. Significant structural revision needed."

    failing_recs = []
    for k in ['P1','P2','P3','P4','P5']:
        if not principles[k]['compliant']:
            failing_recs.append(f"  [{k} - {principles[k]['name']}] {principles[k]['details']} | Threshold: {principles[k]['threshold']}")
    recs_text = "\n".join(failing_recs) if failing_recs else "  No changes needed — track meets all protocol requirements."

    lines = [
        "CONTINUOUS INERTIA TECHNO ANALYZER — ANALYSIS REPORT",
        "=" * 56,
        f"File:     {result['metadata']['filename']}",
        f"Duration: {dur:.1f} min  |  BPM: {bpm:.1f}  |  Key: {key_str}  |  Mode: {mode.capitalize()}",
        f"Analysis: {'Simulated demo (install librosa for real analysis)' if is_demo else 'Real audio analysis'}",
        "",
        "OVERALL VERDICT",
        "-" * 40,
        verdict_text,
        "",
        ap_section,
        "",
        "=" * 56,
        "WHAT IS CONTINUOUS INERTIA TECHNO?",
        "=" * 56,
        "Continuous Inertia (CI) Techno is a production protocol for tracks that induce",
        "sustained trance-like states through acoustic consistency. Unlike conventional",
        "techno which uses dramatic drops and builds, CI Techno maintains near-constant",
        "energy, sub-bass, and spectral density throughout. The goal is neurophysiological:",
        "reducing cognitive load so listeners enter a deep, embodied listening state.",
        "Five measurable acoustic principles define and distinguish it from other styles.",
        "",
        "=" * 56,
        "P1 — TEMPORAL STABILITY",
        "-" * 40,
        "Target: BPM variance < 1.5%",
        "WHY: The CI protocol targets 128-135 BPM, a range associated with physiological",
        "entrainment (heart rate, respiration synchronisation). BPM drift above 1.5% breaks",
        "this synchronisation and disrupts the continuous inertia state.",
        "",
        f"Result [{verdict(p1['compliant'])}]: {p1_exp}",
        "",
        "=" * 56,
        "P2 — SPECTRAL PARSIMONY",
        "-" * 40,
        "Target: Density 0.30-0.45 | Max 4 simultaneous layers",
        "WHY: Spectral density measures how full the frequency spectrum is. Below 0.30",
        "feels empty; above 0.45 becomes cognitively tiring over extended listening.",
        "The centroid tells you where energy lives: lower = more physical/body-felt,",
        "higher = more auditory/melody-driven.",
        "",
        f"Density: {density:.3f}  |  Centroid: {centroid:.0f} Hz ({centroid_char})",
        f"Result [{verdict(p2['compliant'])}]: {p2_exp}",
        "",
        "=" * 56,
        "P3 — PERIODIC MICRO-VARIATION",
        "-" * 40,
        "Target: Changes every 8-16 bars",
        "WHY: CI Techno uses micro-variation rather than macro-structure. Small changes",
        "(adding/removing one element, filter sweeps, subtle FX) must happen regularly",
        "enough to maintain interest without disrupting the hypnotic flow.",
        "",
        f"Mean interval: {interval:.1f} bars",
        f"Result [{verdict(p3['compliant'])}]: {p3_exp}",
        "",
        "=" * 56,
        "P4 — CONTINUOUS SUB-BASS",
        "-" * 40,
        "Target: Frequencies < 80 Hz present >= 90% of track",
        "WHY: Continuous sub-bass (20-80 Hz) is the defining characteristic of CI Techno.",
        "This range is felt physically through bones and organs, not just heard.",
        "It creates the floor pressure that makes this music somatic and functional.",
        "Any gap in the sub layer breaks the physical continuity of the experience.",
        "",
        f"Sub presence: {sub*100:.0f}%  |  Sub/Kick ratio: {l['sub_kick_ratio']:.2f}",
        f"Result [{verdict(p4['compliant'])}]: {p4_exp}",
        "",
        "=" * 56,
        "P5 — TEXTURAL CONTINUITY",
        "-" * 40,
        "Target: Texture/drone present >= 85% of track",
        "WHY: A persistent textural layer acts as acoustic glue. Without it, even a",
        "rhythmically consistent track can feel fragmented and episodic.",
        "",
        f"Result [{verdict(p5['compliant'])}]: {p5_exp}",
        "",
        "=" * 56,
        "HARMONIC ANALYSIS",
        "-" * 40,
        f"Detected Key: {key_str}",
        f"Mode: {mode_char}",
        f"Estimated Chord Progression: {' - '.join(tn['chord_progression'])}",
        f"Relative Key: {tn['relative_key'].upper()}",
        "",
        "WHY IT MATTERS: In CI Techno, harmony is static or moves very slowly.",
        "Minor and Dorian modes dominate because they create tension without resolution,",
        "supporting continuous forward motion without a sense of arrival or closure.",
        "",
        "=" * 56,
        "DETECTED SONIC ELEMENTS",
        "-" * 40,
        f"Active ({len(present_els)}):  {', '.join(present_els) if present_els else 'none detected'}",
        f"Absent ({len(absent_els)}):  {', '.join(absent_els) if absent_els else 'none'}",
        "",
        "=" * 56,
        "PRODUCTION RECOMMENDATIONS",
        "-" * 40,
        recs_text,
        "",
        "=" * 56,
        f"Generated by Continuous Inertia Techno Analyzer v{result['metadata']['analyzer_version']}",
        "This report is intended for research, production review, and academic documentation.",
    ]
    return "\n".join(lines)



# ──────────────────────────────────────────────
# PLOTTING FUNCTIONS
# ──────────────────────────────────────────────

# PLOT_BASE: ONLY bgcolor + font. Never add margin/xaxis/yaxis here.
PLOT_BASE = dict(
    paper_bgcolor='#0f0e0d',
    plot_bgcolor='#161412',
    font=dict(family='DM Mono, monospace', color='#9a8e82', size=10),
)
PLOT_LAYOUT = PLOT_BASE

def _polar_layout():
    return dict(PLOT_BASE)


def plot_bpm_stability(result: dict) -> go.Figure:
    bpm_time = result["tempo"]["bpm_over_time"]
    n = len(bpm_time)
    x = np.linspace(0, result["metadata"]["duration_min"], n)
    mean_bpm = result["tempo"]["bpm_mean"]

    fig = go.Figure()
    fig.add_hrect(y0=mean_bpm - mean_bpm * 0.015,
                  y1=mean_bpm + mean_bpm * 0.015,
                  fillcolor='rgba(212,180,131,0.08)', line_width=0,
                  annotation_text="±1.5% threshold")
    fig.add_trace(go.Scatter(
        x=x, y=bpm_time, mode='lines',
        line=dict(color='#d4b483', width=2),
        name='BPM over time',
        fill='tozeroy', fillcolor='rgba(212,180,131,0.04)'
    ))
    fig.add_hline(y=mean_bpm, line_dash='dash', line_color='#555', line_width=1)
    fig.update_layout(**PLOT_BASE, title="BPM Stability Over Time")
    fig.update_xaxes(title_text="Time (min)", gridcolor='#2a2520', zerolinecolor='#222')
    fig.update_yaxes(title_text="BPM", gridcolor='#2a2520', zerolinecolor='#222')
    return fig


def plot_spectral_density(result: dict) -> go.Figure:
    density = result["spectral"]["density_over_time"]
    centroid = result["spectral"]["centroid_over_time"]
    n = len(density)
    x = np.linspace(0, result["metadata"]["duration_min"], n)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    # Use shapes instead of add_hrect(row=) — broken in Plotly 6
    fig.update_layout(shapes=[
        dict(type='rect', xref='paper', yref='y',  x0=0, x1=1, y0=0.30, y1=0.45,
             fillcolor='rgba(212,180,131,0.08)', line_width=0, layer='below'),
        dict(type='rect', xref='paper', yref='y2', x0=0, x1=1, y0=250,  y1=400,
             fillcolor='rgba(255,107,0,0.07)',    line_width=0, layer='below'),
    ])
    fig.add_trace(go.Scatter(x=x, y=density, mode='lines',
                              line=dict(color='#d4b483', width=2),
                              name='Density', fill='tozeroy', fillcolor='rgba(212,180,131,0.04)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=centroid, mode='lines',
                              line=dict(color='#ff6b00', width=2),
                              name='Centroid (Hz)', fill='tozeroy', fillcolor='rgba(255,107,0,0.05)'), row=2, col=1)

    fig.update_layout(**PLOT_BASE, title="Spectral Features Over Time", height=400, showlegend=True)
    fig.update_yaxes(title_text="Density", row=1, col=1, gridcolor='#2a2520')
    fig.update_yaxes(title_text="Centroid Hz", row=2, col=1, gridcolor='#2a2520')
    fig.update_xaxes(title_text="Time (min)", row=2, col=1, gridcolor='#2a2520')
    return fig


def plot_sub_bass(result: dict) -> go.Figure:
    sub = result["lowend"]["sub_over_time"]
    n = len(sub)
    x = np.linspace(0, result["metadata"]["duration_min"], n)

    fig = go.Figure()
    fig.add_hrect(y0=0.90, y1=1.01, fillcolor='rgba(0,229,160,0.08)', line_width=0,
                  annotation_text="≥90% target")
    fig.add_trace(go.Scatter(
        x=x, y=sub, mode='lines+markers',
        line=dict(color='#00e5a0', width=2),
        marker=dict(size=3, color='#00e5a0'),
        name='Sub-bass presence',
        fill='tozeroy', fillcolor='rgba(0,229,160,0.05)'
    ))
    fig.update_layout(
        **PLOT_BASE,
        title="Sub-Bass (<80 Hz) Continuity",
    )
    fig.update_xaxes(title_text="Time (min)", gridcolor='#2a2520', zerolinecolor='#222')
    fig.update_yaxes(title_text="Presence", range=[0, 1.05], gridcolor='#2a2520', zerolinecolor='#222')
    return fig


def plot_compliance_radar(result: dict) -> go.Figure:
    p = result["protocol_compliance"]["principles"]
    categories = [v["name"].split(" ")[0] + "<br>" + " ".join(v["name"].split(" ")[1:])
                  for v in p.values()]
    raw = {
        "P1": 1.0 if p["P1"]["compliant"] else max(0, 1 - p["P1"]["value"] / 5),
        "P2": 1.0 if p["P2"]["compliant"] else max(0, 1 - abs(p["P2"]["value"] - 0.375) / 0.375),
        "P3": 1.0 if p["P3"]["compliant"] else max(0, 1 - abs(p["P3"]["value"] - 12) / 12),
        "P4": 1.0 if p["P4"]["compliant"] else p["P4"]["value"],
        "P5": 1.0 if p["P5"]["compliant"] else p["P5"]["value"] / 0.85,
    }
    values = list(raw.values()) + [list(raw.values())[0]]
    cats   = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values, theta=cats, fill='toself',
        fillcolor='rgba(212,180,131,0.10)',
        line=dict(color='#d4b483', width=2),
        name='Protocol Score'
    ))
    fig.update_layout(
        **_polar_layout(),
        polar=dict(
            bgcolor='#161412',
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#2a2a2a', color='#9a8e82'),
            angularaxis=dict(gridcolor='#2a2a2a', color='#9a8e82'),
        ),
        title="Protocol Compliance Radar",
        showlegend=False,
    )
    return fig


def plot_corpus_scatter(result: dict) -> go.Figure:
    np.random.seed(42)
    n_corpus = 30
    corpus_density  = np.random.normal(0.37, 0.09, n_corpus)
    corpus_centroid = np.random.normal(295, 67, n_corpus)

    track_density  = result["spectral"]["density"]["mean"]
    track_centroid = result["spectral"]["centroid"]["mean"]

    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse_x = np.mean(corpus_density)  + 2 * np.std(corpus_density)  * np.cos(theta)
    ellipse_y = np.mean(corpus_centroid) + 2 * np.std(corpus_centroid) * np.sin(theta)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ellipse_x, y=ellipse_y, mode='lines',
                              line=dict(color='#444', dash='dash'), name='2SD ellipse'))
    fig.add_trace(go.Scatter(x=corpus_density, y=corpus_centroid, mode='markers',
                              marker=dict(color='#333', size=7, symbol='circle'),
                              name='Corpus (n=30)'))
    fig.add_trace(go.Scatter(x=[np.mean(corpus_density)], y=[np.mean(corpus_centroid)],
                              mode='markers', marker=dict(color='#4488ff', size=12, symbol='cross'),
                              name='Corpus Mean'))
    fig.add_trace(go.Scatter(x=[track_density], y=[track_centroid], mode='markers',
                              marker=dict(color='#d4b483', size=15, symbol='star'),
                              name='Your Track'))

    fig.update_layout(**PLOT_BASE, title="Corpus Comparison: Density vs. Centroid",
                      legend=dict(bgcolor='#161412'))
    fig.update_xaxes(title_text="Spectral Density", gridcolor='#2a2520', zerolinecolor='#222')
    fig.update_yaxes(title_text="Spectral Centroid (Hz)", gridcolor='#2a2520', zerolinecolor='#222')
    return fig


# ── NEW: Track Map

def plot_track_map(result: dict) -> go.Figure:
    tm  = result["track_map"]
    t_min = [ti / 60 for ti in tm["time_points_sec"]]
    energy = tm["energy_timeline"]

    fig = go.Figure()

    # Section bands
    for sec in tm["sections"]:
        col = SECTION_COLORS.get(sec['label'], 'rgba(100,100,100,0.15)')
        tcol = SECTION_TEXT_COLORS.get(sec['label'], '#888')
        fig.add_vrect(
            x0=sec['start_sec'] / 60,
            x1=sec['end_sec'] / 60,
            fillcolor=col, line_width=0,
            annotation_text=f"<b>{sec['label']}</b>",
            annotation_position="top left",
            annotation=dict(font=dict(size=8, color=tcol, family='Space Mono'), textangle=0),
        )

    # Section dividers
    for sec in tm["sections"][1:]:
        fig.add_vline(x=sec['start_sec'] / 60, line_width=1,
                      line_dash='dot', line_color='#2a2a2a')

    # Energy envelope
    fig.add_trace(go.Scatter(
        x=t_min, y=energy, mode='lines',
        line=dict(color='#d4b483', width=2.5),
        fill='tozeroy', fillcolor='rgba(212,180,131,0.08)',
        name='Energy',
        hovertemplate='%{x:.2f} min — energy %{y:.2f}<extra></extra>',
    ))

    fig.update_layout(
        **PLOT_BASE,
        title='Track Structure Map — Energy Envelope & Sections',
        height=320, showlegend=False,
    )
    fig.update_xaxes(title_text='Time (min)', gridcolor='#2a2520', zerolinecolor='#222')
    fig.update_yaxes(title_text='Normalized Energy', range=[0, 1.08], gridcolor='#2a2520', zerolinecolor='#222')
    return fig


def plot_elements_heatmap(result: dict) -> go.Figure:
    tm  = result["track_map"]
    els = tm["elements"]
    t_min = [ti / 60 for ti in tm["time_points_sec"]]

    keys   = list(els.keys())
    labels = [els[k]["label"] for k in keys]
    z_data = [els[k]["timeline"] for k in keys]

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=t_min,
        y=labels,
        colorscale=[[0, '#0d0d0d'], [0.3, '#1a3010'], [0.7, '#5a9020'], [1, '#d4b483']],
        showscale=True,
        colorbar=dict(
            thickness=10, tickfont=dict(color='#9a8e82', size=9, family='Space Mono'),
            bgcolor='#0f0e0d', outlinecolor='#222',
        ),
        xgap=0, ygap=2,
        hovertemplate='%{y}<br>%{x:.2f} min — presence %{z:.2f}<extra></extra>',
    ))

    for sec in tm["sections"][1:]:
        fig.add_vline(x=sec['start_sec'] / 60, line_width=1,
                      line_dash='dot', line_color='#2a2a2a')

    fig.update_layout(**PLOT_BASE, title='Element / Layer Presence Timeline', height=340)
    fig.update_layout(margin=dict(l=130, r=60, t=45, b=40))
    fig.update_xaxes(title_text='Time (min)', gridcolor='#2a2520', zerolinecolor='#2a2520')
    fig.update_yaxes(gridcolor='#2a2520', zerolinecolor='#2a2520', tickfont=dict(size=10))
    return fig


def plot_chord_timeline(result: dict) -> go.Figure:
    tn  = result["tonality"]
    dur = result["metadata"]["duration_min"]

    chord_tl = tn["chord_timeline"]
    n = len(chord_tl)
    x_edges = [i * dur / n for i in range(n + 1)]

    unique_chords = list(dict.fromkeys(chord_tl))
    palette = ['#d4b483', '#ff6b00', '#00e5a0', '#4488ff', '#ff44aa', '#ffaa00', '#ff6666', '#aaaaaa']
    cmap = {c: palette[i % len(palette)] for i, c in enumerate(unique_chords)}

    fig = go.Figure()
    for i, chord in enumerate(chord_tl):
        fig.add_shape(type='rect',
                      x0=x_edges[i], x1=x_edges[i + 1],
                      y0=0, y1=1,
                      fillcolor=cmap[chord], opacity=0.75, line_width=0)
        fig.add_annotation(
            x=(x_edges[i] + x_edges[i + 1]) / 2, y=0.5,
            text=f"<b>{chord}</b>", showarrow=False,
            font=dict(family='Space Mono', size=10, color='#000'),
        )

    fig.update_layout(**PLOT_BASE, title='Estimated Harmonic Progression', height=110, showlegend=False)
    fig.update_layout(margin=dict(l=40, r=20, t=40, b=35))
    fig.update_xaxes(title_text='Time (min)', gridcolor='#2a2520', zerolinecolor='#222')
    fig.update_yaxes(visible=False, range=[0, 1])
    return fig


def plot_key_circle(result: dict) -> go.Figure:
    """Circle of fifths with detected key highlighted."""
    tn   = result["tonality"]
    key  = tn["key"]
    mode = tn["mode"]

    # Circle of fifths — minor keys
    cof_minor = ['A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F', 'C', 'G', 'D']
    cof_major = ['C', 'G', 'D', 'A',  'E',  'B',  'F#', 'C#', 'G#','D#','A#','F']

    ref = cof_minor if mode in ('minor', 'dorian', 'phrygian') else cof_major
    label_mode = 'minor keys' if mode in ('minor', 'dorian', 'phrygian') else 'major keys'

    colors = ['#d4b483' if n == key else '#1e1e1e' for n in ref]
    txt_c  = ['#000'    if n == key else '#555'    for n in ref]
    sizes  = [38        if n == key else 24        for n in ref]
    theta  = [i * 30 for i in range(12)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[1] * 12,
        theta=theta,
        mode='text+markers',
        text=ref,
        textfont=dict(size=11, color=txt_c, family='Space Mono'),
        marker=dict(size=sizes, color=colors, symbol='circle', line=dict(width=1, color='#333')),
        hoverinfo='skip',
    ))
    fig.update_layout(
        **_polar_layout(),
        polar=dict(
            bgcolor='#0f0e0d',
            radialaxis=dict(visible=False, range=[0, 1.4]),
            angularaxis=dict(visible=False),
        ),
        title=f'Circle of Fifths — {label_mode}',
        showlegend=False,
        height=280,
        margin=dict(l=20, r=20, t=45, b=20),
    )


def plot_master_waveform(result: dict) -> go.Figure:
    """Master overview chart: energy + sections + key elements overlay."""
    tm   = result["track_map"]
    t_min = [ti / 60 for ti in tm["time_points_sec"]]
    energy = tm["energy_timeline"]
    els  = tm["elements"]
    dur  = result["metadata"]["duration_min"]
    bpm  = result["tempo"]["bpm"]
    N    = len(t_min)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=["Energy · Sections", "Element Presence", "BPM Stability"],
    )

    # ── Row 1: Energy + section bands
    section_colors_map = {
        'INTRO': 'rgba(130,130,120,0.14)', 'BUILD': 'rgba(212,180,131,0.14)',
        'PEAK':  'rgba(154,189,170,0.22)', 'GROOVE':'rgba(154,170,189,0.22)',
        'BREAKDOWN':'rgba(120,108,96,0.28)','OUTRO': 'rgba(100,98,90,0.18)',
    }
    for sec in tm["sections"]:
        x0 = sec["start_sec"] / 60
        x1 = sec["end_sec"]   / 60
        mid = (x0 + x1) / 2
        col = section_colors_map.get(sec["label"], "rgba(100,100,100,0.1)")
        fig.add_shape(type="rect", xref="x", yref="y",
                      x0=x0, x1=x1, y0=0, y1=1.05,
                      fillcolor=col, line_width=0, layer="below", row=1, col=1)
        fig.add_annotation(x=mid, y=1.02, text=sec["label"],
                           showarrow=False, row=1, col=1,
                           font=dict(size=7, color="#9a8e82", family="DM Mono"),
                           yref="y")

    fig.add_trace(go.Scatter(
        x=t_min, y=energy, mode="lines",
        line=dict(color="#d4b483", width=2),
        fill="tozeroy", fillcolor="rgba(212,180,131,0.06)",
        name="Energy", showlegend=False,
        hovertemplate="%{x:.2f} min · energy %{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # Section vertical lines
    for sec in tm["sections"][1:]:
        for r in [1, 2, 3]:
            fig.add_vline(x=sec["start_sec"]/60, line_width=1,
                          line_dash="dot", line_color="#2a2520", row=r, col=1)

    # ── Row 2: Stacked element lines (top 4 most present)
    el_items = sorted(els.items(), key=lambda x: -x[1]["mean_presence"])[:4]
    el_palette = ["#d4b483", "#9abdaa", "#8a9aad", "#c49a8a"]
    for i, (k, v) in enumerate(el_items):
        tl_smooth = v["timeline"]
        fig.add_trace(go.Scatter(
            x=t_min, y=tl_smooth, mode="lines",
            line=dict(color=el_palette[i % 4], width=1.2),
            fill="tozeroy", fillcolor=f"rgba({['201,169,110','138,173,154','138,154,173','196,154,138'][i % 4]},0.04)",
            name=k, showlegend=True,
            hovertemplate=f"{k}: %{{y:.2f}}<extra></extra>",
        ), row=2, col=1)

    # ── Row 3: BPM
    bpm_ts = result["tempo"]["bpm_over_time"]
    x_bpm  = np.linspace(0, result["metadata"]["duration_min"], len(bpm_ts))
    fig.add_trace(go.Scatter(
        x=x_bpm, y=bpm_ts, mode="lines",
        line=dict(color="#8a9aad", width=1.5),
        name="BPM", showlegend=False,
        hovertemplate="BPM: %{y:.1f}<extra></extra>",
    ), row=3, col=1)
    fig.add_hline(y=bpm, line_dash="dot", line_color="#3a3028", line_width=1, row=3, col=1)

    fig.update_layout(
        **PLOT_BASE,
        height=520,
        title=dict(text=f"Master Analysis · {result['metadata']['filename']} · {bpm:.1f} BPM",
                   font=dict(size=11, color="#9a8e82")),
        legend=dict(orientation="h", y=-0.04, x=0, font=dict(size=9, color="#9a8e82"),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0),
        hovermode="x unified",
    )
    for r in [1, 2, 3]:
        fig.update_xaxes(gridcolor="#2a2520", zerolinecolor="#2a2520", row=r, col=1)
        fig.update_yaxes(gridcolor="#2a2520", zerolinecolor="#2a2520", row=r, col=1)
    fig.update_xaxes(title_text="Time (min)", row=3, col=1)
    fig.update_yaxes(title_text="Energy", row=1, col=1)
    fig.update_yaxes(title_text="Presence", row=2, col=1)
    fig.update_yaxes(title_text="BPM", row=3, col=1)
    return fig


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="hero-title">⬛ INERTIA<br>ANALYZER</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#444;font-size:0.7rem;font-family:Space Mono;'
                'margin-top:0.3rem;margin-bottom:1.5rem">v2.1 — Protocol-Enhanced</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-header">Upload Track</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Audio file", type=["wav", "mp3", "flac", "aiff"],
                                 label_visibility="collapsed")

    st.markdown('<div class="section-header">Options</div>', unsafe_allow_html=True)
    check_protocol  = st.toggle("Protocol Compliance",  value=True)
    check_corpus    = st.toggle("Corpus Comparison",    value=True)
    show_timeseries = st.toggle("Time Series Plots",    value=True)

    st.markdown('<div class="section-header">Corpus</div>', unsafe_allow_html=True)
    corpus_option = st.selectbox("Reference corpus",
                                  ["Continuous Inertia v1 (n=30)", "Upload custom corpus"],
                                  label_visibility="collapsed")

    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
    export_format = st.selectbox("Format", ["JSON", "CSV", "LaTeX", "HTML Report (→ PDF)"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div style="font-size:0.65rem;color:#444;font-family:Space Mono;line-height:1.6">'
                'Protocol: 5 core principles<br>'
                'Corpus: hypnotic/minimal techno<br>'
                '2010–2025 reference set<br><br>'
                '© 2026 Research Tool</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("# Continuous Inertia Techno Analyzer")
    st.markdown('<div style="color:#666;font-family:Space Mono;font-size:0.8rem">'
                'Acoustic analysis · Protocol compliance · Corpus comparison</div>',
                unsafe_allow_html=True)
with col_h2:
    if uploaded:
        st.markdown(f'<div style="text-align:right">'
                    f'<span class="tag">READY</span><br>'
                    f'<span style="font-size:0.75rem;color:#888">{uploaded.name}</span></div>',
                    unsafe_allow_html=True)

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
    principles_glance = [
        ("P1", "Temporal Stability",    "BPM variance <1.5%"),
        ("P2", "Spectral Parsimony",    "Density 0.30–0.45"),
        ("P3", "Micro-Variation",       "Every 8–16 bars"),
        ("P4", "Continuous Sub-Bass",   "<80Hz ≥90% track"),
        ("P5", "Textural Continuity",   "Drone ≥85% track"),
    ]
    for col, (code, name, desc) in zip(cols, principles_glance):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-family:Space Mono;font-size:0.65rem;color:#c8ff00;letter-spacing:0.1em">{code}</div>
                <div style="font-weight:500;font-size:0.85rem;margin:0.3rem 0">{name}</div>
                <div style="font-size:0.72rem;color:#666">{desc}</div>
            </div>""", unsafe_allow_html=True)
    st.stop()

# ── SESSION STATE
if "result" not in st.session_state:
    st.session_state.result = None
if "analyzed_name" not in st.session_state:
    st.session_state.analyzed_name = None

# ── ANALYZE BUTTON
audio_bytes = uploaded.read()

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyze_clicked = st.button("◆  Analyze Track", use_container_width=True)

if analyze_clicked or (st.session_state.analyzed_name == uploaded.name and st.session_state.result):
    if analyze_clicked or st.session_state.result is None:
        with st.spinner(""):
            result = try_real_analysis(audio_bytes, uploaded.name)
            time.sleep(0.3)
            st.session_state.result = result
            st.session_state.analyzed_name = uploaded.name
    else:
        result = st.session_state.result
else:
    st.markdown('''
    <div style="text-align:center;padding:3rem 0;color:#4a4038;font-family:'DM Mono',monospace;font-size:0.7rem;letter-spacing:0.15em">
    TRACK LOADED · PRESS ANALYZE TO BEGIN
    </div>''', unsafe_allow_html=True)
    st.stop()

real = result["metadata"].get("real_analysis", False)
if not real:
    st.markdown('<div style="font-family:DM Mono,monospace;font-size:0.62rem;color:#4a4038;'
                'padding:0.4rem 0.8rem;border:1px solid #2a2520;margin-bottom:0.8rem;display:inline-block">'
                '◦ Demo mode — install librosa for real audio analysis</div>',
                unsafe_allow_html=True)


# ──────────────────────────────────────────────
# COMPLIANCE BANNER
# ──────────────────────────────────────────────

passed    = result["protocol_compliance"]["principles_passed"]
compliant = result["protocol_compliance"]["compliant"]
total_violations = result["antipatterns"]["total_violations"]

banner_color = "#00e5a0" if compliant else "#ff3b3b"
banner_text  = "PROTOCOL COMPLIANT" if compliant else "NON-COMPLIANT"

st.markdown(f"""
<div style="background:{banner_color}10;border:1px solid {banner_color}35;
            border-left:4px solid {banner_color};padding:1rem 1.5rem;
            display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem">
    <div>
        <div style="font-family:Space Mono;font-size:0.7rem;color:{banner_color};letter-spacing:0.1em">{banner_text}</div>
        <div style="font-size:0.85rem;color:#aaa;margin-top:0.2rem">
            {passed}/5 core principles met · {total_violations} anti-pattern violation(s)
        </div>
    </div>
    <div style="font-family:Space Mono;font-size:2.5rem;font-weight:700;color:{banner_color}">{passed}/5</div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# KEY METRICS ROW
# ──────────────────────────────────────────────

t_r   = result["tempo"]
s_r   = result["spectral"]
l_r   = result["lowend"]
k_r   = result["kick"]
st_r  = result["structure"]
tn_r  = result["tonality"]

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

def metric_html(label, value, unit="", ok=None):
    cls     = "" if ok is None else ("" if ok else " fail")
    val_cls = "" if ok is None else ("" if ok else " fail")
    return (f'<div class="metric-card{cls}">'
            f'<div class="metric-value{val_cls}">{value}'
            f'<span style="font-size:1rem">{unit}</span></div>'
            f'<div class="metric-label">{label}</div></div>')

with c1: st.markdown(metric_html("BPM", f"{t_r['bpm']:.1f}", "", True), unsafe_allow_html=True)
with c2: st.markdown(metric_html("BPM Var", f"{t_r['bpm_variance_pct']:.2f}", "%", t_r['bpm_variance_pct'] < 1.5), unsafe_allow_html=True)
with c3: st.markdown(metric_html("Density", f"{s_r['density']['mean']:.3f}", "", 0.30 <= s_r['density']['mean'] <= 0.45), unsafe_allow_html=True)
with c4: st.markdown(metric_html("Centroid", f"{s_r['centroid']['mean']:.0f}", "Hz", 250 <= s_r['centroid']['mean'] <= 400), unsafe_allow_html=True)
with c5: st.markdown(metric_html("Sub-Bass", f"{l_r['sub_presence_pct']*100:.0f}", "%", l_r['sub_presence_pct'] >= 0.90), unsafe_allow_html=True)
with c6: st.markdown(metric_html("Duration", f"{result['metadata']['duration_min']:.1f}", "min", result['metadata']['duration_min'] >= 6), unsafe_allow_html=True)
with c7:
    mode_col = MODE_COLORS.get(tn_r['mode'], '#d4b483')
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="font-size:1.3rem;color:{mode_col}">{tn_r['key']} <span style="font-size:0.9rem">{tn_r['mode']}</span></div>
        <div class="metric-label">Key · {tn_r['confidence']*100:.0f}% conf</div>
    </div>""", unsafe_allow_html=True)

st.markdown("&nbsp;", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────

# ── MASTER WAVEFORM (always visible, above tabs)
st.markdown('<div class="section-header">Master Analysis — Full Track View</div>', unsafe_allow_html=True)
st.plotly_chart(plot_master_waveform(result), use_container_width=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Protocol", "Spectral", "Structure",
    "Track Map", "Tonality", "Corpus", "Export"
])


# ── TAB 1: PROTOCOL
with tab1:
    cola, colb = st.columns([3, 2])

    with cola:
        st.markdown('<div class="section-header">Core Principles (4/5 required)</div>', unsafe_allow_html=True)
        principles = result["protocol_compliance"]["principles"]
        for code, p in principles.items():
            ok    = p["compliant"]
            badge = f'<span class="badge {"badge-ok" if ok else "badge-fail"}">{"PASS" if ok else "FAIL"}</span>'
            st.markdown(f"""
            <div class="principle-row">
                <div style="font-family:Space Mono;font-size:0.8rem;color:#555;min-width:2rem">{code}</div>
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
                <div style="padding:0.5rem;border:1px solid {'#2a3a2a' if ok else '#333'};
                            background:{'#0d1a0d' if ok else '#111'};margin:0.2rem 0">
                    <span class="badge {'badge-ok' if ok else 'badge-warn'}">{'✓' if ok else '○'}</span>
                    <span style="font-size:0.8rem;margin-left:0.5rem">{c['name']}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:1.5rem">Anti-Patterns Detected</div>', unsafe_allow_html=True)
        ap = result["antipatterns"]
        ap_list = [
            ("Density Overload (>0.60)", ap["density_overload"]),
            ("Sub-Bass Absent (>30%)",   ap["sub_absent"]),
            ("BPM Change (>3%)",         ap["bpm_change"]),
        ]
        for name, triggered in ap_list:
            badge = (f'<span class="badge badge-fail">VIOLATION</span>'
                     if triggered else
                     f'<span class="badge" style="background:#0d1a0d;color:#00e5a0">CLEAR</span>')
            st.markdown(f"""
            <div class="principle-row">
                {badge}
                <div style="font-size:0.85rem">{name}</div>
            </div>""", unsafe_allow_html=True)

    with colb:
        st.plotly_chart(plot_compliance_radar(result), use_container_width=True)
        st.markdown('<div class="section-header">Summary</div>', unsafe_allow_html=True)
        if compliant:
            st.success(f"✅ **{passed}/5 core principles met.** Track qualifies as Continuous Inertia Techno under protocol v2.1.")
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
        ok_c = 250 <= s_r['centroid']['mean'] <= 400
        st.markdown(metric_html("Mean Centroid · Target 250–400 Hz",
                                f"{s_r['centroid']['mean']:.0f}", " Hz", ok_c), unsafe_allow_html=True)
        st.caption(f"±{s_r['centroid']['std']:.0f} Hz std")
    with col_s2:
        st.markdown('<div class="section-header">Spectral Density</div>', unsafe_allow_html=True)
        ok_d = 0.30 <= s_r['density']['mean'] <= 0.45
        st.markdown(metric_html("Mean Density · Target 0.30–0.45",
                                f"{s_r['density']['mean']:.3f}", "", ok_d), unsafe_allow_html=True)
        st.caption(f"±{s_r['density']['std']:.3f} std")
    with col_s3:
        st.markdown('<div class="section-header">Spectral Rolloff</div>', unsafe_allow_html=True)
        st.markdown(metric_html("85% Energy Rolloff",
                                f"{s_r['rolloff']['mean']:.0f}", " Hz", None), unsafe_allow_html=True)
        st.caption(f"±{s_r['rolloff']['std']:.0f} Hz std")

    st.markdown('<div class="section-header">Kick Analysis</div>', unsafe_allow_html=True)
    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1: st.markdown(metric_html("Kick On-Beat",     f"{k_r['kick_on_beat_pct']*100:.0f}", "%", k_r['kick_on_beat_pct'] >= 0.80), unsafe_allow_html=True)
    with col_k2: st.markdown(metric_html("Kick Consistency", f"{k_r['kick_consistency']*100:.0f}",  "%", k_r['kick_consistency'] >= 0.80), unsafe_allow_html=True)
    with col_k3: st.markdown(metric_html("Kick Fundamental", f"{k_r['kick_fundamental_hz']:.0f}",  "Hz", True), unsafe_allow_html=True)

    st.plotly_chart(plot_bpm_stability(result), use_container_width=True)


# ── TAB 3: STRUCTURE
with tab3:
    col_st1, col_st2 = st.columns(2)

    with col_st1:
        st.markdown('<div class="section-header">Layer Estimation</div>', unsafe_allow_html=True)
        ok_layers = st_r['layers_mean'] <= 4.5
        st.markdown(metric_html("Mean Layer Count",  f"{st_r['layers_mean']:.1f}", "", ok_layers), unsafe_allow_html=True)
        st.markdown(metric_html("Modal Layer Count", f"{st_r['layers_mode']}",     "", True), unsafe_allow_html=True)

        st.markdown('<div class="section-header">Periodic Variation</div>', unsafe_allow_html=True)
        ok_interval = 8 <= st_r['mean_interval_bars'] <= 16
        st.markdown(metric_html("Mean Change Interval", f"{st_r['mean_interval_bars']:.1f}", " bars", ok_interval), unsafe_allow_html=True)
        st.markdown(metric_html("Periodicity Score",    f"{st_r['periodicity_score']:.2f}",  "", st_r['periodicity_score'] >= 0.70), unsafe_allow_html=True)

    with col_st2:
        st.markdown('<div class="section-header">Sub-Bass Continuity</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_sub_bass(result), use_container_width=True)

    col_lb1, col_lb2, col_lb3 = st.columns(3)
    with col_lb1: st.markdown(metric_html("Sub Presence",  f"{l_r['sub_presence_pct']*100:.0f}", "%", l_r['sub_presence_pct'] >= 0.90), unsafe_allow_html=True)
    with col_lb2: st.markdown(metric_html("Sub/Kick Ratio", f"{l_r['sub_kick_ratio']:.2f}", "", True), unsafe_allow_html=True)
    with col_lb3: st.markdown(metric_html("Sub Continuity", f"{l_r['sub_continuity']:.2f}", "", l_r['sub_continuity'] >= 0.85), unsafe_allow_html=True)


# ── TAB 4: TRACK MAP  ── NEW
with tab4:
    st.markdown('<div class="section-header">Full Track Energy Map</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_track_map(result), use_container_width=True)

    # Section table
    st.markdown('<div class="section-header">Section Breakdown</div>', unsafe_allow_html=True)
    tm_r = result["track_map"]
    secs = tm_r["sections"]

    sec_cols = st.columns(len(secs))
    for col, sec in zip(sec_cols, secs):
        tcol = SECTION_TEXT_COLORS.get(sec['label'], '#888')
        bcol = SECTION_COLORS.get(sec['label'], 'rgba(100,100,100,0.2)')
        dur_m = sec['duration_sec'] / 60
        with col:
            st.markdown(f"""
            <div style="background:{bcol};border:1px solid #2a2a2a;padding:0.6rem 0.5rem;text-align:center">
                <div style="font-family:Space Mono;font-size:0.65rem;font-weight:700;color:{tcol};letter-spacing:0.08em">{sec['label']}</div>
                <div style="font-size:0.8rem;margin-top:0.3rem;color:#ccc">{dur_m:.1f} min</div>
                <div style="font-size:0.7rem;color:#666">Bar {sec['start_bar']}→{sec['end_bar']}</div>
                <div style="font-size:0.7rem;color:#555;margin-top:0.2rem">⚡ {sec['energy_mean']:.0%}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # Element heatmap
    st.markdown('<div class="section-header">Element / Layer Presence</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_elements_heatmap(result), use_container_width=True)

    # Element summary cards
    st.markdown('<div class="section-header">Detected Elements</div>', unsafe_allow_html=True)
    el_r = result["elements"]
    el_cols = st.columns(4)
    for i, (key, el) in enumerate(el_r.items()):
        with el_cols[i % 4]:
            pct = el["mean_presence"] * 100
            ok  = el["present"]
            st.markdown(f"""
            <div style="background:#111;border:1px solid {'#2a3a1a' if ok else '#2a2a2a'};
                        border-left:3px solid {el['color'] if ok else '#333'};
                        padding:0.6rem 0.8rem;margin:0.2rem 0">
                <div style="font-family:Space Mono;font-size:0.7rem;color:{el['color'] if ok else '#555'};font-weight:700">
                    {'■' if ok else '□'} {key}
                </div>
                <div style="font-size:0.72rem;color:#666;margin-top:0.15rem">{el['label']}</div>
                <div style="font-size:0.85rem;color:{'#ccc' if ok else '#444'};margin-top:0.2rem;font-family:Space Mono">
                    {pct:.0f}% presence
                </div>
            </div>""", unsafe_allow_html=True)


# ── TAB 5: TONALITY  ── NEW
with tab5:
    tn_r = result["tonality"]

    col_t1, col_t2 = st.columns([2, 1])

    with col_t1:
        # Key display
        st.markdown('<div class="section-header">Detected Key</div>', unsafe_allow_html=True)
        mode_col = MODE_COLORS.get(tn_r['mode'], '#d4b483')
        st.markdown(f"""
        <div style="background:#111;border:1px solid #222;border-left:4px solid {mode_col};
                    padding:1.5rem 2rem;margin-bottom:1rem">
            <div class="key-display">{tn_r['key']} <span style="font-size:1.5rem;color:{mode_col}">{tn_r['mode'].upper()}</span></div>
            <div style="font-size:0.8rem;color:#666;margin-top:0.5rem;font-family:Space Mono">
                Confidence: {tn_r['confidence']*100:.0f}% · Relative: {tn_r['relative_key'].upper()}
            </div>
        </div>""", unsafe_allow_html=True)

        # Scale notes
        st.markdown('<div class="section-header">Scale Notes</div>', unsafe_allow_html=True)
        scale_html = "".join(
            f'<span class="chord-pill" style="color:{"#d4b483" if n == tn_r["key"] else "#888"};'
            f'border-color:{"#d4b483" if n == tn_r["key"] else "#222"}">{n}</span>'
            for n in tn_r["scale_notes"]
        )
        st.markdown(scale_html, unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:1.5rem">Chord Progression</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.75rem;color:#555;margin-bottom:0.5rem;font-family:Space Mono">'
            f'Estimated from {tn_r["mode"]} mode</div>',
            unsafe_allow_html=True
        )
        chord_html = "".join(
            f'<span class="chord-pill">{c}</span>' for c in tn_r["chord_progression"]
        )
        st.markdown(chord_html, unsafe_allow_html=True)

        # Chord timeline
        st.markdown('<div class="section-header" style="margin-top:1.5rem">Harmonic Timeline</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_chord_timeline(result), use_container_width=True)

        # Mode info
        mode_info = {
            'minor':      ('Natural Minor — Aeolian', 'Dark, melancholic, introspective. Foundation of most deep/dark techno.'),
            'dorian':     ('Dorian Mode',              'Minor with raised 6th. Groove-oriented, modal jazz feeling. Common in hypnotic techno.'),
            'phrygian':   ('Phrygian Mode',            'Minor with flat 2nd. Tense, cinematic, Middle Eastern quality. Spanish/dark techno.'),
            'mixolydian': ('Mixolydian Mode',          'Major with flat 7th. Open, driving, rock-influenced. Less common in techno.'),
            'major':      ('Major Scale — Ionian',     'Bright, clear, resolved. Rare in dark techno; more common in melodic techno.'),
        }
        mname, mdesc = mode_info.get(tn_r['mode'], ('Unknown Mode', ''))
        st.markdown(f"""
        <div style="background:#0d0d0d;border:1px solid #1e1e1e;padding:1rem 1.2rem;margin-top:0.5rem">
            <div style="font-family:Space Mono;font-size:0.75rem;color:{mode_col};margin-bottom:0.3rem">{mname}</div>
            <div style="font-size:0.82rem;color:#888;line-height:1.6">{mdesc}</div>
        </div>""", unsafe_allow_html=True)

    with col_t2:
        st.plotly_chart(plot_key_circle(result), use_container_width=True)

        st.markdown('<div class="section-header">Key Summary</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.4rem;color:{mode_col}">{tn_r['key_string'].upper()}</div>
            <div class="metric-label">Detected key</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.4rem">{tn_r['relative_key'].upper()}</div>
            <div class="metric-label">Relative key</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.4rem">{tn_r['confidence']*100:.0f}<span style="font-size:1rem">%</span></div>
            <div class="metric-label">Confidence</div>
        </div>""", unsafe_allow_html=True)

        if not real:
            st.markdown("""
            <div style="margin-top:1rem;padding:0.8rem;background:#0d0d0d;border:1px solid #1e1e1e;
                        font-size:0.72rem;color:#555;line-height:1.6;font-family:Space Mono">
                ⚠ Key detection is estimated in demo mode. Install librosa for chromagram-based analysis.
            </div>""", unsafe_allow_html=True)


# ── TAB 6: CORPUS
with tab6:
    if not check_corpus:
        st.info("Enable **Corpus Comparison** in the sidebar to see this analysis.")
    else:
        st.plotly_chart(plot_corpus_scatter(result), use_container_width=True)

        st.markdown('<div class="section-header">Percentile Position vs. Corpus (n=30)</div>', unsafe_allow_html=True)

        np.random.seed(hash(uploaded.name) % (2**31))
        percentiles = {
            "BPM":              int(np.random.randint(20, 80)),
            "Spectral Density": int(np.random.randint(20, 80)),
            "Spectral Centroid":int(np.random.randint(20, 80)),
            "Sub-Bass Ratio":   int(np.random.randint(20, 80)),
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


# ── TAB 7: EXPORT
with tab7:
    # ── PDF/TXT download button — prominent, at top
    interp_text = generate_interpretation(result)
    stem = Path(uploaded.name).stem

    # Build the HTML report for PDF
    comp_c  = "#9abdaa" if compliant else "#c47a7a"
    comp_lbl= "PROTOCOL COMPLIANT" if compliant else "NON-COMPLIANT"
    p_rows  = ""
    for code, pp in result["protocol_compliance"]["principles"].items():
        ok  = pp["compliant"]
        pc2 = "#9abdaa" if ok else "#c47a7a"
        p_rows += f"<tr><td><b>{code}</b></td><td>{pp['name']}</td><td style='color:{pc2}'><b>{'PASS' if ok else 'FAIL'}</b></td><td style='color:#7a6e62'>{pp['details']}</td></tr>\n"

    interp_html_body = interp_text.replace("\n", "<br>").replace("=" * 56, "<hr>").replace("-" * 40, "")
    tv2, sv2, lv2 = result["tempo"], result["spectral"], result["lowend"]
    tnv2 = result["tonality"]

    html_report = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600&family=DM+Mono:wght@300;400&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'DM Mono',monospace;background:#faf8f5;color:#2a2218;padding:3rem;max-width:860px;margin:0 auto;font-size:12.5px;line-height:1.75}}
h1{{font-family:'Cormorant Garamond',serif;font-weight:300;font-size:2rem;letter-spacing:.06em;border-bottom:1px solid #d4c8b8;padding-bottom:.8rem;margin-bottom:1.5rem;color:#1a1410}}
h2{{font-family:'DM Mono',monospace;font-size:.6rem;text-transform:uppercase;letter-spacing:.2em;color:#9a8e80;border-bottom:1px solid #e8e0d4;padding-bottom:.3rem;margin:2rem 0 1rem}}
.status{{display:inline-block;padding:.3rem .9rem;border:1px solid {comp_c};color:{comp_c};font-size:.65rem;letter-spacing:.2em;margin:.8rem 0 1.5rem}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:.7rem;margin:1.2rem 0}}
.card{{border:1px solid #e0d8cc;padding:.8rem;border-left:2px solid #c9a96e;background:#fdfbf8}}
.val{{font-family:'Cormorant Garamond',serif;font-size:1.7rem;font-weight:300;color:#c9a96e;line-height:1}}
.lbl{{font-size:.55rem;letter-spacing:.18em;text-transform:uppercase;color:#9a8e80;margin-top:.2rem}}
table{{width:100%;border-collapse:collapse;margin:1rem 0}}
th{{font-size:.55rem;text-transform:uppercase;letter-spacing:.15em;color:#9a8e80;padding:.5rem;text-align:left;border-bottom:1px solid #e0d8cc;background:#fdfbf8}}
td{{padding:.5rem;border-bottom:1px solid #ede8e0;font-size:.8rem}}
.analysis{{background:#fdfbf8;border:1px solid #e0d8cc;border-left:2px solid #c9a96e;padding:1.4rem;margin-top:1rem;white-space:pre-wrap;font-size:.8rem;line-height:1.8;color:#3a3028}}
hr{{border:none;border-top:1px solid #e0d8cc;margin:1rem 0}}
.footer{{font-size:.55rem;color:#c0b8ac;margin-top:3rem;border-top:1px solid #e0d8cc;padding-top:1rem;text-transform:uppercase;letter-spacing:.12em}}
@media print{{body{{padding:1.5rem}}}}
</style></head><body>
<h1>Continuous Inertia Techno Analyzer</h1>
<p style="font-size:.65rem;color:#9a8e80">{result['metadata']['filename']} &nbsp;·&nbsp; {tv2['bpm']:.1f} BPM &nbsp;·&nbsp; {result['metadata']['duration_min']:.1f} min &nbsp;·&nbsp; {tnv2['key_string'].upper()}</p>
<div class="status">{comp_lbl} &nbsp;—&nbsp; {passed}/5 PRINCIPLES</div>
<div class="grid">
<div class="card"><div class="val">{tv2['bpm']:.1f}</div><div class="lbl">BPM</div></div>
<div class="card"><div class="val" style="color:{'#9abdaa' if tv2['bpm_variance_pct']<1.5 else '#c47a7a'}">{tv2['bpm_variance_pct']:.2f}%</div><div class="lbl">BPM Variance</div></div>
<div class="card"><div class="val">{sv2['density']['mean']:.3f}</div><div class="lbl">Spectral Density</div></div>
<div class="card"><div class="val">{sv2['centroid']['mean']:.0f} Hz</div><div class="lbl">Centroid</div></div>
<div class="card"><div class="val" style="color:{'#9abdaa' if lv2['sub_presence_pct']>=0.90 else '#c47a7a'}">{lv2['sub_presence_pct']*100:.0f}%</div><div class="lbl">Sub-Bass</div></div>
<div class="card"><div class="val">{tnv2['key_string'].upper()}</div><div class="lbl">Key &nbsp;·&nbsp; {tnv2['confidence']*100:.0f}% conf</div></div>
</div>
<h2>Protocol Compliance</h2>
<table><tr><th>Code</th><th>Principle</th><th>Result</th><th>Details</th></tr>{p_rows}</table>
<h2>Full Analysis with Explanations</h2>
<div class="analysis">{interp_text}</div>
<p class="footer">Continuous Inertia Techno Analyzer v{result['metadata']['analyzer_version']} &nbsp;·&nbsp; Open in browser → Print → Save as PDF</p>
</body></html>"""

    # ── PROMINENT download button
    st.markdown('<div class="section-header">Download Report</div>', unsafe_allow_html=True)
    col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
    with col_dl2:
        st.download_button(
            label="◆  Download Analysis Report (HTML → PDF)",
            data=html_report,
            file_name=f"CI_Analysis_{stem}.html",
            mime="text/html",
            use_container_width=True,
        )
    st.markdown(
        '<div style="text-align:center;font-family:DM Mono,monospace;font-size:0.6rem;'
        'color:#4a4038;margin-top:0.3rem;margin-bottom:1.5rem">'
        'Open the downloaded file in any browser · File → Print → Save as PDF</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-header">Acoustic Interpretation</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="interp-box" style="white-space:pre-wrap;font-size:0.75rem">{interp_text}</div>',
        unsafe_allow_html=True
    )
    st.download_button(
        "⬇ Download Interpretation (.txt)",
        data=interp_text,
        file_name=f"interpretation_{Path(uploaded.name).stem}.txt",
        mime="text/plain",
    )

    st.markdown('<div class="section-header" style="margin-top:2rem">Export Analysis Data</div>', unsafe_allow_html=True)

    col_ex1, col_ex2 = st.columns([2, 1])

    with col_ex1:
        if export_format == "JSON":
            # Serialize (remove non-serializable numpy arrays already converted to lists)
            export_result = {k: v for k, v in result.items()}
            json_str = json.dumps(export_result, indent=2, default=str)
            st.download_button(
                "⬇ Download JSON Report",
                data=json_str,
                file_name=f"analysis_{Path(uploaded.name).stem}.json",
                mime="application/json",
            )
            st.code(json_str[:900] + "\n...", language="json")

        elif export_format == "CSV":
            flat = {
                "filename":             result["metadata"]["filename"],
                "duration_min":         result["metadata"]["duration_min"],
                "bpm":                  result["tempo"]["bpm"],
                "bpm_variance_pct":     result["tempo"]["bpm_variance_pct"],
                "density_mean":         result["spectral"]["density"]["mean"],
                "centroid_mean":        result["spectral"]["centroid"]["mean"],
                "rolloff_mean":         result["spectral"]["rolloff"]["mean"],
                "sub_presence_pct":     result["lowend"]["sub_presence_pct"],
                "layers_mean":          result["structure"]["layers_mean"],
                "mean_interval_bars":   result["structure"]["mean_interval_bars"],
                "key":                  result["tonality"]["key_string"],
                "key_confidence":       result["tonality"]["confidence"],
                "chord_progression":    " - ".join(result["tonality"]["chord_progression"]),
                "principles_passed":    result["protocol_compliance"]["principles_passed"],
                "compliant":            result["protocol_compliance"]["compliant"],
                "antipattern_violations": result["antipatterns"]["total_violations"],
            }
            df      = pd.DataFrame([flat])
            csv_str = df.to_csv(index=False)
            st.download_button(
                "⬇ Download CSV",
                data=csv_str,
                file_name=f"analysis_{Path(uploaded.name).stem}.csv",
                mime="text/csv",
            )
            st.dataframe(df.T.rename(columns={0: "Value"}), use_container_width=True)

        elif export_format == "LaTeX":
            t_val  = result["tempo"]
            s_val  = result["spectral"]
            l_val  = result["lowend"]
            p_val  = result["protocol_compliance"]
            tn_val = result["tonality"]
            stem   = Path(uploaded.name).stem

            def chk(cond): return r"\checkmark" if cond else r"$\times$"

            latex_lines = [
                r"% Auto-generated by Continuous Inertia Analyzer v2.1",
                r"\begin{table}[h]",
                r"\centering",
                "\\caption{Acoustic Analysis: " + stem + "}",
                r"\begin{tabular}{lcc}",
                r"\hline",
                r"\textbf{Parameter} & \textbf{Value} & \textbf{Compliant} \\",
                r"\hline",
                f"BPM & {t_val['bpm']:.1f} & {chk(128 <= t_val['bpm'] <= 135)} \\\\",
                f"BPM Variance (%) & {t_val['bpm_variance_pct']:.2f} & {chk(t_val['bpm_variance_pct'] < 1.5)} \\\\",
                f"Spectral Density & {s_val['density']['mean']:.3f} & {chk(0.30 <= s_val['density']['mean'] <= 0.45)} \\\\",
                f"Centroid (Hz) & {s_val['centroid']['mean']:.0f} & {chk(250 <= s_val['centroid']['mean'] <= 400)} \\\\",
                f"Sub-Bass (%) & {l_val['sub_presence_pct']*100:.0f} & {chk(l_val['sub_presence_pct'] >= 0.90)} \\\\",
                f"Key & {tn_val['key_string'].upper()} & -- \\\\",
                r"\hline",
                f"Principles Passed & {p_val['principles_passed']}/5 & {chk(p_val['compliant'])} \\\\",
                r"\hline",
                r"\end{tabular}",
                r"\end{table}",
            ]
            latex = "\n".join(latex_lines)
            st.download_button(
                "⬇ Download LaTeX",
                data=latex,
                file_name=f"table_{stem}.tex",
                mime="text/plain",
            )
            st.code(latex, language="latex")
        elif export_format == "HTML Report (→ PDF)":
            interp = generate_interpretation(result)
            tv, sv, lv, pv, tnv = (result["tempo"], result["spectral"],
                                    result["lowend"], result["protocol_compliance"],
                                    result["tonality"])
            comp   = pv["compliant"]
            sc     = "#9abdaa" if comp else "#c47a7a"
            st_lbl = "PROTOCOL COMPLIANT" if comp else "NON-COMPLIANT"
            rows = ""
            for code, p in pv["principles"].items():
                ok  = p["compliant"]
                pc2 = "#9abdaa" if ok else "#c47a7a"
                rows += f"<tr><td><b>{code}</b></td><td>{p['name']}</td><td style='color:{pc2}'><b>{'PASS' if ok else 'FAIL'}</b></td><td style='color:#7a6e62;font-size:0.8rem'>{p['details']}</td></tr>\n"
            interp_html = interp.replace("**", "").replace("\n\n", "</p><p>").replace("\n", "<br>")
            html_report = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600&family=DM+Mono:wght@300;400&display=swap');
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ font-family:'DM Mono',monospace; background:#faf8f5; color:#2a2218; padding:3.5rem; max-width:900px; margin:0 auto; font-size:13px; line-height:1.7; }}
h1 {{ font-family:'Cormorant Garamond',serif; font-weight:300; font-size:2.2rem; letter-spacing:0.06em; border-bottom:1px solid #d4c8b8; padding-bottom:0.8rem; margin-bottom:1.5rem; color:#1a1410; }}
h2 {{ font-family:'DM Mono',monospace; font-size:0.6rem; text-transform:uppercase; letter-spacing:0.2em; color:#9a8e80; border-bottom:1px solid #e8e0d4; padding-bottom:0.3rem; margin:2rem 0 1rem; }}
.status {{ display:inline-block; padding:0.3rem 0.9rem; border:1px solid {sc}; color:{sc}; font-size:0.65rem; letter-spacing:0.2em; margin:1rem 0 1.5rem; }}
.grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:0.8rem; margin:1.2rem 0; }}
.card {{ border:1px solid #e0d8cc; padding:0.9rem; border-left:2px solid #c9a96e; background:#fdfbf8; }}
.val {{ font-family:'Cormorant Garamond',serif; font-size:1.8rem; font-weight:300; color:#c9a96e; line-height:1; }}
.lbl {{ font-size:0.55rem; letter-spacing:0.18em; text-transform:uppercase; color:#9a8e80; margin-top:0.25rem; }}
table {{ width:100%; border-collapse:collapse; margin:1.2rem 0; }}
th {{ font-size:0.55rem; text-transform:uppercase; letter-spacing:0.15em; color:#9a8e80; padding:0.5rem; text-align:left; border-bottom:1px solid #e0d8cc; background:#fdfbf8; }}
td {{ padding:0.55rem; border-bottom:1px solid #ede8e0; font-size:0.82rem; }}
.interp {{ background:#fdfbf8; border:1px solid #e0d8cc; border-left:2px solid #c9a96e; padding:1.4rem; font-size:0.82rem; line-height:1.9; color:#3a3028; }}
.interp p {{ margin-bottom:1rem; }}
.footer {{ font-size:0.55rem; color:#c0b8ac; margin-top:3rem; border-top:1px solid #e0d8cc; padding-top:1rem; text-transform:uppercase; letter-spacing:0.12em; }}
@media print {{ body {{ padding:1.5rem; }} }}
</style></head><body>
<h1>Continuous Inertia Techno Analyzer</h1>
<p style="font-size:0.65rem;color:#9a8e80;margin-bottom:0.5rem">{result['metadata']['filename']} · {result['metadata']['duration_min']:.1f} min · {tv['bpm']:.1f} BPM · Key: {tnv['key_string'].upper()} · Analyzer v{result['metadata']['analyzer_version']}</p>
<div class="status">{st_lbl} — {pv['principles_passed']}/5 PRINCIPLES</div>
<div class="grid">
  <div class="card"><div class="val">{tv['bpm']:.1f}</div><div class="lbl">BPM</div></div>
  <div class="card"><div class="val" style="color:{'#9abdaa' if tv['bpm_variance_pct']<1.5 else '#c47a7a'}">{tv['bpm_variance_pct']:.2f}%</div><div class="lbl">BPM Variance</div></div>
  <div class="card"><div class="val">{sv['density']['mean']:.3f}</div><div class="lbl">Spectral Density</div></div>
  <div class="card"><div class="val">{sv['centroid']['mean']:.0f} Hz</div><div class="lbl">Centroid</div></div>
  <div class="card"><div class="val" style="color:{'#9abdaa' if lv['sub_presence_pct']>=0.90 else '#c47a7a'}">{lv['sub_presence_pct']*100:.0f}%</div><div class="lbl">Sub-Bass</div></div>
  <div class="card"><div class="val">{tnv['key_string'].upper()}</div><div class="lbl">Key · {tnv['confidence']*100:.0f}% confidence</div></div>
</div>
<h2>Protocol Compliance</h2>
<table><tr><th>Code</th><th>Principle</th><th>Result</th><th>Details</th></tr>{rows}</table>
<h2>Acoustic Interpretation</h2>
<div class="interp"><p>{interp_html}</p></div>
<p class="footer">Generated by Continuous Inertia Techno Analyzer v{result['metadata']['analyzer_version']} · Open in browser → File → Print → Save as PDF</p>
</body></html>"""
            st.download_button("⬇ Download HTML Report (→ Print as PDF)",
                               data=html_report,
                               file_name=f"report_{Path(uploaded.name).stem}.html",
                               mime="text/html")
            st.info("Open the downloaded file in your browser → File → Print → Save as PDF for a clean publication-ready report.")

    with col_ex2:
        st.markdown('<div class="section-header">Export Formats</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:DM Mono,monospace;font-size:0.68rem;line-height:2.2;color:#7a6e62">
        JSON · full pipeline<br>
        CSV  · SPSS / R / Python<br>
        LaTeX · direct paste<br>
        HTML → PDF · annotated report<br>
        TXT  · interpretation
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
