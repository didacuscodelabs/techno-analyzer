"""
db.py — SQLite Corpus Database for CI Techno Analyzer
Persists every analysis, enables ML training, historical comparison, trend analysis.
"""

import sqlite3
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional

DB_PATH = Path("corpus.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tracks (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash     TEXT UNIQUE,
            filename      TEXT NOT NULL,
            analyzed_at   TEXT NOT NULL,
            duration_min  REAL,
            is_real       INTEGER DEFAULT 0,

            -- Tempo
            bpm           REAL,
            bpm_variance  REAL,

            -- Spectral
            density_mean  REAL,
            density_std   REAL,
            centroid_mean REAL,
            centroid_std  REAL,
            rolloff_mean  REAL,
            layers_mean   REAL,

            -- Low end
            sub_presence  REAL,
            sub_kick_ratio REAL,

            -- Structure
            mean_interval REAL,
            texture_presence REAL,

            -- Tonality
            key_root      TEXT,
            mode          TEXT,
            key_confidence REAL,

            -- Protocol
            p1_pass       INTEGER,
            p2_pass       INTEGER,
            p3_pass       INTEGER,
            p4_pass       INTEGER,
            p5_pass       INTEGER,
            principles_passed INTEGER,
            compliant     INTEGER,

            -- Anti-patterns
            density_overload INTEGER,
            sub_absent    INTEGER,
            bpm_change    INTEGER,

            -- Label for ML
            label         TEXT DEFAULT 'unlabeled',   -- 'ci_techno' | 'non_ci' | 'unlabeled'
            label_source  TEXT DEFAULT 'auto',         -- 'auto' | 'manual' | 'model'
            ml_confidence REAL DEFAULT NULL,

            -- Full JSON blob for anything else
            full_json     TEXT
        );

        CREATE TABLE IF NOT EXISTS corpus_stats (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            computed_at   TEXT NOT NULL,
            n_tracks      INTEGER,
            n_compliant   INTEGER,
            mean_bpm      REAL,
            std_bpm       REAL,
            mean_density  REAL,
            std_density   REAL,
            mean_centroid REAL,
            std_centroid  REAL,
            mean_sub      REAL,
            std_sub       REAL
        );

        CREATE INDEX IF NOT EXISTS idx_tracks_hash   ON tracks(file_hash);
        CREATE INDEX IF NOT EXISTS idx_tracks_label  ON tracks(label);
        CREATE INDEX IF NOT EXISTS idx_tracks_date   ON tracks(analyzed_at);
        CREATE INDEX IF NOT EXISTS idx_tracks_comply ON tracks(compliant);
    """)
    conn.commit()
    conn.close()


def file_hash(filename: str, bpm: float, duration: float) -> str:
    """Stable hash that identifies a track (filename + key metrics)."""
    h = hashlib.md5(f"{filename}:{bpm:.1f}:{duration:.2f}".encode()).hexdigest()
    return h[:16]


def save_analysis(result: dict, label: str = "unlabeled", label_source: str = "auto") -> int:
    """
    Insert or update a track analysis in the corpus.
    Returns the row id.
    """
    init_db()
    conn = get_connection()

    t  = result["tempo"]
    s  = result["spectral"]
    l  = result["lowend"]
    st = result["structure"]
    tn = result["tonality"]
    p  = result["protocol_compliance"]
    ap = result["antipatterns"]
    md = result["metadata"]

    fh = file_hash(md["filename"], t["bpm"], md["duration_min"])
    ts = datetime.utcnow().isoformat()

    row = dict(
        file_hash        = fh,
        filename         = md["filename"],
        analyzed_at      = ts,
        duration_min     = md["duration_min"],
        is_real          = int(md.get("real_analysis", False)),

        bpm              = t["bpm"],
        bpm_variance     = t["bpm_variance_pct"],

        density_mean     = s["density"]["mean"],
        density_std      = s["density"]["std"],
        centroid_mean    = s["centroid"]["mean"],
        centroid_std     = s["centroid"]["std"],
        rolloff_mean     = s.get("rolloff", {}).get("mean", None),
        layers_mean      = st["layers_mean"],

        sub_presence     = l["sub_presence_pct"],
        sub_kick_ratio   = l["sub_kick_ratio"],

        mean_interval    = st["mean_interval_bars"],
        texture_presence = st.get("texture_presence", None),

        key_root         = tn["key"],
        mode             = tn["mode"],
        key_confidence   = tn["confidence"],

        p1_pass          = int(p["principles"]["P1"]["compliant"]),
        p2_pass          = int(p["principles"]["P2"]["compliant"]),
        p3_pass          = int(p["principles"]["P3"]["compliant"]),
        p4_pass          = int(p["principles"]["P4"]["compliant"]),
        p5_pass          = int(p["principles"]["P5"]["compliant"]),
        principles_passed= p["principles_passed"],
        compliant        = int(p["compliant"]),

        density_overload = int(ap["density_overload"]),
        sub_absent       = int(ap["sub_absent"]),
        bpm_change       = int(ap["bpm_change"]),

        label            = label,
        label_source     = label_source,
        ml_confidence    = None,

        full_json        = json.dumps(result, default=str),
    )

    placeholders = ", ".join(["?"] * len(row))
    cols         = ", ".join(row.keys())
    vals         = list(row.values())

    try:
        cur = conn.execute(
            f"INSERT INTO tracks ({cols}) VALUES ({placeholders})", vals
        )
        row_id = cur.lastrowid
    except sqlite3.IntegrityError:
        # Already exists — update
        set_clause = ", ".join([f"{k}=?" for k in row.keys() if k != "file_hash"])
        update_vals = [v for k, v in row.items() if k != "file_hash"] + [fh]
        conn.execute(
            f"UPDATE tracks SET {set_clause} WHERE file_hash=?", update_vals
        )
        cur2 = conn.execute("SELECT id FROM tracks WHERE file_hash=?", (fh,))
        row_id = cur2.fetchone()[0]

    conn.commit()
    conn.close()
    return row_id


def get_all_tracks(label_filter: Optional[str] = None) -> list[dict]:
    """Return all tracks, optionally filtered by label."""
    init_db()
    conn = get_connection()
    if label_filter:
        rows = conn.execute(
            "SELECT * FROM tracks WHERE label=? ORDER BY analyzed_at DESC", (label_filter,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM tracks ORDER BY analyzed_at DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_corpus_features() -> dict:
    """Aggregate statistics over the full corpus — used for scatter plot reference."""
    init_db()
    conn = get_connection()
    rows = conn.execute("""
        SELECT bpm, density_mean, centroid_mean, sub_presence,
               layers_mean, bpm_variance, compliant, label, filename,
               key_root, mode, mean_interval
        FROM tracks
        ORDER BY analyzed_at DESC
    """).fetchall()
    conn.close()
    if not rows:
        return {"n": 0, "rows": []}
    return {"n": len(rows), "rows": [dict(r) for r in rows]}


def get_corpus_stats() -> dict:
    """Mean ± std for key features across the corpus."""
    import numpy as np
    data = get_corpus_features()
    if data["n"] == 0:
        return {}
    rows = data["rows"]

    def _stats(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)} if vals else {}

    return {
        "n_tracks":    data["n"],
        "n_compliant": sum(1 for r in rows if r["compliant"]),
        "bpm":         _stats("bpm"),
        "density":     _stats("density_mean"),
        "centroid":    _stats("centroid_mean"),
        "sub":         _stats("sub_presence"),
        "layers":      _stats("layers_mean"),
        "bpm_var":     _stats("bpm_variance"),
    }


def update_label(track_id: int, label: str, source: str = "manual"):
    """Manually label a track for ML training."""
    init_db()
    conn = get_connection()
    conn.execute(
        "UPDATE tracks SET label=?, label_source=? WHERE id=?",
        (label, source, track_id)
    )
    conn.commit()
    conn.close()


def delete_track(track_id: int):
    init_db()
    conn = get_connection()
    conn.execute("DELETE FROM tracks WHERE id=?", (track_id,))
    conn.commit()
    conn.close()


def export_for_ml() -> "pd.DataFrame":
    """
    Export feature matrix ready for sklearn.
    Returns a DataFrame with features + label column.
    """
    import pandas as pd
    rows = get_all_tracks()
    if not rows:
        return pd.DataFrame()

    feature_cols = [
        "bpm", "bpm_variance", "density_mean", "density_std",
        "centroid_mean", "centroid_std", "layers_mean",
        "sub_presence", "sub_kick_ratio", "mean_interval",
        "p1_pass", "p2_pass", "p3_pass", "p4_pass", "p5_pass",
        "compliant", "label", "filename", "analyzed_at"
    ]
    df = pd.DataFrame(rows)[feature_cols]
    return df


def get_production_feedback(result: dict) -> list[dict]:
    """
    Compare current track against corpus stats.
    Returns actionable feedback items sorted by priority.
    """
    stats = get_corpus_stats()
    if stats.get("n_tracks", 0) < 5:
        return []  # Not enough data yet

    p = result["protocol_compliance"]["principles"]
    t = result["tempo"]
    s = result["spectral"]
    l = result["lowend"]
    st2 = result["structure"]

    feedback = []

    # P1 — BPM variance
    if not p["P1"]["compliant"]:
        corpus_var = stats["bpm_var"]["mean"]
        feedback.append({
            "priority": "HIGH",
            "principle": "P1",
            "issue": "BPM drift too high",
            "value": f"{t['bpm_variance_pct']:.2f}%",
            "target": "< 1.5%",
            "corpus_mean": f"{corpus_var:.2f}%",
            "action": "Tighten quantization grid. In Ableton: right-click MIDI clip → Quantize to 1/16. In Ableton with audio: use Warp markers at every 4 bars.",
        })

    # P2 — Spectral density
    if not p["P2"]["compliant"]:
        d = s["density"]["mean"]
        cd = stats["density"]["mean"]
        if d > 0.45:
            feedback.append({
                "priority": "HIGH",
                "principle": "P2",
                "issue": f"Spectral density too high ({d:.3f} > 0.45)",
                "value": f"{d:.3f}",
                "target": "0.30–0.45",
                "corpus_mean": f"{cd:.3f}",
                "action": "Remove or mute one layer and re-export. The most common culprits: mid-range pads, secondary percussion, or reverb tails that are too loud. Try muting each bus individually and re-analyzing.",
            })
        else:
            feedback.append({
                "priority": "MEDIUM",
                "principle": "P2",
                "issue": f"Spectral density too low ({d:.3f} < 0.30)",
                "value": f"{d:.3f}",
                "target": "0.30–0.45",
                "corpus_mean": f"{cd:.3f}",
                "action": "Add a subtle continuous element: low-level noise layer, detuned oscillator, or room reverb return. Keep it at -20dBFS or below.",
            })

    # P3 — Variation interval
    if not p["P3"]["compliant"]:
        iv = st2["mean_interval_bars"]
        feedback.append({
            "priority": "MEDIUM",
            "principle": "P3",
            "issue": f"Change interval {iv:.1f} bars ({'too long' if iv > 16 else 'too short'})",
            "value": f"{iv:.1f} bars",
            "target": "8–16 bars",
            "corpus_mean": f"{stats.get('layers',{}).get('mean', 0):.1f} mean layers",
            "action": (
                "Introduce a subtle change every 8 bars: automate a filter cutoff, add/remove a hi-hat layer, or shift the reverb send level. These should be barely perceptible — not drops or builds."
                if iv > 16 else
                "Reduce the frequency of changes. Let each element settle for at least 8 bars before the next variation. Remove excessive automation lanes."
            ),
        })

    # P4 — Sub bass
    if not p["P4"]["compliant"]:
        sub = l["sub_presence_pct"]
        feedback.append({
            "priority": "HIGH",
            "principle": "P4",
            "issue": f"Sub-bass continuity {sub*100:.0f}% (need ≥90%)",
            "value": f"{sub*100:.0f}%",
            "target": "≥ 90%",
            "corpus_mean": f"{stats['sub']['mean']*100:.0f}%",
            "action": "Check your kick/sub relationship. If you're using a kick-driven sub (sidechain ducking), reduce the sidechain depth or ratio. Consider adding a sustained 808-style sub underneath the kick. The sub layer should be active even in the breakdown — lower the level but keep it present.",
        })

    # P5 — Texture
    if not p["P5"]["compliant"]:
        feedback.append({
            "priority": "MEDIUM",
            "principle": "P5",
            "issue": "Textural continuity insufficient",
            "value": f"{p['P5']['value']*100:.0f}%",
            "target": "≥ 85%",
            "corpus_mean": "~88%",
            "action": "Add a textural layer that runs the full duration at low level: vinyl noise, filtered white noise, or a slow-moving pad. This should sit at -24 to -18 dBFS, barely audible but felt. Even during the breakdown it should remain.",
        })

    feedback.sort(key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[x["priority"]])
    return feedback
