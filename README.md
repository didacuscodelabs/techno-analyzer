# ⬛ Continuous Inertia Techno Analyzer

**Protocol-enhanced acoustic analysis tool for continuous inertia techno research.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## What it does

Analyzes techno tracks for compliance with the **Continuous Inertia Techno Protocol** — 5 core acoustic principles grounded in neurophysiological mechanisms of rhythmic entrainment and autonomic regulation.

| Principle | Description | Threshold |
|-----------|-------------|-----------|
| P1 Temporal Stability | BPM variance | < 1.5% |
| P2 Spectral Parsimony | Spectral density | 0.30–0.45 |
| P3 Periodic Micro-Variation | Change interval | 8–16 bars |
| P4 Continuous Sub-Bass | Sub-bass presence | ≥ 90% |
| P5 Textural Continuity | Texture presence | ≥ 85% |

## Features

- **Protocol compliance** — automatic pass/fail for all 5 principles + anti-pattern detection
- **Corpus comparison** — scatter plots and percentile position vs. reference corpus (n=30)
- **Time series plots** — BPM stability, spectral density, sub-bass continuity over time
- **Export** — JSON, CSV (for R/SPSS/Python), LaTeX table (paste directly into paper)
- **Batch CLI** — process full corpus, export combined CSV
- **Demo mode** — runs without audio files for UI exploration

## Installation

```bash
git clone https://github.com/your-username/techno-analyzer
cd techno-analyzer
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as main file
4. Deploy (free)

> **Note:** `librosa` and `essentia` require additional system dependencies on Streamlit Cloud. See `packages.txt` for apt dependencies.

## CLI Usage

```bash
# Single track with protocol check
python -m techno_analyzer analyze track.wav --protocol --corpus corpus_db.json

# Batch processing
python -m techno_analyzer batch tracks/*.wav --protocol --output results.csv

# Build corpus from analyzed tracks
python -m techno_analyzer build-corpus analyzed/*.json --output corpus_db.json
```

## Project Structure

```
techno_analyzer_app/
├── app.py                    # Streamlit application
├── requirements.txt
├── techno_analyzer/
│   ├── core/
│   │   ├── audio_loader.py
│   │   ├── tempo_analyzer.py
│   │   ├── spectral_analyzer.py
│   │   ├── structure_analyzer.py
│   │   └── lowend_analyzer.py
│   ├── protocol/
│   │   ├── compliance_checker.py
│   │   ├── antipattern_detector.py
│   │   └── thresholds.py
│   ├── comparison/
│   │   ├── corpus_manager.py
│   │   └── similarity.py
│   └── reporting/
│       ├── report_generator.py
│       └── visualizer.py
```

## Research Context

Based on: *"Acoustic Principles of Continuous Inertia Techno: A Production Protocol Based on Autonomic Regulation"* (Preprint v1.0, 2026).

Protocol grounded in:
- Large & Jones (1999) — Dynamic Attending Theory
- Friston (2010) — Predictive Coding
- Porges (2011) — Polyvagal Theory
- Thaut et al. (2015) — Rhythmic Auditory Stimulation

## License

MIT — Free to use for research and production purposes.

---

*Built for [Partícula Primordial] — Music as Embodied Technology*
