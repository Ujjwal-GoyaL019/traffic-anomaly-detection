# 🚦 Road Traffic Anomaly Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=flat-square&logo=pytorch"/>
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00BFFF?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Platform-Kaggle%20%7C%20Colab-orange?style=flat-square"/>
</p>

<p align="center">
  Unsupervised anomaly detection for road traffic video —
  <b>YOLOv8</b> · <b>ByteTrack</b> · <b>LSTM Autoencoder</b> · <b>Isolation Forest</b>
</p>

---

## What it does

Detects anomalous vehicle behaviour in traffic video without needing any labeled anomaly data. The system learns what *normal* traffic looks like and flags deviations at inference time.

**Detects:**
- 🔴 Wrong-way driving
- 🔴 Sudden stops in active lanes
- 🟠 Near-miss situations
- 🟠 Congestion clusters
- 🔴 Collision events

**Output:** annotated video with per-vehicle colour-coded bounding boxes (🟢 normal · 🟠 borderline · 🔴 anomalous) and a per-frame anomaly timeline.

---

## Pipeline

```
Video frames
    │
    ▼
YOLOv8n  ──── detects vehicles (cars, buses, trucks, bikes, persons)
    │
    ▼
ByteTrack ─── assigns consistent track IDs across frames
    │
    ▼
Feature Extractor ── builds 9D feature vector per track per frame
    │           [cx, cy, w, h, speed, accel, heading, hchg, IoU]
    │
    ├──▶ LSTM Autoencoder ── trained on normal sequences only
    │                         flags high reconstruction error
    │
    └──▶ Isolation Forest ── trained on trajectory statistics
                              flags spatial outliers
    │
    ▼
Weighted Ensemble (0.6 × LSTM + 0.4 × IF)
    │
    ▼
Auto-tuned threshold → Anomaly / Normal decision
```

---

## Quickstart

### Kaggle (recommended — BDD100K available)
```
1. Add BDD100K dataset to the notebook
2. Settings → Accelerator → GPU T4 x2
3. Run All
```

### Google Colab
```
1. Runtime → Change runtime type → T4 GPU
2. Run All  (synthetic data generated automatically)
```

### Dependencies
```bash
pip install ultralytics supervision scikit-learn joblib matplotlib opencv-python-headless
```

Tested on:
```
ultralytics  8.4.39
supervision  0.27.0
torch        2.10.0+cu128
CUDA         Tesla T4
```

---

## Feature Vector

Each tracked vehicle gets a **9-dimensional** feature vector per frame:

| # | Feature | Description |
|---|---------|-------------|
| 1 | `cx` | Bounding box center x |
| 2 | `cy` | Bounding box center y |
| 3 | `w` | Box width |
| 4 | `h` | Box height |
| 5 | `speed` | Euclidean displacement × FPS |
| 6 | `accel` | Rate of change of speed |
| 7 | `heading` | Direction in degrees |
| 8 | `hchg` | Heading change from previous frame |
| 9 | `iou` | Max IoU with any neighbouring vehicle |

The IoU feature acts as a proximity risk signal — key for detecting near-miss and collision events without explicit labels.

---

## Models

### LSTM Autoencoder
- 2-layer encoder → latent dim 32 → 2-layer decoder
- Hidden dim: 64 · Dropout: 0.2
- Trained on **normal sequences only** (MSE loss)
- Anomaly score = reconstruction error
- Threshold at 95th percentile of training errors

### Isolation Forest
- 200 trees, trained on trajectory summary statistics
- Each sequence `(20, 9)` → `(45,)` vector: mean, std, min, max, delta per feature
- Score min-max normalized to `[0, 1]`

### Ensemble
```
combined_score = 0.6 × LSTM_score + 0.4 × IF_score
```
Threshold auto-tuned by maximising F1 over 80 candidates on a validation split.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Sequence length | 20 frames |
| Frame rate | 5 FPS (subsampled) |
| Detection resolution | 416 × 416 |
| Min frames for LSTM scoring | 8 |
| LSTM threshold percentile | 95th |
| Ensemble weights | 0.6 / 0.4 |
| Random seed | 42 |

---

## Inference Modes

**Video mode** — full pipeline, temporal + spatial scoring. Best results.

**Image mode** — single frame, Isolation Forest only. Works for spatial anomalies (near-miss, congestion). Not suitable for temporal anomalies (wrong-way, collision).

---

## Saved Artifacts

After training, the following are saved to disk:

```
lstm_autoencoder.pt
isolation_forest.joblib
scaler.joblib
threshold.json
```

Loaded automatically at inference — no retraining needed.

---

## Dataset

Primary training uses a **synthetic dataset** generated procedurally:
- Train: 300 normal + 150 anomaly sequences
- Test: 100 normal + 50 anomaly sequences
- Each sequence shape: `(20, 9)`

BDD100K is supported for real-video feature extraction when mounted on Kaggle. Used here only for pipeline validation since it has no anomaly labels for the tracking split.

---

## Limitations

- Needs ≥ 8 frames of track history before scoring a new vehicle
- Processes at 5 FPS — very fast transient events may be missed
- Quantitative evaluation on labeled real-world data is pending
- Image inference has low recall for temporal anomaly types

---

## Future

- [ ] Evaluation on DoTA dataset
- [ ] Transformer / hierarchical LSTM for longer trajectories
- [ ] Online threshold adaptation for changing traffic conditions
- [ ] Multi-camera vehicle re-identification
- [ ] Edge deployment on NVIDIA Jetson

---

## License

MIT — see [LICENSE](LICENSE) for details.
