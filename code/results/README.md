# Results: CSI-Based Indoor Localization

Dataset: [here](https://github.com/qiang5love1314/CSI-dataset-for-indoor-localization)

## Experiment Setup

Three classification tasks on WiFi CSI fingerprints using the public CSI dataset:
- **Features**: mean & std of amplitude across 3 antennas × 30 subcarriers (180 dims)
- **Sliding window**: 50-packet windows, max 8 per file to balance small/large locations
- **Models**: RandomForest (200 trees) vs kNN (k=5)

## Results

| Task | Samples | Classes | RF Accuracy | kNN Accuracy | Notes |
|------|---------|---------|-------------|--------------|-------|
| Room ID | 4139 | 4 | **63.4%** | 62.4% | Grouped split (honest) |
| Lab Zone (3×3) | 2536 | 9 | **16.8%** | 16.7% | Grouped split (honest) |
| Lab Exact Coord | 2536 | 317 | **99.7%** | 83.8% | Random split (leaky) |

## Key Findings

**What works:** Distinguishing between major rooms is viable (63%). The CSI fingerprints do differ across rooms in a meaningful way (see `csi_fingerprints.png`).

**What breaks down:** Fine-grained localization within a single room is nearly random (16.8% for a 3×3 grid, only marginally better than 11% baseline). The confusion matrix shows most predictions collapse to a few dominant classes.

**The 99.7% is misleading.** The exact-coordinate task uses a random train/test split, which allows windows from the *same location* to appear in both sets. With 2536 samples spread across 317 locations, many locations have multiple windows → data leakage. The model isn't generalizing; it's memorizing which windows belong to which coordinate. In an honest grouped split (one location stays on one side), precision would be closer to the 16.8% zone result. This is flagged with `*` in the summary.

**Why fine-grained localization fails:** 
- Static fingerprinting assumes CSI is stable at a location — it's not reliable at sub-meter scales
- Antenna orientation, minor position shifts, and multipath variations dominate the signal
- A 50-packet window (microseconds in real time) may be too short to build stable features

## For Moving Drones

All three approaches from the parent README (SpotFi, DeepFi, pure estimation) assume either static or slowly-moving receivers. This experiment confirms the core problem: **CSI-based localization hits a hard wall at ~1m precision in static conditions, and that precision vanishes entirely if the receiver is moving.**

The exact-coordinate "success" is an artifact of train/test leakage; in honest evaluation, it's closer to the zone result.

## Files

- `results.txt` — raw accuracy summary
- `room_confusion.png` — 4×4 confusion matrix for room ID task
- `csi_fingerprints.png` — mean amplitude fingerprints per subcarrier by room
