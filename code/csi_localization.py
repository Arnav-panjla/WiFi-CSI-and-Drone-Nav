"""CSI indoor localization: basic feature extraction + simple classifiers.

Dataset: per-location .mat files holding `myData` of shape (3 antennas, 30
subcarriers, ~N packets). Filenames encode the (x,y) grid coordinate, e.g.
`coordinate715` -> (7,15). We run two tasks: room identification and
within-room zone localization.
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

OUT_DIR = "results"

DATA_ROOT = "CSI-dataset-for-indoor-localization-main"
WINDOW = 50           # packets per window -> one feature sample
MAX_WIN = 8           # cap windows per file so big/small files stay balanced
RNG = np.random.RandomState(0)

# Each room maps to its coordinate folders (real/amplitude part only).
ROOMS = {
    "Lab": "Lab Dataset",
    "MeetingRoom": "Meeting Room Dataset",
    "ConferenceRoom": "Conference Room",
    "miniLab": "miniLab",
}


def parse_coord(fname):
    """Extract trailing integer -> (x, y); last 2 digits are y, rest is x."""
    n = int(re.search(r"(\d+)\.mat$", fname).group(1))
    return n // 100, n % 100


def window_features(arr):
    """arr: (3,30,N) amplitude. Yield per-window feature vectors.

    Features = mean and std of amplitude across the packets in the window,
    flattened over the 3*30 antenna-subcarrier pairs (180 dims total).
    """
    n = arr.shape[2]
    for i, start in enumerate(range(0, n - WINDOW + 1, WINDOW)):
        if i >= MAX_WIN:
            break
        w = arr[:, :, start:start + WINDOW]
        feat = np.concatenate([w.mean(axis=2).ravel(), w.std(axis=2).ravel()])
        yield feat


def load_room(room_dir):
    """Return (X, coords, groups) for every window in a room's .mat files.

    `groups` is a per-file id used to keep all windows of one location on the
    same side of the train/test split (prevents location leakage).
    """
    pattern = os.path.join(DATA_ROOT, room_dir, "**", "*.mat")
    mats = sorted(glob.glob(pattern, recursive=True))
    # Skip the imaginary_part folder so each location is counted once.
    mats = [m for m in mats if "imaginary" not in m.lower()]
    X, coords, groups = [], [], []
    for gid, m in enumerate(mats):
        arr = sio.loadmat(m)["myData"]
        if arr.shape[2] < WINDOW:
            continue
        # ~5% of files carry inf/NaN samples; replace with the finite mean.
        if not np.isfinite(arr).all():
            fill = np.nanmean(arr[np.isfinite(arr)]) if np.isfinite(arr).any() else 0.0
            arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
        c = parse_coord(m)
        for feat in window_features(arr):
            X.append(feat)
            coords.append(c)
            groups.append(gid)
    return np.array(X), coords, np.array(groups)


def evaluate(X, y, label, groups=None):
    """Train RF + kNN on a scaled split and print accuracies.

    With `groups`, windows from one file stay on one side (honest split);
    without it we fall back to a random split (optimistic, leaks locations).
    """
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
        tr, te = next(gss.split(X, y, groups))
    else:
        idx = RNG.permutation(len(X))
        cut = int(len(X) * 0.7)
        tr, te = idx[:cut], idx[cut:]
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
    print(f"\n=== {label} ===")
    print(f"samples={len(X)}  features={X.shape[1]}  classes={len(set(y))}")
    results = {}
    last_pred = None
    for name, clf in [
        ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=0)),
        ("kNN(k=5)", KNeighborsClassifier(n_neighbors=5)),
    ]:
        clf.fit(Xtr, np.array(y)[tr])
        pred = clf.predict(Xte)
        acc = accuracy_score(np.array(y)[te], pred)
        results[name] = acc
        print(f"  {name:14s} accuracy = {acc:.3f}")
        if name == "RandomForest":
            last_pred = (np.array(y)[te], pred)
    return results, len(X), len(set(y)), last_pred


def plot_confusion(y_true, y_pred, labels, title, path):
    """Save a confusion-matrix heatmap (RandomForest predictions)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # ---- Load every room ----
    data = {}
    offset = 0
    for room, d in ROOMS.items():
        X, coords, groups = load_room(d)
        if len(X):
            data[room] = (X, coords, groups + offset)  # globally-unique file ids
            offset += int(groups.max()) + 1
            print(f"loaded {room:15s} windows={len(X):5d} locations={len(set(coords))}")

    summary = []

    # ---- Task A: room identification (grouped split, no location leakage) ----
    Xa = np.vstack([data[r][0] for r in data])
    ya = [r for r in data for _ in range(len(data[r][0]))]
    ga = np.concatenate([data[r][2] for r in data])
    res, n, k, pred = evaluate(Xa, ya, f"Task A: Room identification ({len(data)} rooms)", groups=ga)
    plot_confusion(pred[0], pred[1], list(data), "Room identification (RF)",
                   os.path.join(OUT_DIR, "room_confusion.png"))
    summary.append((f"Room ID ({len(data)}-class)", n, k, res))

    # ---- Task B: within-room zone localization (Lab, grouped split) ----
    # Bin the Lab grid into a 3x3 set of zones to keep it a sane #classes.
    Xb, coords, gb = data["Lab"]
    xs = np.array([c[0] for c in coords])
    ys = np.array([c[1] for c in coords])
    xz = np.digitize(xs, np.quantile(xs, [1/3, 2/3]))
    yz = np.digitize(ys, np.quantile(ys, [1/3, 2/3]))
    yb = [f"z{a}{b}" for a, b in zip(xz, yz)]
    res, n, k, _ = evaluate(Xb, yb, "Task B: Lab zone localization (3x3 grid, grouped split)", groups=gb)
    summary.append(("Lab zone (9-class)", n, k, res))

    # ---- Task C: exact-coordinate fingerprint (random split, OPTIMISTIC) ----
    # Each location is its own class, so a grouped split is impossible; the
    # random split lets windows of a location sit in both train and test.
    yc = [f"{c[0]}_{c[1]}" for c in coords]
    res, n, k, _ = evaluate(Xb, yc, "Task C: Lab exact coordinate (random split, leaky)")
    summary.append(("Lab exact coord*", n, k, res))

    # ---- Fingerprint visualization: mean amplitude per subcarrier ----
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for room in data:
        ax.plot(data[room][0][:, :90].mean(0).reshape(3, 30).mean(0), label=room)
    ax.set_xlabel("subcarrier"); ax.set_ylabel("mean amplitude")
    ax.set_title("Mean CSI amplitude per subcarrier"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "csi_fingerprints.png"), dpi=120)
    plt.close(fig)

    lines = ["================ SUMMARY ================"]
    for name, n, k, res in summary:
        accs = "  ".join(f"{m}={a:.3f}" for m, a in res.items())
        lines.append(f"{name:22s} n={n:5d} classes={k:4d} | {accs}")
    lines.append("* random split leaks locations across train/test (optimistic).")
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(OUT_DIR, "results.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
