
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from features import extract_features

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "UCI HAR Dataset"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Load labels and participant IDs
activities = np.loadtxt(
    DATA / "train/y_train.txt", dtype=int
)
subjects = np.loadtxt(
    DATA / "train/subject_train.txt", dtype=int
)

names = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z"
]

signals = np.stack([
    np.loadtxt(
        DATA / "train/Inertial Signals" / f"{name}_train.txt"
    )
    for name in names
], axis=1)

# Only walking windows
walking_indices = np.where(activities == 1)[0]

X = extract_features(signals[walking_indices])
y = subjects[walking_indices]

X_scaled = StandardScaler().fit_transform(X)

distances = pairwise_distances(
    X_scaled, metric="euclidean"
)

# Original positions in the full UCI HAR train partition
positions = walking_indices

same_person = y[:, None] == y[None, :]
position_gap = np.abs(
    positions[:, None] - positions[None, :]
)

# Exclude matches that are close in the dataset's row order.
# This is a proxy for temporal proximity, not verified timestamps.
exclusion_gaps = [0, 1, 2, 5, 10, 20, 50]

records = []

print("\n=== GaitTrace Fingerprint Leakage Test ===")
print(f"Walking windows: {len(y)}")
print(f"Participants: {len(np.unique(y))}")

for gap in exclusion_gaps:
    adjusted = distances.copy()

    # Never allow a recording to match itself
    np.fill_diagonal(adjusted, np.inf)

    # Exclude nearby windows belonging to the same person
    too_close = same_person & (position_gap <= gap)
    adjusted[too_close] = np.inf

    nearest = np.argmin(adjusted, axis=1)
    nearest_distances = adjusted[
        np.arange(len(y)), nearest
    ]

    valid = np.isfinite(nearest_distances)

    accuracy = np.mean(
        y[nearest[valid]] == y[valid]
    )

    records.append({
        "excluded_row_gap": gap,
        "accuracy": float(accuracy),
        "valid_windows": int(np.sum(valid))
    })

    print(
        f"Exclude +/- {gap:>2} rows | "
        f"Accuracy: {accuracy * 100:5.1f}% | "
        f"Evaluated: {np.sum(valid)}"
    )

csv_path = RESULTS / "fingerprint_leakage_results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "excluded_row_gap",
            "accuracy",
            "valid_windows"
        ]
    )
    writer.writeheader()
    writer.writerows(records)

fig, ax = plt.subplots(figsize=(10, 6))

gaps = [r["excluded_row_gap"] for r in records]
accuracies = [
    r["accuracy"] * 100 for r in records
]

ax.plot(
    gaps,
    accuracies,
    marker="o",
    linewidth=2.5,
    color="tab:blue"
)

for gap, accuracy in zip(gaps, accuracies):
    ax.annotate(
        f"{accuracy:.1f}%",
        (gap, accuracy),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )

chance = 100 / len(np.unique(y))

ax.axhline(
    chance,
    linestyle="--",
    color="gray",
    label=f"Uniform random baseline ({chance:.1f}%)"
)

ax.set_title(
    "GaitTrace: Does Fingerprint Accuracy Survive "
    "Nearby-Window Exclusion?",
    fontweight="bold"
)
ax.set_xlabel(
    "Excluded neighboring rows on each side"
)
ax.set_ylabel(
    "Nearest-neighbor participant accuracy (%)"
)
ax.set_ylim(0, 105)
ax.grid(alpha=0.2)
ax.legend()

fig.tight_layout()

plot_path = RESULTS / "fingerprint_leakage.png"
fig.savefig(plot_path, dpi=200)

print(f"\nSaved: {csv_path}")
print(f"Saved: {plot_path}")

plt.show()
