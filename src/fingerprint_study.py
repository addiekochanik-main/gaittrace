
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from features import extract_features

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "UCI HAR Dataset"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

activities = np.loadtxt(DATA / "train/y_train.txt", dtype=int)
subjects = np.loadtxt(DATA / "train/subject_train.txt", dtype=int)

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

walking = activities == 1

X = extract_features(signals[walking])
y = subjects[walking]

# Use one common feature scale for all participants
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compare each window against other windows
distances = pairwise_distances(
    X_scaled, metric="euclidean"
)

# Exclude self-matches
np.fill_diagonal(distances, np.inf)

same_person = y[:, None] == y[None, :]
different_person = ~same_person

# Exclude the diagonal from within-person comparisons
within_mask = same_person & ~np.eye(
    len(y), dtype=bool
)

within_distances = distances[within_mask]
between_distances = distances[different_person]

within_mean = np.mean(within_distances)
between_mean = np.mean(between_distances)

# Find the nearest non-identical window
nearest_indices = np.argmin(distances, axis=1)
predicted_subjects = y[nearest_indices]

accuracy = np.mean(predicted_subjects == y)

print("\n=== GaitTrace Gait Fingerprint Study ===")
print(f"Walking windows: {len(y)}")
print(f"Participants: {len(np.unique(y))}")
print(f"Mean within-person distance: {within_mean:.3f}")
print(f"Mean between-person distance: {between_mean:.3f}")
print(f"Between/within distance ratio: {between_mean / within_mean:.3f}")
print(f"Nearest-neighbor identity accuracy: {accuracy * 100:.1f}%")

# Per-participant analysis
records = []

for participant in np.unique(y):
    person_mask = y == participant

    within = distances[
        np.ix_(person_mask, person_mask)
    ]
    within = within[np.isfinite(within)]

    between = distances[
        np.ix_(person_mask, ~person_mask)
    ]

    records.append({
        "participant": int(participant),
        "within_distance": float(np.mean(within)),
        "between_distance": float(np.mean(between)),
        "distinctiveness_ratio": float(
            np.mean(between) / np.mean(within)
        )
    })

csv_path = RESULTS / "fingerprint_results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "participant",
            "within_distance",
            "between_distance",
            "distinctiveness_ratio"
        ]
    )
    writer.writeheader()
    writer.writerows(records)

# PCA visualization
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(
    1, 2, figsize=(14, 6)
)

# Show a few participants for readability
selected = np.unique(y)[:8]

for participant in selected:
    mask = y == participant

    axes[0].scatter(
        X_2d[mask, 0],
        X_2d[mask, 1],
        s=14,
        alpha=0.55,
        label=f"Participant {participant}"
    )

axes[0].set_title("Walking Feature Fingerprints")
axes[0].set_xlabel("Principal component 1")
axes[0].set_ylabel("Principal component 2")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.2)

# Distance comparison
rng = np.random.default_rng(42)
sample_size = min(
    5000,
    len(within_distances),
    len(between_distances)
)

within_sample = rng.choice(
    within_distances,
    size=sample_size,
    replace=False
)

between_sample = rng.choice(
    between_distances,
    size=sample_size,
    replace=False
)

axes[1].hist(
    within_sample,
    bins=45,
    alpha=0.6,
    density=True,
    label="Same participant"
)

axes[1].hist(
    between_sample,
    bins=45,
    alpha=0.6,
    density=True,
    label="Different participants"
)

axes[1].set_title("Within-Person vs Between-Person Distance")
axes[1].set_xlabel("Feature distance")
axes[1].set_ylabel("Density")
axes[1].legend()
axes[1].grid(alpha=0.2)

fig.suptitle(
    "GaitTrace: Are Walking Patterns Individual?",
    fontsize=15,
    fontweight="bold"
)

fig.tight_layout()

plot_path = RESULTS / "fingerprint_stability.png"
fig.savefig(plot_path, dpi=200)

print(f"\nSaved: {csv_path}")
print(f"Saved: {plot_path}")

plt.show()
