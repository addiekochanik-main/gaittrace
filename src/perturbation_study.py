
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from features import extract_features

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "UCI HAR Dataset"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

activities = np.loadtxt(
    DATA_DIR / "train" / "y_train.txt", dtype=int
)
subjects = np.loadtxt(
    DATA_DIR / "train" / "subject_train.txt", dtype=int
)

names = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z"
]

signal_dir = DATA_DIR / "train" / "Inertial Signals"

all_signals = np.stack(
    [
        np.loadtxt(signal_dir / f"{name}_train.txt")
        for name in names
    ],
    axis=1
)

walking = activities == 1
participant_ids = np.unique(subjects[walking])

LEVELS = [0, 5, 10, 20, 30, 40, 50]
TARGET_FAR = 0.05

CHANGE_TYPES = [
    "Increase acceleration",
    "Decrease acceleration",
    "Increase rotation",
    "Add sensor noise"
]

def fit_model(signals):
    scaler = StandardScaler()
    X = scaler.fit_transform(extract_features(signals))

    model = IsolationForest(
        n_estimators=150,
        random_state=42
    )
    model.fit(X)

    return scaler, model

def score(scaler, model, signals):
    X = scaler.transform(extract_features(signals))
    return -model.score_samples(X)

def modify(signals, change_type, level, seed):
    changed = signals.copy()
    fraction = level / 100

    if change_type == "Increase acceleration":
        changed[:, :3, :] *= 1 + fraction

    elif change_type == "Decrease acceleration":
        changed[:, :3, :] *= 1 - fraction

    elif change_type == "Increase rotation":
        changed[:, 3:, :] *= 1 + fraction

    elif change_type == "Add sensor noise":
        rng = np.random.default_rng(seed)

        # Noise magnitude relative to each axis's
        # variability in the original test recordings.
        axis_std = np.std(
            signals, axis=(0, 2), keepdims=True
        )

        changed += rng.normal(
            0, 1, size=changed.shape
        ) * axis_std * fraction

    return changed

records = []

print("\n=== GaitTrace Perturbation Study ===")

for person_id in participant_ids:
    person = all_signals[
        walking & (subjects == person_id)
    ]

    others = all_signals[
        walking & (subjects != person_id)
    ]

    if len(person) < 40:
        continue

    n = len(person)

    train_end = int(n * 0.5)
    val_end = int(n * 0.7)
    test_start = int(n * 0.8)

    personal_train = person[:train_end]
    validation = person[train_end:val_end]
    test = person[test_start:]

    p_scaler, p_model = fit_model(personal_train)
    g_scaler, g_model = fit_model(others)

    p_threshold = np.quantile(
        score(p_scaler, p_model, validation),
        1 - TARGET_FAR
    )

    g_threshold = np.quantile(
        score(g_scaler, g_model, validation),
        1 - TARGET_FAR
    )

    for change_type in CHANGE_TYPES:
        for level in LEVELS:
            modified = modify(
                test,
                change_type,
                level,
                seed=int(person_id) * 1000 + level
            )

            p_scores = score(
                p_scaler, p_model, modified
            )
            g_scores = score(
                g_scaler, g_model, modified
            )

            records.append({
                "participant": int(person_id),
                "change_type": change_type,
                "level": level,
                "personal_rate": float(
                    np.mean(p_scores > p_threshold) * 100
                ),
                "population_rate": float(
                    np.mean(g_scores > g_threshold) * 100
                )
            })

    print(f"Completed participant {person_id}")

# Save every participant's results
csv_path = RESULTS / "perturbation_results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(records[0].keys())
    )
    writer.writeheader()
    writer.writerows(records)

print(f"\nSaved: {csv_path}")

# Create four panels of sensitivity curves
fig, axes = plt.subplots(
    2, 2, figsize=(12, 9)
)

for ax, change_type in zip(
    axes.flat, CHANGE_TYPES
):
    personal_means = []
    population_means = []

    for level in LEVELS:
        matching = [
            r for r in records
            if r["change_type"] == change_type
            and r["level"] == level
        ]

        personal_means.append(
            np.mean([
                r["personal_rate"] for r in matching
            ])
        )

        population_means.append(
            np.mean([
                r["population_rate"] for r in matching
            ])
        )

    ax.plot(
        LEVELS, personal_means,
        marker="o", label="Personalized"
    )
    ax.plot(
        LEVELS, population_means,
        marker="s", label="Population"
    )

    ax.set_title(change_type)
    ax.set_xlabel("Simulated change (%)")
    ax.set_ylabel("Windows flagged (%)")
    ax.set_ylim(-5, 105)
    ax.set_xticks(LEVELS)
    ax.grid(alpha=0.25)
    ax.legend()

    print(f"\n{change_type}")
    for level, p, g in zip(
        LEVELS, personal_means, population_means
    ):
        print(
            f"  {level:2d}%: "
            f"Personal {p:5.1f}% | "
            f"Population {g:5.1f}%"
        )

fig.suptitle(
    "GaitTrace: Sensitivity Across 21 Participants",
    fontsize=15, fontweight="bold"
)
fig.tight_layout()

plot_path = RESULTS / "perturbation_study.png"
fig.savefig(plot_path, dpi=200)

print(f"\nSaved graph: {plot_path}")
plt.show()
