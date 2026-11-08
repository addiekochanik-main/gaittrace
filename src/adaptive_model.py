
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
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

LEVELS = [5, 10, 20, 30, 40, 50]

def fit_model(signals):
    X = extract_features(signals)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=150,
        random_state=42
    )
    model.fit(X_scaled)

    return scaler, model

def scores(scaler, model, signals):
    X = scaler.transform(extract_features(signals))
    return -model.score_samples(X)

def normalize_scores(values, validation_scores):
    # Convert scores to comparable units using
    # the normal validation score distribution.
    center = np.mean(validation_scores)
    spread = np.std(validation_scores)

    if spread < 1e-8:
        spread = 1.0

    return (values - center) / spread

records = []

print("\n=== GaitTrace Adaptive Model Study ===")

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

    # Normal validation scores calibrate each model
    p_val = scores(p_scaler, p_model, validation)
    g_val = scores(g_scaler, g_model, validation)

    # Score the original test recordings
    p_original = normalize_scores(
        scores(p_scaler, p_model, test), p_val
    )
    g_original = normalize_scores(
        scores(g_scaler, g_model, test), g_val
    )

    for level in LEVELS:
        modified = test.copy()
        modified[:, :3, :] *= 1 + level / 100

        p_changed = normalize_scores(
            scores(p_scaler, p_model, modified), p_val
        )
        g_changed = normalize_scores(
            scores(g_scaler, g_model, modified), g_val
        )

        # Fixed fusion: 50% personal + 50% population
        fusion_original = (
            0.5 * p_original + 0.5 * g_original
        )
        fusion_changed = (
            0.5 * p_changed + 0.5 * g_changed
        )

        labels = np.concatenate([
            np.zeros(len(test)),
            np.ones(len(test))
        ])

        model_scores = {
            "Personalized": np.concatenate([
                p_original, p_changed
            ]),
            "Population": np.concatenate([
                g_original, g_changed
            ]),
            "Adaptive fusion": np.concatenate([
                fusion_original, fusion_changed
            ])
        }

        for model_name, values in model_scores.items():
            auc = roc_auc_score(labels, values)

            records.append({
                "participant": int(person_id),
                "change": level,
                "model": model_name,
                "auroc": float(auc)
            })

    print(f"Completed participant {person_id}")

# Save detailed results
csv_path = RESULTS / "adaptive_results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(records[0].keys())
    )
    writer.writeheader()
    writer.writerows(records)

print(f"\nSaved: {csv_path}")

# Summarize mean AUROC across participants
model_names = [
    "Personalized",
    "Population",
    "Adaptive fusion"
]

means = {}

for model_name in model_names:
    means[model_name] = []

    for level in LEVELS:
        values = [
            r["auroc"] for r in records
            if r["model"] == model_name
            and r["change"] == level
        ]

        means[model_name].append(np.mean(values))

print("\n=== Average AUROC Across Participants ===")
print(
    f"{'Change':>8s} | "
    f"{'Personal':>9s} | "
    f"{'Population':>10s} | "
    f"{'Fusion':>9s}"
)
print("-" * 49)

for i, level in enumerate(LEVELS):
    print(
        f"{level:7d}% | "
        f"{means['Personalized'][i]:8.3f} | "
        f"{means['Population'][i]:9.3f} | "
        f"{means['Adaptive fusion'][i]:8.3f}"
    )

# Plot comparison
plt.figure(figsize=(10, 6))

for model_name in model_names:
    plt.plot(
        LEVELS,
        means[model_name],
        marker="o",
        linewidth=2.5,
        label=model_name
    )

plt.axhline(
    0.5, linestyle="--", color="gray",
    label="Chance-level ranking"
)

plt.title(
    "GaitTrace: Can Combining Models Improve Detection?",
    fontsize=14,
    fontweight="bold"
)
plt.xlabel("Simulated acceleration increase (%)")
plt.ylabel("Mean AUROC")
plt.xticks(LEVELS)
plt.ylim(0, 1.05)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plot_path = RESULTS / "adaptive_comparison.png"
plt.savefig(plot_path, dpi=200)

print(f"\nSaved graph: {plot_path}")
plt.show()
