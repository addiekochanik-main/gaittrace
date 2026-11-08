
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from features import extract_features

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "UCI HAR Dataset"

# Load the walking dataset
activities = np.loadtxt(DATA_DIR / "train" / "y_train.txt", dtype=int)
subjects = np.loadtxt(DATA_DIR / "train" / "subject_train.txt", dtype=int)

signal_dir = DATA_DIR / "train" / "Inertial Signals"

names = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]

all_signals = np.stack(
    [np.loadtxt(signal_dir / f"{name}_train.txt") for name in names],
    axis=1
)

walking = activities == 1
participant_ids = np.unique(subjects[walking])

CHANGE_LEVELS = [0, 5, 10, 20, 30, 40, 50]
TARGET_FALSE_ALARM = 0.05

results = []

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

def anomaly_scores(scaler, model, signals):
    X = scaler.transform(extract_features(signals))
    return -model.score_samples(X)

def calibrated_threshold(scaler, model, validation_signals):
    scores = anomaly_scores(scaler, model, validation_signals)
    return np.quantile(scores, 1 - TARGET_FALSE_ALARM)

print("\n=== GaitTrace Multi-Participant Evaluation ===")
print(f"Participants available: {len(participant_ids)}")

for person_id in participant_ids:

    person_signals = all_signals[
        walking & (subjects == person_id)
    ]

    other_signals = all_signals[
        walking & (subjects != person_id)
    ]

    # Use only participants with enough windows
    if len(person_signals) < 40:
        continue

    # Split participant data:
    # 50% training, 20% validation, 10% gap, 20% test
    n = len(person_signals)

    train_end = int(n * 0.5)
    val_end = int(n * 0.7)
    test_start = int(n * 0.8)

    personal_train = person_signals[:train_end]
    validation = person_signals[train_end:val_end]
    test = person_signals[test_start:]

    # Train personalized and population models
    p_scaler, p_model = fit_model(personal_train)
    g_scaler, g_model = fit_model(other_signals)

    # Calibrate both on the SAME person's normal
    # validation recordings at a 5% target alert rate
    p_threshold = calibrated_threshold(
        p_scaler, p_model, validation
    )

    g_threshold = calibrated_threshold(
        g_scaler, g_model, validation
    )

    for change in CHANGE_LEVELS:
        modified = test.copy()

        # Simulate stronger acceleration
        modified[:, :3, :] *= 1 + change / 100

        p_scores = anomaly_scores(
            p_scaler, p_model, modified
        )

        g_scores = anomaly_scores(
            g_scaler, g_model, modified
        )

        p_rate = np.mean(p_scores > p_threshold) * 100
        g_rate = np.mean(g_scores > g_threshold) * 100

        results.append({
            "participant": int(person_id),
            "change": change,
            "personal": p_rate,
            "population": g_rate,
        })

    print(f"Completed participant {person_id}")

# Aggregate results across participants
print("\n=== Average Detection Rates ===")
print(f"{'Change':>8s} | {'Personal':>10s} | {'Population':>10s}")
print("-" * 36)

personal_means = []
population_means = []

for change in CHANGE_LEVELS:
    rows = [r for r in results if r["change"] == change]

    p = np.mean([r["personal"] for r in rows])
    g = np.mean([r["population"] for r in rows])

    personal_means.append(p)
    population_means.append(g)

    print(f"{change:7d}% | {p:9.1f}% | {g:9.1f}%")

# Save detailed results
import csv

results_dir = ROOT / "results"
results_dir.mkdir(exist_ok=True)

csv_path = results_dir / "multi_person_results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["participant", "change", "personal", "population"]
    )
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved detailed results: {csv_path}")

# Plot average detection curves
plt.figure(figsize=(10, 6))

plt.plot(
    CHANGE_LEVELS, personal_means,
    marker="o", linewidth=2.5,
    label="Personalized model"
)

plt.plot(
    CHANGE_LEVELS, population_means,
    marker="s", linewidth=2.5,
    label="Population model"
)

plt.title(
    "GaitTrace: Multi-Participant Sensitivity Comparison",
    fontsize=14, fontweight="bold"
)

plt.xlabel("Simulated acceleration increase (%)")
plt.ylabel("Average windows flagged (%)")
plt.xticks(CHANGE_LEVELS)
plt.ylim(-5, 105)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plot_path = results_dir / "multi_person_comparison.png"
plt.savefig(plot_path, dpi=200)

print(f"Saved graph: {plot_path}")
plt.show()
