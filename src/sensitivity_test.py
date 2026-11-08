
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from features import extract_features

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "UCI HAR Dataset"

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

# Participant 1 versus everyone else
target = (subjects == 1) & (activities == 1)
others = (subjects != 1) & (activities == 1)

target_signals = all_signals[target]
other_signals = all_signals[others]

# Keep a gap between personal training and testing
n = len(target_signals)
personal_train = target_signals[:int(n * 0.6)]
test_signals = target_signals[int(n * 0.8):]

def train(signals):
    X = extract_features(signals)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )
    model.fit(X_scaled)

    return scaler, model

personal_scaler, personal_model = train(personal_train)
population_scaler, population_model = train(other_signals)

def detection_rate(scaler, model, signals):
    X = scaler.transform(extract_features(signals))
    return np.mean(model.predict(X) == -1) * 100

# Test increasingly large acceleration changes
change_levels = [0, 5, 10, 20, 30, 40, 50]

personal_rates = []
population_rates = []

print("\n=== GaitTrace Sensitivity Experiment ===")
print(f"{'Change':>8s} | {'Personal':>10s} | {'Population':>10s}")
print("-" * 36)

for change in change_levels:
    modified = test_signals.copy()

    # Increase acceleration by the selected percentage
    modified[:, :3, :] *= 1 + change / 100

    p = detection_rate(
        personal_scaler, personal_model, modified
    )
    g = detection_rate(
        population_scaler, population_model, modified
    )

    personal_rates.append(p)
    population_rates.append(g)

    print(f"{change:7d}% | {p:9.1f}% | {g:9.1f}%")

# Draw sensitivity curves
plt.figure(figsize=(10, 6))

plt.plot(
    change_levels, personal_rates,
    marker="o", linewidth=2.5,
    label="Personalized model"
)

plt.plot(
    change_levels, population_rates,
    marker="s", linewidth=2.5,
    label="Population model"
)

plt.title(
    "GaitTrace: Sensitivity to Increasing Sensor Changes",
    fontsize=14, fontweight="bold"
)
plt.xlabel("Artificial acceleration increase (%)")
plt.ylabel("Walking windows flagged (%)")
plt.ylim(-5, 105)
plt.xticks(change_levels)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

results_dir = ROOT / "results"
results_dir.mkdir(exist_ok=True)

output = results_dir / "sensitivity_curve.png"
plt.savefig(output, dpi=200)

print(f"\nSaved graph: {output}")

plt.show()
