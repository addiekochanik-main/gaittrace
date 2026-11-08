
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
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

signal_dir = DATA / "train" / "Inertial Signals"

signals = np.stack([
    np.loadtxt(signal_dir / f"{name}_train.txt")
    for name in names
], axis=1)

walking = activities == 1
participants = np.unique(subjects[walking])

SEVERITIES = [0, 5, 10, 20, 30, 40, 50]

SCENARIOS = [
    "Sensor noise",
    "Accelerometer dropout",
    "Gyroscope drift",
    "Accelerometer gain error"
]

def train_model(data):
    X = extract_features(data)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=150,
        random_state=42
    )
    model.fit(X_scaled)

    return scaler, model

def anomaly_score(data, scaler, model):
    X = scaler.transform(extract_features(data))
    return -model.score_samples(X)

def normalize(values, reference):
    return (
        values - np.mean(reference)
    ) / max(np.std(reference), 1e-8)

def sabotage(data, scenario, severity, rng):
    changed = data.copy()

    if severity == 0:
        return changed

    amount = severity / 100

    if scenario == "Sensor noise":
        # Noise scaled to each channel's variability
        scale = np.std(
            data, axis=(0, 2), keepdims=True
        )
        noise = rng.normal(
            0, 1, size=data.shape
        )
        changed += amount * scale * noise

    elif scenario == "Accelerometer dropout":
        # Progressive signal loss on X acceleration
        changed[:, 0, :] *= (1 - amount)

    elif scenario == "Gyroscope drift":
        # Increasing offset through each recording
        t = np.linspace(0, 1, data.shape[2])

        gyro_scale = np.std(
            data[:, 3, :],
        )

        changed[:, 3, :] += (
            amount * gyro_scale * t[None, :]
        )

    elif scenario == "Accelerometer gain error":
        # Scale only X acceleration
        changed[:, 0, :] *= (1 + amount)

    return changed

records = []

print("\n=== GaitTrace Sensor Sabotage Study ===")

for participant in participants:
    person = signals[
        walking & (subjects == participant)
    ]
    others = signals[
        walking & (subjects != participant)
    ]

    if len(person) < 40:
        continue

    n = len(person)

    train = person[:int(n * 0.5)]
    validation = person[int(n * 0.5):int(n * 0.7)]
    test = person[int(n * 0.8):]

    p_scaler, p_model = train_model(train)
    g_scaler, g_model = train_model(others)

    p_val = anomaly_score(
        validation, p_scaler, p_model
    )
    g_val = anomaly_score(
        validation, g_scaler, g_model
    )

    p_original = normalize(
        anomaly_score(test, p_scaler, p_model),
        p_val
    )
    g_original = normalize(
        anomaly_score(test, g_scaler, g_model),
        g_val
    )

    fusion_original = (
        0.5 * p_original + 0.5 * g_original
    )

    for scenario in SCENARIOS:
        for severity in SEVERITIES:
            rng = np.random.default_rng(
                int(participant) * 1000
                + severity
                + SCENARIOS.index(scenario) * 100
            )

            modified = sabotage(
                test, scenario, severity, rng
            )

            p_changed = normalize(
                anomaly_score(
                    modified, p_scaler, p_model
                ),
                p_val
            )
            g_changed = normalize(
                anomaly_score(
                    modified, g_scaler, g_model
                ),
                g_val
            )

            fusion_changed = (
                0.5 * p_changed + 0.5 * g_changed
            )

            labels = np.concatenate([
                np.zeros(len(test)),
                np.ones(len(test))
            ])

            comparisons = {
                "Personalized": (
                    p_original, p_changed
                ),
                "Population": (
                    g_original, g_changed
                ),
                "Hybrid": (
                    fusion_original, fusion_changed
                )
            }

            for model_name, (original, changed) in comparisons.items():
                values = np.concatenate([
                    original, changed
                ])

                auc = roc_auc_score(labels, values)

                records.append({
                    "participant": int(participant),
                    "scenario": scenario,
                    "severity": severity,
                    "model": model_name,
                    "auroc": float(auc)
                })

    print(f"Completed participant {participant}")

csv_path = RESULTS / "sensor_sabotage_results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "participant",
            "scenario",
            "severity",
            "model",
            "auroc"
        ]
    )
    writer.writeheader()
    writer.writerows(records)

print("\n=== Hybrid Model AUROC by Scenario ===")
print(
    f"{'Scenario':<28} | "
    f"{'10%':>7} | {'30%':>7} | {'50%':>7}"
)
print("-" * 58)

for scenario in SCENARIOS:
    values = []

    for severity in [10, 30, 50]:
        selected = [
            r["auroc"] for r in records
            if r["scenario"] == scenario
            and r["severity"] == severity
            and r["model"] == "Hybrid"
        ]

        values.append(np.mean(selected))

    print(
        f"{scenario:<28} | "
        f"{values[0]:7.3f} | "
        f"{values[1]:7.3f} | "
        f"{values[2]:7.3f}"
    )

# Plot hybrid AUROC across fault severity
fig, ax = plt.subplots(figsize=(11, 6))

for scenario in SCENARIOS:
    mean_scores = []

    for severity in SEVERITIES:
        selected = [
            r["auroc"] for r in records
            if r["scenario"] == scenario
            and r["severity"] == severity
            and r["model"] == "Hybrid"
        ]

        mean_scores.append(np.mean(selected))

    ax.plot(
        SEVERITIES,
        mean_scores,
        marker="o",
        linewidth=2.5,
        label=scenario
    )

ax.axhline(
    0.5, linestyle="--", color="gray",
    label="Chance-level ranking"
)

ax.set_title(
    "GaitTrace: Robustness to Simulated Sensor Faults",
    fontsize=14,
    fontweight="bold"
)
ax.set_xlabel("Simulated fault severity (%)")
ax.set_ylabel("Mean AUROC")
ax.set_xticks(SEVERITIES)
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()

plot_path = RESULTS / "sensor_sabotage.png"
fig.savefig(plot_path, dpi=200)

print(f"\nSaved: {csv_path}")
print(f"Saved: {plot_path}")

plt.show()
