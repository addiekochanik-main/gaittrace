
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

signals = np.stack([
    np.loadtxt(
        DATA / "train/Inertial Signals" / f"{name}_train.txt"
    )
    for name in names
], axis=1)

walking = activities == 1
participants = np.unique(subjects[walking])

SENSORS = {
    "Acceleration only": slice(0, 3),
    "Gyroscope only": slice(3, 6),
    "Both sensors": slice(0, 6)
}

CHANGES = {
    "Acceleration +20%": ("acc", 0.20),
    "Rotation +20%": ("gyro", 0.20),
    "Both +20%": ("both", 0.20)
}

def train_model(data):
    features = extract_features(data)
    scaler = StandardScaler()
    transformed = scaler.fit_transform(features)

    model = IsolationForest(
        n_estimators=150,
        random_state=42
    )
    model.fit(transformed)

    return scaler, model

def score(data, scaler, model):
    features = extract_features(data)
    transformed = scaler.transform(features)
    return -model.score_samples(transformed)

def standardize(values, reference):
    spread = max(np.std(reference), 1e-8)
    return (values - np.mean(reference)) / spread

def perturb(data, change_type, amount):
    changed = data.copy()

    if change_type in ("acc", "both"):
        changed[:, :3, :] *= 1 + amount

    if change_type in ("gyro", "both"):
        changed[:, 3:, :] *= 1 + amount

    return changed

records = []

print("\n=== GaitTrace Sensor Ablation ===")

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

    personal_train = person[:int(n * 0.5)]
    validation = person[int(n * 0.5):int(n * 0.7)]
    test = person[int(n * 0.8):]

    for sensor_name, channels in SENSORS.items():
        train_subset = personal_train[:, channels, :]
        population_subset = others[:, channels, :]
        validation_subset = validation[:, channels, :]
        test_subset = test[:, channels, :]

        p_scaler, p_model = train_model(train_subset)
        g_scaler, g_model = train_model(population_subset)

        p_val = score(validation_subset, p_scaler, p_model)
        g_val = score(validation_subset, g_scaler, g_model)

        p_normal = standardize(
            score(test_subset, p_scaler, p_model), p_val
        )
        g_normal = standardize(
            score(test_subset, g_scaler, g_model), g_val
        )

        normal_fusion = 0.5 * p_normal + 0.5 * g_normal

        for change_name, (change_type, amount) in CHANGES.items():
            changed = perturb(test, change_type, amount)
            changed_subset = changed[:, channels, :]

            p_changed = standardize(
                score(changed_subset, p_scaler, p_model),
                p_val
            )
            g_changed = standardize(
                score(changed_subset, g_scaler, g_model),
                g_val
            )

            changed_fusion = (
                0.5 * p_changed + 0.5 * g_changed
            )

            labels = np.concatenate([
                np.zeros(len(test)),
                np.ones(len(test))
            ])

            values = np.concatenate([
                normal_fusion, changed_fusion
            ])

            auc = roc_auc_score(labels, values)

            records.append({
                "participant": int(participant),
                "sensors": sensor_name,
                "change": change_name,
                "auroc": float(auc)
            })

    print(f"Completed participant {participant}")

csv_path = RESULTS / "sensor_ablation_results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["participant", "sensors", "change", "auroc"]
    )
    writer.writeheader()
    writer.writerows(records)

print("\n=== Mean AUROC Across Participants ===")
print(f"{'Change':<22} | {'Acceleration':>12} | {'Gyroscope':>10} | {'Both':>8}")
print("-" * 64)

summary = {}

for change_name in CHANGES:
    summary[change_name] = {}

    for sensor_name in SENSORS:
        values = [
            row["auroc"]
            for row in records
            if row["change"] == change_name
            and row["sensors"] == sensor_name
        ]

        summary[change_name][sensor_name] = np.mean(values)

    s = summary[change_name]

    print(
        f"{change_name:<22} | "
        f"{s['Acceleration only']:>12.3f} | "
        f"{s['Gyroscope only']:>10.3f} | "
        f"{s['Both sensors']:>8.3f}"
    )

# Grouped bar chart
fig, ax = plt.subplots(figsize=(11, 6))

x = np.arange(len(CHANGES))
width = 0.25

for i, sensor_name in enumerate(SENSORS):
    values = [
        summary[change][sensor_name]
        for change in CHANGES
    ]

    ax.bar(
        x + (i - 1) * width,
        values,
        width,
        label=sensor_name
    )

ax.axhline(
    0.5, color="gray", linestyle="--",
    label="Chance-level ranking"
)

ax.set_xticks(x)
ax.set_xticklabels(list(CHANGES))
ax.set_ylim(0, 1.05)
ax.set_ylabel("Mean AUROC")
ax.set_title(
    "GaitTrace: Which Motion Sensors Matter?",
    fontweight="bold"
)
ax.legend()
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()

plot_path = RESULTS / "sensor_ablation.png"
fig.savefig(plot_path, dpi=200)

print(f"\nSaved: {csv_path}")
print(f"Saved: {plot_path}")

plt.show()
