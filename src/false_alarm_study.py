
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
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

BUDGETS = [0.01, 0.05, 0.10]
CHANGES = [0.10, 0.20, 0.30]
MODELS = ["Personalized", "Population", "Hybrid"]

def fit_model(data):
    X = extract_features(data)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=150,
        random_state=42
    )
    model.fit(X_scaled)

    return scaler, model

def score(data, scaler, model):
    X = scaler.transform(extract_features(data))
    return -model.score_samples(X)

def normalize(values, reference):
    spread = max(np.std(reference), 1e-8)
    return (values - np.mean(reference)) / spread

records = []

print("\n=== GaitTrace False Alarm Budget Study ===")

for participant in participants:
    person = signals[
        (activities == 1) & (subjects == participant)
    ]

    others = signals[
        (activities == 1) & (subjects != participant)
    ]

    if len(person) < 40:
        continue

    n = len(person)

    train = person[:int(n * 0.5)]
    validation = person[int(n * 0.5):int(n * 0.7)]
    test = person[int(n * 0.8):]

    p_scaler, p_model = fit_model(train)
    g_scaler, g_model = fit_model(others)

    p_val = score(validation, p_scaler, p_model)
    g_val = score(validation, g_scaler, g_model)

    p_test = score(test, p_scaler, p_model)
    g_test = score(test, g_scaler, g_model)

    val_scores = {
        "Personalized": normalize(p_val, p_val),
        "Population": normalize(g_val, g_val),
        "Hybrid": (
            normalize(p_val, p_val)
            + normalize(g_val, g_val)
        ) / 2
    }

    test_scores = {
        "Personalized": normalize(p_test, p_val),
        "Population": normalize(g_test, g_val),
        "Hybrid": (
            normalize(p_test, p_val)
            + normalize(g_test, g_val)
        ) / 2
    }

    for change in CHANGES:
        modified = test.copy()
        modified[:, :3, :] *= 1 + change

        p_changed = score(
            modified, p_scaler, p_model
        )
        g_changed = score(
            modified, g_scaler, g_model
        )

        changed_scores = {
            "Personalized": normalize(p_changed, p_val),
            "Population": normalize(g_changed, g_val),
            "Hybrid": (
                normalize(p_changed, p_val)
                + normalize(g_changed, g_val)
            ) / 2
        }

        for budget in BUDGETS:
            for model_name in MODELS:
                reference = val_scores[model_name]

                threshold = np.quantile(
                    reference, 1 - budget
                )

                false_alarm = np.mean(
                    test_scores[model_name] > threshold
                )

                detection = np.mean(
                    changed_scores[model_name] > threshold
                )

                records.append({
                    "participant": int(participant),
                    "budget": budget,
                    "change": int(change * 100),
                    "model": model_name,
                    "false_alarm_rate": float(false_alarm),
                    "detection_rate": float(detection)
                })

    print(f"Completed participant {participant}")

csv_path = RESULTS / "false_alarm_results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "participant",
            "budget",
            "change",
            "model",
            "false_alarm_rate",
            "detection_rate"
        ]
    )
    writer.writeheader()
    writer.writerows(records)

print("\n=== Mean Detection Rate at 20% Change ===")
print(
    f"{'Budget':<10} | "
    f"{'Model':<13} | "
    f"{'False alarms':>12} | "
    f"{'Detection':>10}"
)
print("-" * 57)

for budget in BUDGETS:
    for model_name in MODELS:
        selected = [
            r for r in records
            if r["budget"] == budget
            and r["change"] == 20
            and r["model"] == model_name
        ]

        false_alarm = np.mean([
            r["false_alarm_rate"] for r in selected
        ]) * 100

        detection = np.mean([
            r["detection_rate"] for r in selected
        ]) * 100

        print(
            f"{budget * 100:>5.0f}%     | "
            f"{model_name:<13} | "
            f"{false_alarm:>11.1f}% | "
            f"{detection:>9.1f}%"
        )

# Plot detection vs actual false alarms
fig, ax = plt.subplots(figsize=(10, 6))

colors = {
    "Personalized": "tab:blue",
    "Population": "tab:orange",
    "Hybrid": "tab:green"
}

for model_name in MODELS:
    xs = []
    ys = []

    for budget in BUDGETS:
        selected = [
            r for r in records
            if r["budget"] == budget
            and r["change"] == 20
            and r["model"] == model_name
        ]

        xs.append(
            np.mean([
                r["false_alarm_rate"] for r in selected
            ]) * 100
        )

        ys.append(
            np.mean([
                r["detection_rate"] for r in selected
            ]) * 100
        )

    ax.plot(
        xs, ys,
        marker="o",
        linewidth=2.5,
        label=model_name,
        color=colors[model_name]
    )

    for x, y, budget in zip(xs, ys, BUDGETS):
        ax.annotate(
            f"{budget * 100:.0f}% target",
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8
        )

ax.plot(
    [0, 100], [0, 100],
    "--", color="gray", alpha=0.5
)

ax.set_title(
    "GaitTrace: Detection vs False Alarm Tradeoff",
    fontweight="bold"
)
ax.set_xlabel("Actual false alarm rate (%)")
ax.set_ylabel("Detection rate at +20% acceleration (%)")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.grid(alpha=0.2)
ax.legend()

fig.tight_layout()

plot_path = RESULTS / "false_alarm_tradeoff.png"
fig.savefig(plot_path, dpi=200)

print(f"\nSaved: {csv_path}")
print(f"Saved: {plot_path}")

plt.show()
