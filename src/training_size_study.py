
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

LEVELS = [0, 10, 20, 30, 50]
REPEATS = 5
TARGET_FAR = 0.05

def fit_model(signals):
    scaler = StandardScaler()
    X = scaler.fit_transform(extract_features(signals))

    model = IsolationForest(
        n_estimators=100,
        random_state=42
    )
    model.fit(X)

    return scaler, model

def scores(scaler, model, signals):
    X = scaler.transform(extract_features(signals))
    return -model.score_samples(X)

def test_model(scaler, model, validation, test):
    threshold = np.quantile(
        scores(scaler, model, validation),
        1 - TARGET_FAR
    )

    rates = []

    for level in LEVELS:
        changed = test.copy()
        changed[:, :3, :] *= 1 + level / 100

        rate = np.mean(
            scores(scaler, model, changed) > threshold
        ) * 100

        rates.append(rate)

    return rates

records = []

print("\n=== GaitTrace Training-Size Study ===")

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

    training_size = len(personal_train)

    # Model A: Personalized
    p_scaler, p_model = fit_model(personal_train)

    p_rates = test_model(
        p_scaler, p_model, validation, test
    )

    # Model B: Full population
    f_scaler, f_model = fit_model(others)

    f_rates = test_model(
        f_scaler, f_model, validation, test
    )

    for i, level in enumerate(LEVELS):
        records.append({
            "participant": int(person_id),
            "model": "Personalized",
            "repeat": 0,
            "level": level,
            "rate": float(p_rates[i])
        })

        records.append({
            "participant": int(person_id),
            "model": "Full population",
            "repeat": 0,
            "level": level,
            "rate": float(f_rates[i])
        })

    # Model C: Size-matched population
    for repeat in range(REPEATS):
        rng = np.random.default_rng(
            int(person_id) * 100 + repeat
        )

        selected = rng.choice(
            len(others),
            size=training_size,
            replace=False
        )

        sampled_signals = others[selected]

        m_scaler, m_model = fit_model(
            sampled_signals
        )

        m_rates = test_model(
            m_scaler, m_model, validation, test
        )

        for i, level in enumerate(LEVELS):
            records.append({
                "participant": int(person_id),
                "model": "Size-matched population",
                "repeat": repeat,
                "level": level,
                "rate": float(m_rates[i])
            })

    print(f"Completed participant {person_id}")

# Save detailed results
csv_path = RESULTS / "training_size_results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(records[0].keys())
    )
    writer.writeheader()
    writer.writerows(records)

print(f"\nSaved: {csv_path}")

# Calculate average detection rates
model_names = [
    "Personalized",
    "Size-matched population",
    "Full population"
]

means = {}

print("\n=== Average Detection Rates ===")

for model_name in model_names:
    means[model_name] = []

    for level in LEVELS:
        # First average across repeats for each person,
        # then average equally across participants.
        per_person = []

        for person_id in participant_ids:
            matching = [
                r["rate"] for r in records
                if r["participant"] == person_id
                and r["model"] == model_name
                and r["level"] == level
            ]

            if matching:
                per_person.append(np.mean(matching))

        means[model_name].append(
            np.mean(per_person)
        )

print(
    f"{'Change':>8s} | "
    f"{'Personal':>9s} | "
    f"{'Matched':>9s} | "
    f"{'Full':>9s}"
)
print("-" * 47)

for i, level in enumerate(LEVELS):
    print(
        f"{level:7d}% | "
        f"{means['Personalized'][i]:8.1f}% | "
        f"{means['Size-matched population'][i]:8.1f}% | "
        f"{means['Full population'][i]:8.1f}%"
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

plt.title(
    "GaitTrace: Does Training Data Quantity Matter?",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Simulated acceleration increase (%)")
plt.ylabel("Average windows flagged (%)")
plt.xticks(LEVELS)
plt.ylim(-5, 105)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plot_path = RESULTS / "training_size_comparison.png"
plt.savefig(plot_path, dpi=200)

print(f"\nSaved graph: {plot_path}")
plt.show()
