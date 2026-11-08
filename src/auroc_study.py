
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
REPEATS = 5

def fit_model(signals):
    X = extract_features(signals)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_scaled)

    return scaler, model

def anomaly_scores(scaler, model, signals):
    X = scaler.transform(extract_features(signals))
    return -model.score_samples(X)

def calculate_auroc(scaler, model, test, level):
    # Original walking recordings: label 0
    original_scores = anomaly_scores(
        scaler, model, test
    )

    # Artificially altered recordings: label 1
    changed = test.copy()
    changed[:, :3, :] *= 1 + level / 100

    changed_scores = anomaly_scores(
        scaler, model, changed
    )

    labels = np.concatenate([
        np.zeros(len(original_scores)),
        np.ones(len(changed_scores))
    ])

    scores = np.concatenate([
        original_scores,
        changed_scores
    ])

    return roc_auc_score(labels, scores)

records = []

print("\n=== GaitTrace AUROC Study ===")

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
    test_start = int(n * 0.8)

    personal_train = person[:train_end]
    test = person[test_start:]

    # Personalized model
    p_scaler, p_model = fit_model(personal_train)

    # Full population model
    f_scaler, f_model = fit_model(others)

    models = [
        ("Personalized", 0, p_scaler, p_model),
        ("Full population", 0, f_scaler, f_model)
    ]

    # Five size-matched population samples
    for repeat in range(REPEATS):
        rng = np.random.default_rng(
            int(person_id) * 100 + repeat
        )

        selected = rng.choice(
            len(others),
            size=len(personal_train),
            replace=False
        )

        m_scaler, m_model = fit_model(
            others[selected]
        )

        models.append((
            "Size-matched population",
            repeat,
            m_scaler,
            m_model
        ))

    for model_name, repeat, scaler, model in models:
        for level in LEVELS:
            auc = calculate_auroc(
                scaler, model, test, level
            )

            records.append({
                "participant": int(person_id),
                "model": model_name,
                "repeat": repeat,
                "change": level,
                "auroc": float(auc)
            })

    print(f"Completed participant {person_id}")

# Save all individual results
csv_path = RESULTS / "auroc_results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(records[0].keys())
    )
    writer.writeheader()
    writer.writerows(records)

print(f"\nSaved results: {csv_path}")

# Average across repeats first, then participants
model_names = [
    "Personalized",
    "Size-matched population",
    "Full population"
]

means = {}

for model_name in model_names:
    means[model_name] = []

    for level in LEVELS:
        participant_aucs = []

        for person_id in participant_ids:
            values = [
                r["auroc"]
                for r in records
                if r["participant"] == person_id
                and r["model"] == model_name
                and r["change"] == level
            ]

            if values:
                participant_aucs.append(
                    np.mean(values)
                )

        means[model_name].append(
            np.mean(participant_aucs)
        )

print("\n=== Average AUROC Across Participants ===")
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
        f"{means['Personalized'][i]:8.3f} | "
        f"{means['Size-matched population'][i]:8.3f} | "
        f"{means['Full population'][i]:8.3f}"
    )

# Plot AUROC sensitivity
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
    0.5,
    linestyle="--",
    color="gray",
    label="Chance-level ranking"
)

plt.title(
    "GaitTrace: AUROC Across Simulated Movement Changes",
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

plot_path = RESULTS / "auroc_comparison.png"
plt.savefig(plot_path, dpi=200)

print(f"\nSaved graph: {plot_path}")
plt.show()
