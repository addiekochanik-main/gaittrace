
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from features import extract_features

DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "UCI HAR Dataset"
)

activities = np.loadtxt(DATA_DIR / "train" / "y_train.txt", dtype=int)
subjects = np.loadtxt(DATA_DIR / "train" / "subject_train.txt", dtype=int)

signal_dir = DATA_DIR / "train" / "Inertial Signals"

names = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]

# Load all six sensor axes for all training-set windows
all_signals = np.stack(
    [np.loadtxt(signal_dir / f"{name}_train.txt") for name in names],
    axis=1
)

# Select walking recordings
walking = activities == 1
target = walking & (subjects == 1)
others = walking & (subjects != 1)

target_signals = all_signals[target]
other_signals = all_signals[others]

# Split target participant's windows chronologically,
# with a gap between training and testing
n = len(target_signals)
train_end = int(n * 0.6)
test_start = int(n * 0.8)

personal_train = target_signals[:train_end]
test_signals = target_signals[test_start:]

# Train both models
def train_model(signals):
    features = extract_features(signals)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )
    model.fit(scaled)

    return scaler, model

personal_scaler, personal_model = train_model(personal_train)
population_scaler, population_model = train_model(other_signals)

# Create identical synthetic changes for both models
rng = np.random.default_rng(42)

conditions = {
    "Original walking": test_signals,
    "Acceleration x1.5": test_signals.copy(),
    "Added noise": test_signals.copy(),
    "Rotation x1.5": test_signals.copy(),
}

conditions["Acceleration x1.5"][:, :3, :] *= 1.5
conditions["Added noise"][:] += rng.normal(
    0, 0.15, size=test_signals.shape
)
conditions["Rotation x1.5"][:, 3:, :] *= 1.5

def evaluate(scaler, model, signals):
    X = scaler.transform(extract_features(signals))
    flagged = model.predict(X) == -1
    scores = -model.score_samples(X)

    return np.mean(flagged) * 100, np.mean(scores)

print("\n=== GaitTrace: Personalized vs Population ===")
print(f"Personal training windows: {len(personal_train)}")
print(f"Population training windows: {len(other_signals)}")
print(f"Test windows: {len(test_signals)}\n")

print(f"{'Condition':22s} | {'Personal':>10s} | {'Population':>10s}")
print("-" * 50)

for name, signals in conditions.items():
    p_rate, p_score = evaluate(
        personal_scaler, personal_model, signals
    )
    g_rate, g_score = evaluate(
        population_scaler, population_model, signals
    )

    print(
        f"{name:22s} | "
        f"{p_rate:9.1f}% | "
        f"{g_rate:9.1f}%"
    )
