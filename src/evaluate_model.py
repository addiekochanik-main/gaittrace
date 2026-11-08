
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

# Load activity labels and participant IDs
activities = np.loadtxt(DATA_DIR / "train" / "y_train.txt", dtype=int)
subjects = np.loadtxt(DATA_DIR / "train" / "subject_train.txt", dtype=int)

rows = np.where((subjects == 1) & (activities == 1))[0]

signal_dir = DATA_DIR / "train" / "Inertial Signals"
names = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]

signals = np.stack(
    [np.loadtxt(signal_dir / f"{name}_train.txt")[rows] for name in names],
    axis=1
)

# Chronological split with a gap
n = len(signals)
train_end = int(n * 0.6)
test_start = int(n * 0.8)

train_signals = signals[:train_end]
test_signals = signals[test_start:]

# Extract features and fit the model
X_train = extract_features(train_signals)
X_test = extract_features(test_signals)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)
model.fit(X_train_scaled)

# Create controlled synthetic sensor changes
rng = np.random.default_rng(42)

# Test A: 50% stronger acceleration signals
amplitude_changed = test_signals.copy()
amplitude_changed[:, :3, :] *= 1.5

# Test B: Added random sensor noise
noisy = test_signals.copy()
noisy += rng.normal(0, 0.15, size=noisy.shape)

# Test C: 50% stronger gyroscope signals
rotation_changed = test_signals.copy()
rotation_changed[:, 3:, :] *= 1.5

def evaluate(label, data):
    X = scaler.transform(extract_features(data))
    flagged = model.predict(X) == -1
    scores = -model.score_samples(X)

    print(
        f"{label:25s} | "
        f"Flagged: {np.mean(flagged) * 100:5.1f}% | "
        f"Average anomaly score: {np.mean(scores):.3f}"
    )

print("\n=== GaitTrace Controlled Change Experiment ===")
evaluate("Original walking", test_signals)
evaluate("Stronger acceleration", amplitude_changed)
evaluate("Added sensor noise", noisy)
evaluate("Stronger rotation", rotation_changed)
