from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from features import extract_features

# Find the dataset
DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "UCI HAR Dataset"
)

# Load participant IDs and activity labels
activities = np.loadtxt(
    DATA_DIR / "train" / "y_train.txt",
    dtype=int
)

subjects = np.loadtxt(
    DATA_DIR / "train" / "subject_train.txt",
    dtype=int
)

# Select Participant 1's walking recordings
person_rows = np.where(
    (subjects == 1) & (activities == 1)
)[0]

# Load the six motion-sensor axes
signal_dir = DATA_DIR / "train" / "Inertial Signals"

signal_names = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
]

signals = np.stack(
    [
        np.loadtxt(
            signal_dir / f"{name}_train.txt"
        )[person_rows]
        for name in signal_names
    ],
    axis=1
)

# Convert raw signals into 24 features
features = extract_features(signals)

# Split the recordings in their original order.
# Leave a gap between training and testing to
# reduce overlap between nearby sensor windows.
n = len(features)

train_end = int(n * 0.6)
test_start = int(n * 0.8)

X_train = features[:train_end]
X_test = features[test_start:]

# Normalize features using training data only
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train an Isolation Forest anomaly detector
model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

model.fit(X_train_scaled)

# Higher scores indicate more unusual movement
anomaly_scores = -model.score_samples(X_test_scaled)

# -1 = flagged as unusual; 1 = typical
predictions = model.predict(X_test_scaled)

false_alarm_rate = np.mean(predictions == -1) * 100

print("\n=== GaitTrace Personalized Baseline ===")
print("Participant: 1")
print(f"Training windows: {len(X_train)}")
print(f"Testing windows: {len(X_test)}")
print(f"False alarm rate: {false_alarm_rate:.1f}%")
print(f"Average anomaly score: {np.mean(anomaly_scores):.3f}")
print("Anomaly scores:", np.round(anomaly_scores, 3))
