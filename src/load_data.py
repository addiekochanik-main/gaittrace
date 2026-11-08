
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Location of the UCI HAR dataset
DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "UCI HAR Dataset"
)

# Load activity labels and participant IDs
activities = np.loadtxt(
    DATA_DIR / "train" / "y_train.txt",
    dtype=int
)

subjects = np.loadtxt(
    DATA_DIR / "train" / "subject_train.txt",
    dtype=int
)

# Activity 1 means WALKING
walking_rows = np.where(activities == 1)[0]

# Choose one participant
person_id = subjects[walking_rows[0]]

person_rows = np.where(
    (subjects == person_id) & (activities == 1)
)[0]

# Load all six motion sensor axes
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
    axis=1,
)

print(f"Participant: {person_id}")
print(f"Walking windows: {len(person_rows)}")
print(f"Signal shape: {signals.shape}")

# Plot the acceleration data from one walking window
time = np.arange(128) / 50

plt.figure(figsize=(10, 5))

for axis, label in enumerate(["X", "Y", "Z"]):
    plt.plot(
        time,
        signals[0, axis],
        label=f"Acceleration {label}"
    )

plt.title(
    f"GaitTrace — Walking Signals (Participant {person_id})"
)
plt.xlabel("Time (seconds)")
plt.ylabel("Body acceleration (g)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


from features import extract_features

features = extract_features(signals)

print("Feature matrix shape:", features.shape)
print("First window features:")
print(np.round(features[0], 3))

