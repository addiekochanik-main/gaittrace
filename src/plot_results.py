
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

conditions = [
    "Original walking",
    "Stronger acceleration",
    "Added sensor noise",
    "Stronger rotation",
]

personalized = [5.3, 100.0, 100.0, 100.0]
population = [0.0, 94.7, 31.6, 52.6]

x = np.arange(len(conditions))
width = 0.36

fig, ax = plt.subplots(figsize=(11, 6))

bars1 = ax.bar(
    x - width / 2,
    personalized,
    width,
    label="Personalized model",
    color="#2563eb",
)

bars2 = ax.bar(
    x + width / 2,
    population,
    width,
    label="Population model",
    color="#f59e0b",
)

ax.set_ylabel("Recordings flagged as unusual (%)")
ax.set_title(
    "GaitTrace: Personalized vs. Population Anomaly Detection",
    fontsize=14,
    fontweight="bold",
)

ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylim(0, 115)
ax.legend()
ax.grid(axis="y", alpha=0.2)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 2,
            f"{height:.1f}%",
            ha="center",
            fontsize=10,
        )

fig.text(
    0.5,
    0.01,
    "UCI HAR walking data | 19 held-out windows | "
    "Synthetic sensor perturbations, not clinical gait abnormalities",
    ha="center",
    fontsize=9,
    color="gray",
)

plt.tight_layout(rect=[0, 0.04, 1, 1])

output_dir = Path(__file__).resolve().parent.parent / "results"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "model_comparison.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")

print(f"Saved chart to: {output_path}")

plt.show()
