
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Load the results from our previous AUROC experiment
df = pd.read_csv(RESULTS / "auroc_results.csv")

# Average repeated samples for each participant and model
summary = (
    df.groupby(["participant", "change", "model"])["auroc"]
    .mean()
    .reset_index()
)

# Put the three models side by side
wide = summary.pivot(
    index=["participant", "change"],
    columns="model",
    values="auroc"
).reset_index()

# Positive difference means personalization performed better
wide["personal_advantage"] = (
    wide["Personalized"] - wide["Full population"]
)

wide["matched_advantage"] = (
    wide["Personalized"] - wide["Size-matched population"]
)

print("\n=== GaitTrace: Who Benefits from Personalization? ===")
print(f"Participants analyzed: {wide['participant'].nunique()}")

print("\n=== Personalized vs Full Population ===")
print(
    f"{'Change':>8s} | {'Personal wins':>13s} | "
    f"{'Mean difference':>15s}"
)
print("-" * 45)

for level in sorted(wide["change"].unique()):
    rows = wide[wide["change"] == level]

    wins = (rows["personal_advantage"] > 0).sum()
    total = len(rows)
    mean_diff = rows["personal_advantage"].mean()

    print(
        f"{level:7d}% | "
        f"{wins:2d}/{total:<10d} | "
        f"{mean_diff:+.3f}"
    )

# Detailed participant comparison at 20% change
selected_level = 20
selected = wide[
    wide["change"] == selected_level
].sort_values("personal_advantage")

print(f"\n=== Participant Results at {selected_level}% Change ===")
print(
    selected[
        [
            "participant",
            "Personalized",
            "Size-matched population",
            "Full population",
            "personal_advantage"
        ]
    ].round(3).to_string(index=False)
)

# Save participant-level data
output_csv = RESULTS / "participant_analysis.csv"
wide.to_csv(output_csv, index=False)

# Plot participant differences at 20% change
fig, ax = plt.subplots(figsize=(11, 7))

colors = [
    "steelblue" if x > 0 else "darkorange"
    for x in selected["personal_advantage"]
]

ax.barh(
    selected["participant"].astype(str),
    selected["personal_advantage"],
    color=colors
)

ax.axvline(0, color="black", linewidth=1)

ax.set_title(
    "GaitTrace: Who Benefits from Personalization?",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel(
    "AUROC difference (Personalized - Full Population)"
)
ax.set_ylabel("Participant ID")
ax.grid(axis="x", alpha=0.2)

fig.tight_layout()

output_plot = RESULTS / "participant_comparison.png"
fig.savefig(output_plot, dpi=200)

print(f"\nSaved: {output_csv}")
print(f"Saved: {output_plot}")

plt.show()