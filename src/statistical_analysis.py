
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

df = pd.read_csv(RESULTS / "adaptive_results.csv")

# Put each participant's three AUROC scores side by side
wide = df.pivot_table(
    index=["participant", "change"],
    columns="model",
    values="auroc",
    aggfunc="mean"
).reset_index()

wide["fusion_vs_population"] = (
    wide["Adaptive fusion"] - wide["Population"]
)

wide["fusion_vs_personal"] = (
    wide["Adaptive fusion"] - wide["Personalized"]
)

rng = np.random.default_rng(42)

def bootstrap_ci(values, repetitions=10000):
    values = np.asarray(values, dtype=float)

    # Resample participants, not individual sensor windows
    samples = rng.choice(
        values,
        size=(repetitions, len(values)),
        replace=True
    )

    bootstrap_means = samples.mean(axis=1)

    return np.percentile(
        bootstrap_means, [2.5, 97.5]
    )

rows = []

print("\n=== GaitTrace Statistical Reliability ===")
print(
    f"{'Change':>8} | "
    f"{'Mean gain':>10} | "
    f"{'95% bootstrap CI':>23} | "
    f"{'Fusion wins':>12}"
)
print("-" * 65)

for level in sorted(wide["change"].unique()):
    group = wide[wide["change"] == level]

    differences = group[
        "fusion_vs_population"
    ].to_numpy()

    mean_gain = differences.mean()
    low, high = bootstrap_ci(differences)

    wins = np.sum(differences > 0)
    ties = np.sum(np.isclose(differences, 0))
    total = len(differences)

    rows.append({
        "change": level,
        "mean_gain": mean_gain,
        "ci_low": low,
        "ci_high": high,
        "fusion_wins": int(wins),
        "ties": int(ties),
        "participants": total
    })

    print(
        f"{level:7d}% | "
        f"{mean_gain:+.3f}      | "
        f"[{low:+.3f}, {high:+.3f}]"
        f"       | {wins}/{total}"
    )

summary = pd.DataFrame(rows)

output_csv = RESULTS / "statistical_summary.csv"
summary.to_csv(output_csv, index=False)

# Plot mean improvement with 95% bootstrap intervals
fig, ax = plt.subplots(figsize=(10, 6))

x = summary["change"].to_numpy()
means = summary["mean_gain"].to_numpy()

lower_errors = means - summary["ci_low"].to_numpy()
upper_errors = summary["ci_high"].to_numpy() - means

ax.errorbar(
    x,
    means,
    yerr=[lower_errors, upper_errors],
    fmt="o-",
    capsize=5,
    linewidth=2,
    markersize=7
)

ax.axhline(
    0,
    linestyle="--",
    color="gray",
    label="No improvement"
)

ax.set_title(
    "GaitTrace: Is Hybrid Fusion Consistently Better?",
    fontsize=14,
    fontweight="bold"
)
ax.set_xlabel("Simulated acceleration increase (%)")
ax.set_ylabel(
    "Mean AUROC gain (Hybrid - Population)"
)
ax.set_xticks(x)
ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()

output_plot = RESULTS / "statistical_reliability.png"
fig.savefig(output_plot, dpi=200)

print(f"\nSaved: {output_csv}")
print(f"Saved: {output_plot}")

plt.show()
