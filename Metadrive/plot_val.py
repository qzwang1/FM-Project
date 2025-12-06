import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Optional: 更好看的风格（建议）
plt.style.use("seaborn-v0_8-whitegrid")
# -------------------------------------------------------------------

RESULTS_DIR = "./eval_results"  

csv_files = glob.glob(os.path.join(RESULTS_DIR, "eval_*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No files matched eval_*.csv under {RESULTS_DIR}")

dfs = []
for path in csv_files:
    base = os.path.basename(path)           # eval_CE.csv
    name = os.path.splitext(base)[0]        # eval_CE
    model_name = name.split("eval_")[-1]    # CE

    df = pd.read_csv(path)
    if "model" not in df.columns:
        df["model"] = model_name
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

# 验证列
for col in ["reward", "collision", "coverage", "length"]:
    if col not in all_df.columns:
        raise KeyError(f"Column '{col}' not found in CSV files.")

# Summary
summary = (
    all_df.groupby("model")
    .agg(
        reward_mean=("reward", "mean"),
        reward_std=("reward", "std"),
        collision_rate=("collision", "mean"),
        coverage_mean=("coverage", "mean"),
        length_mean=("length", "mean"),
    )
    .reset_index()
)

print("Summary:")
print(summary)

models = summary["model"].tolist()
x = np.arange(len(models))
bar_width = 0.35  # 更细的柱子

# -------------------------------------------------------------------
# Plot 1: Reward
# -------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.bar(
    x,
    summary["reward_mean"].values,
    yerr=summary["reward_std"].values,
    width=bar_width,
    capsize=5,
    edgecolor="black",
)
plt.xticks(x, models)
plt.ylabel("Mean Reward")
plt.title("Mean Episode Reward by Model")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "reward_by_model.png"), dpi=200)

# -------------------------------------------------------------------
# Plot 2: Collision Rate
# -------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.bar(
    x,
    summary["collision_rate"].values,
    width=bar_width,
    edgecolor="black",
)
plt.xticks(x, models)
plt.ylabel("Collision Rate")
plt.title("Collision Rate by Model")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "collision_rate_by_model.png"), dpi=200)

# -------------------------------------------------------------------
# Plot 3: Coverage
# -------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.bar(
    x,
    summary["coverage_mean"].values,
    width=bar_width,
    edgecolor="black",
)
plt.xticks(x, models)
plt.ylabel("Mean Coverage (lower is better)")
plt.title("Mean Coverage by Model")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "coverage_by_model.png"), dpi=200)

# -------------------------------------------------------------------
# Plot 4: Episode Length
# -------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.bar(
    x,
    summary["length_mean"].values,
    width=bar_width,
    edgecolor="black",
)
plt.xticks(x, models)
plt.ylabel("Mean Episode Length")
plt.title("Mean Episode Length by Model")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "length_by_model.png"), dpi=200)

print("\nAll plots saved to", RESULTS_DIR)
