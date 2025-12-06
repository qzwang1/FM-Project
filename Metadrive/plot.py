import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


for col in ["reward", "collision", "coverage", "length"]:
    if col not in all_df.columns:
        raise KeyError(f"Column '{col}' not found in CSV files.")


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


plt.figure(figsize=(6, 4))
y = summary["reward_mean"].values
yerr = summary["reward_std"].values
x = np.arange(len(models))

plt.bar(x, y, yerr=yerr, capsize=5)
plt.xticks(x, models)
plt.ylabel("Mean Reward")
plt.title("Mean Episode Reward by Model")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "reward_by_model.png"), dpi=200)
# plt.show()


plt.figure(figsize=(6, 4))
y = summary["collision_rate"].values
x = np.arange(len(models))

plt.bar(x, y)
plt.xticks(x, models)
plt.ylabel("Collision Rate")
plt.title("Collision Rate by Model")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "collision_rate_by_model.png"), dpi=200)
# plt.show()


plt.figure(figsize=(6, 4))
y = summary["coverage_mean"].values
x = np.arange(len(models))

plt.bar(x, y)
plt.xticks(x, models)
plt.ylabel("Mean Coverage (lower is better)")
plt.title("Mean Coverage by Model")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "coverage_by_model.png"), dpi=200)
# plt.show()


plt.figure(figsize=(6, 4))
y = summary["length_mean"].values
x = np.arange(len(models))

plt.bar(x, y)
plt.xticks(x, models)
plt.ylabel("Mean Episode Length")
plt.title("Mean Episode Length by Model")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "length_by_model.png"), dpi=200)
# plt.show()
