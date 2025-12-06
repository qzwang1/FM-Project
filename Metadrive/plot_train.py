import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./verifai-bo/training_metrics_incremental.csv")

episodes = df["episode"]
rewards = df["reward"]
collisions = df["total_collision"]
coverages = df["total_coverage"]

# ---- Reward Curve ----
plt.figure(figsize=(6,4))
plt.plot(episodes, rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Reward Over Time")
plt.grid()
plt.savefig("reward_curve.png", dpi=200)
plt.close()

# ---- Collision Curve ----
plt.figure(figsize=(6,4))
plt.plot(episodes, collisions)
plt.xlabel("Episode")
plt.ylabel("Cumulative Collisions")
plt.title("Collision Trend During Training")
plt.grid()
plt.savefig("collision_curve.png", dpi=200)
plt.close()

# ---- Coverage Curve ----
plt.figure(figsize=(6,4))
plt.plot(episodes, coverages)
plt.xlabel("Episode")
plt.ylabel("Cumulative Coverage")
plt.title("Coverage Trend During Training")
plt.grid()
plt.savefig("coverage_curve.png", dpi=200)
plt.close()

print("All curves saved.")
