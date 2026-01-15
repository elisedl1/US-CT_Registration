import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

JSON_PATH = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra2/output_python_cma_group_allcases/ivd_diagnostics.json"
LAMBDA_IVD = 0.01
COLLISION_THRESH = 1.5  # mm

with open(JSON_PATH, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} CMA evaluations")

# COLLECT PER-PAIR METRICS
pair_stats = defaultdict(lambda: {
    "min_dist": [],
    "mean_dist": [],
    "std_dist": [],
    "n_collisions": [],
    "raw_loss": [],
    "weighted_loss": []
})

global_ivd_loss = []
mean_sim = []

for entry in data:
    mean_sim.append(entry["mean_sim"])
    global_ivd_loss.append(entry["ivd_loss"] * LAMBDA_IVD)

    for pair, metrics in entry["ivd_metrics"].items():
        pair_stats[pair]["min_dist"].append(metrics["current_min"])
        pair_stats[pair]["mean_dist"].append(metrics["current_mean"])
        pair_stats[pair]["std_dist"].append(metrics["current_std"])
        pair_stats[pair]["n_collisions"].append(metrics["n_collisions"])
        pair_stats[pair]["raw_loss"].append(metrics["total_loss"])
        pair_stats[pair]["weighted_loss"].append(
            LAMBDA_IVD * metrics["total_loss"]
        )

# NUMERICAL SUMMARY
print("\n=== IVD DIAGNOSTICS SUMMARY ===\n")

for pair, stats in pair_stats.items():
    min_d = np.array(stats["min_dist"])
    n_col = np.array(stats["n_collisions"])
    w_loss = np.array(stats["weighted_loss"])
    tot_loss = np.array(stats["raw_loss"])

    print(f"--- {pair} ---")
    print(f"  Min distance (mm): "
          f"mean={min_d.mean():.2f}, "
          f"min={min_d.min():.2f}")

    print(f"  Collision rate: "
          f"{np.mean(n_col > 0) * 100:.1f}% of evaluations")

    print(f"  Mean weighted loss: "
          f"{w_loss.mean():.4f}")

    if min_d.min() < COLLISION_THRESH:
        print("  ⚠️ COLLISIONS OCCUR")
    else:
        print("  ✅ No collisions")

    if w_loss.mean() < 1e-3:
        print("  ⚠️ IVD loss likely too weak")
    else:
        print("  ✅ IVD loss numerically active")

    print()

# PLOTS
x = np.arange(len(data))

plt.figure(figsize=(10, 5))
plt.plot(x, global_ivd_loss, label="λ · IVD loss")
plt.plot(x, mean_sim, label="Mean similarity")
plt.xlabel("CMA evaluation")
plt.ylabel("Value")
plt.title("IVD loss vs similarity (effective scale)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
