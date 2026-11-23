import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

out = Path("imatges")
out.mkdir(exist_ok=True)

models = [
    "Hybrid-\nTimeMixer v9",
    "AROME\nFrance HD",
    "AROME\nFrance",
    "GFS\n0,11°",
    "Baseline\ninterna",
]

# Comparació horària homogènia
tmax_mae  = [0.641, 0.672, 0.695, 0.815, 1.340]
tmax_rmse = [0.847, 0.848, 0.876, 1.018, 1.717]

tmin_mae  = [0.485, 0.933, 0.822, 0.853, 1.233]
tmin_rmse = [0.639, 1.100, 1.032, 1.098, 1.585]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#d1d5db",
    "axes.linewidth": 1.0,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "grid.color": "#e2e8f0",
    "grid.linewidth": 0.8,
})

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)

x = np.arange(len(models))
width = 0.36

panels = [
    (axes[0], "Tmax diària", tmax_mae, tmax_rmse),
    (axes[1], "Tmin diària", tmin_mae, tmin_rmse),
]

for ax, title, mae, rmse in panels:
    b1 = ax.bar(x - width/2, mae, width, label="MAE")
    b2 = ax.bar(x + width/2, rmse, width, label="RMSE")

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Error (°C)")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)

    ax.bar_label(b1, fmt="%.2f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.2f", fontsize=8, padding=2)

axes[0].legend(frameon=False, loc="upper left")

fig.suptitle(
    "Comparació horària homogènia de Tmax/Tmin: model local vs NWP operacional",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)

fig.tight_layout()
fig.savefig(out / "fig_resultats_tmax_tmin_mae_rmse_nwp.pdf", bbox_inches="tight")
fig.savefig(out / "fig_resultats_tmax_tmin_mae_rmse_nwp.png", dpi=300, bbox_inches="tight")
plt.close(fig)
