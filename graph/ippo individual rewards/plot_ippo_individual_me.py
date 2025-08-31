# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

# ────────────────────────────────────────────────────────────────────────────────
# Formatting helpers (from File B)
# ────────────────────────────────────────────────────────────────────────────────
def _decorate_axis(ax: Axes, wrect: int = 10, hrect: int = 10,
                   ticklabelsize: str = "large") -> Axes:
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params(length=0.1, width=0.1, labelsize=ticklabelsize)
    ax.spines["left"].set_position(("outward", hrect))
    ax.spines["bottom"].set_position(("outward", wrect))
    return ax

def annotate_and_decorate_axis(
    ax: Axes,
    labelsize: str = "x-large",
    ticklabelsize: str = "x-large",
    xticks=None,
    xticklabels=None,
    yticks=None,
    yticklabels=None,
    legend: bool = False,
    grid_alpha: float = 0.2,
    legendsize: str = "x-large",
    xlabel: str = "",
    ylabel: str = "",
    wrect: int = 10,
    hrect: int = 10,
) -> Axes:
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
    ax.grid(True, alpha=grid_alpha)
    ax = _decorate_axis(ax, wrect=wrect, hrect=hrect, ticklabelsize=ticklabelsize)
    if legend:
        ax.legend(fontsize=legendsize)
    return ax

def plot_mean_std_xy(X, mean, std, ax: Axes, label=None, marker=None, markevery=None, **kwargs):
    ax.plot(X, mean, label=label, marker=marker, markevery=markevery, **kwargs)
    fill_kwargs = {"alpha": 0.2}
    if "color" in kwargs:
        fill_kwargs["color"] = kwargs["color"]
    ax.fill_between(X, mean - std, mean + std, **fill_kwargs)

# ────────────────────────────────────────────────────────────────────────────────
# Data loading and processing
# ────────────────────────────────────────────────────────────────────────────────
FILE_PATH = "/app/Craftax/graph/ippo_individual.csv"
NORMALIZE = 266

AGENTS = [
    {"name": "1 Agent", "col": "IPPO - Basic - Individual Rewards - 1 Agents - individual_returns", "color": "#4f0180", "marker": None},
    {"name": "2 Agents", "col": "IPPO - Basic - Individual Rewards - 2 Agents - individual_returns", "color": "#c2185b", "marker": None},
    {"name": "4 Agents", "col": "IPPO - Basic - Individual Rewards - 4 Agents - individual_returns", "color": "#f57c00", "marker": None},
    {"name": "8 Agents", "col": "IPPO - Basic - Individual Rewards - 8 Agents - individual_returns", "color": "#fdd835", "marker": None},
]

data = pd.read_csv(FILE_PATH)
fig, ax = plt.subplots(figsize=(7, 4.5))
window = 100

# Calculate Timestep (M) and mask for <=1B steps
timesteps = data["env_step"]
timesteps_M = timesteps / 1e6
mask = timesteps <= 1e9  # 1B

# For marker placement at every 200M
xticks_M = np.arange(0, 1_000_000_001, 200_000_000) / 1e6  # [0,200,400,...,1000]

for agents in AGENTS:
    pct = (data[agents["col"]] / NORMALIZE) * 100
    smoothed_mean = pd.Series(pct).rolling(window=window, min_periods=1).mean().values
    smoothed_std = pd.Series(pct).rolling(window=window, min_periods=1).std().fillna(0).values
    # Only plot data up to 1B steps
    plot_mean_std_xy(
        timesteps_M[mask],
        smoothed_mean[mask],
        smoothed_std[mask],
        ax,
        label=agents["name"],
        color=agents["color"],
        linewidth=2.5,
        marker=agents["marker"],
        markevery=[np.abs(timesteps_M[mask] - xt).argmin() for xt in xticks_M],
    )

# Add Craftax dashed line at 15.3%
craftax_y = 15.3
ax.axhline(y=craftax_y, color="red", linestyle="--", linewidth=2, label="Craftax")

annotate_and_decorate_axis(
    ax,
    xlabel="Timestep (M)",
    ylabel="Reward (% of max)",
    legend=False,  # We handle the legend manually below
    grid_alpha=0.6,
    wrect=5,
    hrect=5,
    xticks=xticks_M,
    xticklabels=[f"{int(x)}" for x in xticks_M],
    yticks=np.arange(0, 21, 5),      # Y-axis ticks every 5%
    yticklabels=[f"{y}" for y in np.arange(0, 21, 5)],
)

# Place the legend in the lower right, including Craftax
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize="x-large", loc="lower right", frameon=True)

plt.xlim(-30, 1030)
plt.ylim(0, 17)
plt.tight_layout()
fig.savefig('ippo_individual_rewards.pdf', bbox_inches='tight')
plt.show()
# %%
