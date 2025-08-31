# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

# === Axis Styling Utilities ===

def _decorate_axis(ax: Axes, wrect=10, hrect=10, ticklabelsize='large'):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(length=0.1, width=0.1, labelsize=ticklabelsize)
    ax.spines['left'].set_position(('outward', hrect))
    ax.spines['bottom'].set_position(('outward', wrect))
    return ax

def annotate_and_decorate_axis(ax: Axes,
                                labelsize='x-large',
                                ticklabelsize='x-large',
                                xticks=None,
                                xticklabels=None,
                                yticks=None,
                                yticklabels=None,
                                legend=False,
                                grid_alpha=0.2,
                                legendsize='x-large',
                                xlabel='',
                                ylabel='',
                                wrect=10,
                                hrect=10):
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    if xticks is not None:
        ax.set_xticks(ticks=xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticks is not None:
        ax.set_yticks(yticks)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    ax.grid(True, alpha=grid_alpha)
    ax = _decorate_axis(ax, wrect=wrect, hrect=hrect, ticklabelsize=ticklabelsize)
    if legend:
        ax.legend(fontsize=legendsize)
    return ax

# === Data ===

resources = ['Food', 'Drink', 'Stone', 'Coal', 'Iron']
x = np.arange(len(resources))
width = 0.2

collection_1_agent = np.array([99.4, 89.43, 97.87, 89.56, 66.98])
collection_2_agents = np.array([95.92, 87.605, 97.87, 83.69, 60.83])
collection_4_agents = np.array([92.723, 87.45, 98.3, 79.474, 58.316])
collection_8_agents = np.array([82.648, 87.36, 98.6, 72.92, 48])

errors_1_agent = np.array([0.6, 1.8, 1.5, 2.1, 1.9])
errors_2_agents = np.array([0.7, 1.8, 1.4, 2.0, 2.0])
errors_4_agents = np.array([0.9, 1.7, 1.3, 3.0, 2.05])
errors_8_agents = np.array([1.4, 1.9, 1.2, 3.0, 2.6])

COLORS = {
    '1 Agent': '#4f0180',
    '2 Agents': '#c2185b',
    '4 Agents': '#f57c00',
    '8 Agents': '#fdd835',
    # '16 Agents': '#43a047'
}

# === Custom Bar Plot with Narrow Top/Bottom Error Lines ===

def draw_bars_with_narrow_error_caps(ax, x_pos, heights, errors, width, color, label, cap_fraction=0.4):
    cap_width = width * cap_fraction
    ax.bar(x_pos, heights, width, color=color, label=label)
    for x, y, err in zip(x_pos, heights, errors):
        # Top horizontal cap
        ax.plot([x - cap_width / 2, x + cap_width / 2], [y + err, y + err], color='black', linewidth=1)
        # Bottom horizontal cap
        ax.plot([x - cap_width / 2, x + cap_width / 2], [y - err, y - err], color='black', linewidth=1)
        # Vertical connecting line
        ax.vlines(x, y - err, y + err, color='black', linewidth=1)

# === Plotting ===

fig, ax = plt.subplots(figsize=(6, 4))

draw_bars_with_narrow_error_caps(ax, x - 1.5*width, collection_1_agent, errors_1_agent, width, COLORS['1 Agent'], '1 Agent')
draw_bars_with_narrow_error_caps(ax, x - 0.5*width, collection_2_agents, errors_2_agents, width, COLORS['2 Agents'], '2 Agents')
draw_bars_with_narrow_error_caps(ax, x + 0.5*width, collection_4_agents, errors_4_agents, width, COLORS['4 Agents'], '4 Agents')
draw_bars_with_narrow_error_caps(ax, x + 1.5*width, collection_8_agents, errors_8_agents, width, COLORS['8 Agents'], '8 Agents')

annotate_and_decorate_axis(
    ax,
    xlabel='Resource',
    ylabel='Collection Rate (%)',
    xticks=x,
    xticklabels=resources,
    legend=True,
    grid_alpha=0.3,
    labelsize='x-large',
    ticklabelsize='large',
    legendsize='large',
    wrect=5,
    hrect=5
)

ax.set_ylim(0, 100)

# Semi-transparent legend
legend = ax.get_legend()
legend.get_frame().set_alpha(1.0)

plt.tight_layout()
fig.savefig("ippo_collect_pct.pdf", format="pdf", bbox_inches="tight")
plt.show()

# %%
