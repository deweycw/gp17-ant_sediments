"""
Box-and-whisker plots of Fe EXAFS interatomic distances from literature,
grouped by mineral class with sub-shells shown separately.

Sub-shells: within a single mineral, the same scattering path (e.g. Fe-O)
can appear at multiple distinct distances (split shells). These are indexed
as Fe-O₁ (short), Fe-O₂ (long), etc. Minerals with a single distance for
that path are labelled without a subscript.

Orthopyroxene M2 has 6 individual Fe-O bond lengths from XRD-constrained
EXAFS — these are collapsed to short (≤2.10 Å) and long (>2.10 Å) groups
to avoid dominating the silicate category.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict

# ── Load and clean data ──────────────────────────────────────────────────────
df = pd.read_csv("Fe_EXAFS_params_literature.csv")
df = df[(df["R Angstrom"] > 0) & (df["Shell Label"] != "Fe-S")].copy()

# ── Collapse orthopyroxene 6 crystallographic bonds into short/long means ────
opx_mask = df["Mineral Name"].str.startswith("orthopyroxene")
opx = df[opx_mask & (df["Shell Label"] == "Fe-O")].copy()
if not opx.empty:
    df = df[~(opx_mask & (df["Shell Label"] == "Fe-O"))]
    template = opx.iloc[0].copy()
    short_mean = opx.loc[opx["R Angstrom"] <= 2.10, "R Angstrom"].mean()
    long_mean = opx.loc[opx["R Angstrom"] > 2.10, "R Angstrom"].mean()
    new_rows = []
    for val in [short_mean, long_mean]:
        row = template.copy()
        row["R Angstrom"] = val
        row["Mineral Name"] = "orthopyroxene (M2)"
        new_rows.append(row)
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

# ── Separate first shell (Fe-O) and second shell (Fe-M) ─────────────────────
first_shell = df[df["Shell Label"] == "Fe-O"].copy()
second_shell = df[df["Shell Label"] != "Fe-O"].copy()

# First shell: pool all Fe-O distances per mineral class
first_shell["Subshell Label"] = "Fe–O"

# Second shell: group by scattering pair type (Fe-Fe, Fe-Si, Fe-Mg), no sub-indexing
second_shell["Subshell Label"] = second_shell["Shell Label"].str.replace("-", "–")

# ── Mineral class ordering ───────────────────────────────────────────────────
class_order = ["Oxyhydroxide", "Silicate", "Phyllosilicate"]
class_colors = {
    "Oxyhydroxide": "#d62728",
    "Silicate": "#1f77b4",
    "Phyllosilicate": "#2ca02c",
}

# ── Mineral markers (unique symbol per mineral) ─────────────────────────────
_all_markers = ["o", "s", "^", "v", "D", "p", "P", "*", "X", "h", "d", "<", ">", "H", "8"]
_mineral_names = sorted(df["Mineral Name"].unique())
mineral_markers = {name: _all_markers[i % len(_all_markers)] for i, name in enumerate(_mineral_names)}
mineral_display = {name: name for name in _mineral_names}

# ── Plotting helper ──────────────────────────────────────────────────────────
def make_panel(ax, data, title):
    """Create grouped box plots for one shell type."""
    plot_groups = []
    for cls in class_order:
        cls_data = data[data["Mineral Class"] == cls]
        if cls_data.empty:
            continue
        # Unique subshell labels sorted by median distance
        labels = cls_data.groupby("Subshell Label")["R Angstrom"].median().sort_values()
        for label in labels.index:
            subset = cls_data[cls_data["Subshell Label"] == label]
            if len(subset) > 0:
                plot_groups.append({
                    "class": cls,
                    "label": label,
                    "values": subset["R Angstrom"].values,
                    "minerals": subset["Mineral Name"].values,
                })

    if not plot_groups:
        ax.set_visible(False)
        return

    positions = []
    tick_labels = []
    class_spans = defaultdict(list)
    pos = 1
    prev_cls = None

    for g in plot_groups:
        if prev_cls is not None and g["class"] != prev_cls:
            pos += 1.5  # wider gap between classes
        positions.append(pos)
        tick_labels.append(g["label"])
        class_spans[g["class"]].append(pos)
        prev_cls = g["class"]
        pos += 1

    # Beeswarm-style layout: deterministic dodge to avoid overlap
    def beeswarm_x(values, center, point_size_pts=40, y_range=None):
        """Simple 1-D beeswarm: stack points that would overlap."""
        if y_range is None:
            y_range = ax.get_ylim()
        # Estimate point radius in data coords
        fig_height = fig.get_figheight() * fig.dpi
        ax_height = ax.get_position().height * fig_height
        r_data = (y_range[1] - y_range[0]) * (np.sqrt(point_size_pts) / ax_height) * 2.5
        r_x = 0.32  # horizontal dodge step

        order = np.argsort(values)
        x_out = np.full(len(values), center, dtype=float)
        placed_y = []
        placed_x = []
        for idx in order:
            y = values[idx]
            x = center
            # Check against already-placed points; dodge if overlapping
            for py, px in zip(placed_y, placed_x):
                if abs(y - py) < r_data:
                    # Overlaps vertically — try alternating left/right
                    for sign in [1, -1, 2, -2, 3, -3]:
                        candidate = center + sign * r_x * 0.5
                        ok = True
                        for py2, px2 in zip(placed_y, placed_x):
                            if abs(y - py2) < r_data and abs(candidate - px2) < r_x * 0.45:
                                ok = False
                                break
                        if ok:
                            x = candidate
                            break
            x_out[idx] = x
            placed_y.append(y)
            placed_x.append(x)
        return x_out

    # First pass: set axis limits so beeswarm spacing is computed correctly
    all_vals = np.concatenate([g["values"] for g in plot_groups])
    pad = (all_vals.max() - all_vals.min()) * 0.08
    ax.set_ylim(all_vals.min() - pad, all_vals.max() + pad)

    # Draw median lines and beeswarm points
    plotted_minerals = set()
    for i, g in enumerate(plot_groups):
        color = class_colors[g["class"]]
        # Range line (min to max)
        ax.vlines(positions[i], g["values"].min(), g["values"].max(),
                  colors=color, linewidths=1, alpha=0.4, zorder=3)

        # Beeswarm points
        xs = beeswarm_x(g["values"], positions[i], point_size_pts=40)
        for x, val, mname in zip(xs, g["values"], g["minerals"]):
            marker = mineral_markers.get(mname, "o")
            display = mineral_display.get(mname, mname)
            ax.scatter(
                x, val,
                color=color,
                marker=marker,
                s=40,
                zorder=5,
                alpha=0.85,
                edgecolors="k",
                linewidths=0.4,
                label=display if display not in plotted_minerals else None,
            )
            plotted_minerals.add(display)

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("R (Å)", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    # Class labels along bottom
    for cls in class_order:
        if cls in class_spans:
            span = class_spans[cls]
            mid = np.mean(span)
            ax.annotate(
                cls,
                xy=(mid, 0),
                xycoords=("data", "axes fraction"),
                xytext=(0, -45),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color=class_colors[cls],
            )

    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Create figure ────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(180 / 25.4, 110 / 25.4), dpi=300,
    gridspec_kw={"width_ratios": [1.1, 1]},
)
fig.subplots_adjust(bottom=0.28, wspace=0.40, left=0.09, right=0.97, top=0.88)

make_panel(ax1, first_shell, "First Shell (Fe–O)")
make_panel(ax2, second_shell, "Second Shell (Fe–M)")

fig.suptitle(
    "Fe EXAFS Interatomic Distances by Mineral Class",
    fontsize=12,
    fontweight="bold",
    y=0.97,
)

# Shared legend: collect unique handles from both axes, ordered by class then name
handles, labels = [], []
seen = set()
for ax in (ax1, ax2):
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in seen:
            handles.append(h)
            labels.append(l)
            seen.add(l)

# Sort legend entries by mineral class then name
def _sort_key(label):
    cls_rank = {"Oxyhydroxide": 0, "Silicate": 1, "Phyllosilicate": 2}
    # Try direct match first, then check display name mapping
    row = df[df["Mineral Name"] == label]
    if row.empty:
        # Find by display name
        orig = [k for k, v in mineral_display.items() if v == label]
        if orig:
            row = df[df["Mineral Name"] == orig[0]]
    if row.empty:
        return (3, label)
    return (cls_rank.get(row["Mineral Class"].iloc[0], 3), label)

order = sorted(range(len(labels)), key=lambda i: _sort_key(labels[i]))
handles = [handles[i] for i in order]
labels = [labels[i] for i in order]

fig.legend(
    handles, labels,
    loc="lower center",
    ncol=5,
    fontsize=6.5,
    frameon=True,
    framealpha=0.9,
    edgecolor="0.7",
    bbox_to_anchor=(0.5, -0.02),
    handletextpad=0.3,
    columnspacing=1.0,
)

fig.savefig("figure_exafs_distances.png", dpi=300, bbox_inches="tight")
fig.savefig("figure_exafs_distances.pdf", bbox_inches="tight")
print("Saved figure_exafs_distances.png and .pdf")

# ── Print summary table ─────────────────────────────────────────────────────
print("\n── First Shell (Fe-O) summary ──")
for cls in class_order:
    subset = first_shell[first_shell["Mineral Class"] == cls]
    if subset.empty:
        continue
    print(f"\n  {cls}:")
    for label in sorted(subset["Subshell Label"].unique()):
        vals = subset[subset["Subshell Label"] == label]["R Angstrom"]
        print(f"    {label}: n={len(vals)}, median={vals.median():.3f}, "
              f"range=[{vals.min():.3f}, {vals.max():.3f}]")

print("\n── Second Shell (Fe-M) summary ──")
for cls in class_order:
    subset = second_shell[second_shell["Mineral Class"] == cls]
    if subset.empty:
        continue
    print(f"\n  {cls}:")
    for label in sorted(subset["Subshell Label"].unique()):
        vals = subset[subset["Subshell Label"] == label]["R Angstrom"]
        print(f"    {label}: n={len(vals)}, median={vals.median():.3f}, "
              f"range=[{vals.min():.3f}, {vals.max():.3f}]")
