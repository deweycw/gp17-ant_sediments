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
df = pd.read_csv("Fe_EXAFS_params_merged.csv")
df = df[(df["R Angstrom"] > 0) & (df["Shell Label"] != "Fe-S")].copy()

# Exclude distant shells (> 4 Å) — these are 3rd/4th coordination shells
df = df[df["R Angstrom"] <= 4.0].copy()

# ── Load GP17-ANT bulk EXAFS fit parameters ─────────────────────────────────
gp17 = pd.read_csv("bulk/feffit_parameters.csv")
gp17_rows = []
for _, row in gp17.iterrows():
    base = {
        "Mineral Name": row["sample"],
        "Mineral Class": "This study",
    }
    # First shell: Fe-O
    gp17_rows.append({**base, "Shell Label": "Fe-O", "R Angstrom": row["R_O203"]})
    # Second shell: Fe-Fe
    gp17_rows.append({**base, "Shell Label": "Fe-Fe", "R Angstrom": row["R_Fe302"]})
    # Second shell: Fe-Al
    gp17_rows.append({**base, "Shell Label": "Fe-Al", "R Angstrom": row["R_Al328"]})
gp17_df = pd.DataFrame(gp17_rows)
df = pd.concat([df, gp17_df], ignore_index=True)

# ── Separate first shell (Fe-O) and second shell ────────────────────────────
# Cap first-shell Fe-O at orthopyroxene max (2.54 Å); longer Fe-O are 2nd/3rd
# coordination shells in phyllosilicates, not true first-shell distances
first_shell = df[(df["Shell Label"] == "Fe-O") & (df["R Angstrom"] <= 2.54)].copy()
second_shell = df[df["Shell Label"] != "Fe-O"].copy()

# First shell: pool all Fe-O distances per mineral class
first_shell["Subshell Label"] = "Fe–O"

# Remap Fe-Al/Mg → Fe-Al, then drop Fe-Mg
second_shell["Shell Label"] = second_shell["Shell Label"].replace("Fe-Al/Mg", "Fe-Al")
second_shell = second_shell[second_shell["Shell Label"] != "Fe-Mg"].copy()

# Second shell: group by scattering pair type (Fe-Fe, Fe-Si, Fe-Al), no sub-indexing
second_shell["Subshell Label"] = second_shell["Shell Label"].str.replace("-", "–")

# ── Mineral class ordering ───────────────────────────────────────────────────
class_order = ["Oxyhydroxide", "Silicate", "Phyllosilicate", "This study"]
class_colors = {
    "Oxyhydroxide": "#d62728",
    "Silicate": "#1f77b4",
    "Phyllosilicate": "#2ca02c",
    "This study": "#555555",
}

# ── Mineral markers (unique symbol per mineral) ─────────────────────────────
_all_markers = ["o", "s", "^", "v", "D", "p", "P", "*", "h", "d", "<", ">", "H", "8",
                 (4, 1, 0), (4, 1, 45), (5, 1, 0), (6, 1, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0)]
# Assign markers only for literature minerals; GP17-ANT samples all get "o"
_lit_names = sorted(df.loc[df["Mineral Class"] != "This study", "Mineral Name"].unique())
mineral_markers = {name: _all_markers[i % len(_all_markers)] for i, name in enumerate(_lit_names)}
mineral_display = {name: name for name in _lit_names}
mineral_display["orthopyroxene (M2 site)"] = "orthopyroxene"

# All GP17-ANT samples share one marker and one legend entry
for samp in gp17["sample"].unique():
    mineral_markers[samp] = "o"
    mineral_display[samp] = "GP17-ANT bulk"

# ── Plotting helper ──────────────────────────────────────────────────────────
def make_panel(ax, data, title, use_class_labels=False):
    """Create grouped box plots for one shell type."""
    class_abbrev = {
        "Oxyhydroxide": "Oxyhydr.",
        "Silicate": "Silicate",
        "Phyllosilicate": "Phyllosil.",
        "This study": "This study",
    }
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
                    "label": class_abbrev.get(cls, cls) if use_class_labels else label,
                    "values": subset["R Angstrom"].values,
                    "minerals": subset["Mineral Name"].values,
                    "oxstates": subset["Fe Oxidation State"].values
                    if "Fe Oxidation State" in subset.columns
                    else np.full(len(subset), ""),
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
            pos += 2.0  # wider gap between classes
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
    pad = (all_vals.max() - all_vals.min()) * 0.04
    ax.set_ylim(all_vals.min() - pad, all_vals.max() + pad)

    # Draw points (beeswarm for literature, box-and-whisker for This study)
    plotted_minerals = set()
    for i, g in enumerate(plot_groups):
        color = class_colors[g["class"]]

        if g["class"] == "This study":
            # Box-and-whisker for measured data
            bp = ax.boxplot(
                [g["values"]],
                positions=[positions[i]],
                widths=0.5,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="k", linewidth=1.5),
                whiskerprops=dict(color="k", linewidth=0.8),
                capprops=dict(color="k", linewidth=0.8),
            )
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.35)
            bp["boxes"][0].set_edgecolor("k")
            bp["boxes"][0].set_linewidth(0.8)
            # Add legend entry
            if "GP17-ANT bulk" not in plotted_minerals:
                ax.scatter([], [], color=color, marker="s", s=40,
                           edgecolors="k", linewidths=0.4,
                           label="GP17-ANT bulk")
                plotted_minerals.add("GP17-ANT bulk")
        else:
            # Range line (min to max)
            ax.vlines(positions[i], g["values"].min(), g["values"].max(),
                      colors=color, linewidths=1, alpha=0.4, zorder=3)

            # Beeswarm points with oxidation-state fill
            xs = beeswarm_x(g["values"], positions[i], point_size_pts=40)
            for x, val, mname, oxst in zip(xs, g["values"], g["minerals"], g["oxstates"]):
                marker = mineral_markers.get(mname, "o")
                display = mineral_display.get(mname, mname)
                oxst_str = str(oxst) if pd.notna(oxst) else ""
                if "mixed" in oxst_str.lower():
                    # Show both filled and open overlaid
                    # Filled (Fe(III)) slightly left, open (Fe(II)) slightly right
                    dx = 0.08
                    ax.scatter(
                        x - dx, val, color=color, marker=marker,
                        s=32, zorder=5, alpha=0.85,
                        edgecolors=color, linewidths=1.0,
                        label=display if display not in plotted_minerals else None,
                    )
                    ax.scatter(
                        x + dx, val, color="white", marker=marker,
                        s=32, zorder=5, alpha=0.85,
                        edgecolors=color, linewidths=1.0,
                    )
                elif "Fe(II)" in oxst_str and "Fe(III)" not in oxst_str:
                    ax.scatter(
                        x, val, color="white", marker=marker,
                        s=40, zorder=5, alpha=0.85,
                        edgecolors=color, linewidths=1.0,
                        label=display if display not in plotted_minerals else None,
                    )
                else:
                    ax.scatter(
                        x, val, color=color, marker=marker,
                        s=40, zorder=5, alpha=0.85,
                        edgecolors=color, linewidths=1.0,
                        label=display if display not in plotted_minerals else None,
                    )
                plotted_minerals.add(display)

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("R (Å)", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)


    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Tighten x limits
    ax.set_xlim(positions[0] - 0.8, positions[-1] + 0.8)


# ── Create figure ────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(190 / 25.4, 110 / 25.4), dpi=300,
    gridspec_kw={"width_ratios": [1, 1.3]},
)
fig.subplots_adjust(bottom=0.30, wspace=0.30, left=0.08, right=0.98, top=0.93)

make_panel(ax1, first_shell, "First Shell (Fe–O)", use_class_labels=True)
make_panel(ax2, second_shell, "Second Shell (Fe–M)")

# Panel labels
for ax, label in zip((ax1, ax2), ("a", "b")):
    ax.text(-0.16, 1.05, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")


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
    cls_rank = {"Oxyhydroxide": 0, "Silicate": 1, "Phyllosilicate": 2, "This study": 3}
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

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── Build table-style legend: one row per mineral class ──────────────────────
# Group handles by mineral class
cls_groups = defaultdict(list)
seen_display = set()
for a in (ax1, ax2):
    for h, l in zip(*a.get_legend_handles_labels()):
        if l in seen_display:
            continue
        seen_display.add(l)
        row = df[df["Mineral Name"] == l]
        if row.empty:
            orig = [k for k, v in mineral_display.items() if v == l]
            if orig:
                row = df[df["Mineral Name"] == orig[0]]
        if row.empty:
            cls_groups["This study"].append((h, l))
        else:
            cls_groups[row["Mineral Class"].iloc[0]].append((h, l))

# Sort minerals within each class
for cls in cls_groups:
    cls_groups[cls].sort(key=lambda x: x[1])

# Build legend manually using ax.text + scatter for full layout control
legend_ax = fig.add_axes([0.02, -0.06, 0.96, 0.20])
legend_ax.set_xlim(0, 1)
legend_ax.set_ylim(0, 1)
legend_ax.axis("off")

# Look up oxidation state for a mineral
def _get_oxstate(mname):
    row = df[df["Mineral Name"] == mname]
    if row.empty:
        orig = [k for k, v in mineral_display.items() if v == mname]
        if orig:
            row = df[df["Mineral Name"] == orig[0]]
    if row.empty:
        return ""
    return str(row["Fe Oxidation State"].iloc[0])

# Look up internal mineral name from display name
def _internal_name(display):
    for k, v in mineral_display.items():
        if v == display:
            return k
    return display

ncol = 6  # minerals per row under each header
row_height = 0.14
col_width = 0.95 / ncol
y = 0.96

# Oxidation state header + key
legend_ax.text(0.01, y, "Oxidation State", fontsize=7.5, fontweight="bold", color="k",
               va="center", transform=legend_ax.transAxes)
y -= row_height
ox_items = [
    ("0.5", "0.3", "Fe(III) — filled"),
    ("white", "0.3", "Fe(II) — open"),
]
for i, (fc, ec, lbl) in enumerate(ox_items):
    x = 0.03 + i * col_width
    legend_ax.scatter(x, y, marker="o", s=35, facecolors=fc, edgecolors=ec,
                      linewidths=1.0, transform=legend_ax.transAxes, clip_on=False)
    legend_ax.text(x + 0.015, y, lbl, fontsize=5.5, va="center",
                   transform=legend_ax.transAxes)

y -= row_height * 1.3

# Mineral classes with headers
for cls in class_order:
    if cls not in cls_groups:
        continue
    minerals = cls_groups[cls]
    color = class_colors[cls]

    # Header
    legend_ax.text(0.01, y, cls, fontsize=7.5, fontweight="bold", color=color,
                   va="center", transform=legend_ax.transAxes)
    y -= row_height

    # Minerals
    for j, (h, l) in enumerate(minerals):
        col = j % ncol
        if j > 0 and col == 0:
            y -= row_height
        x = 0.03 + col * col_width

        iname = _internal_name(l)
        mk = mineral_markers.get(iname, "o")
        oxst = _get_oxstate(l)

        if "mixed" in oxst.lower():
            # Paired filled + open
            legend_ax.scatter(x - 0.005, y, marker=mk, s=22, facecolors=color,
                              edgecolors=color, linewidths=0.8,
                              transform=legend_ax.transAxes, clip_on=False)
            legend_ax.scatter(x + 0.005, y, marker=mk, s=22, facecolors="white",
                              edgecolors=color, linewidths=0.8,
                              transform=legend_ax.transAxes, clip_on=False)
        elif "Fe(II)" in oxst and "Fe(III)" not in oxst:
            legend_ax.scatter(x, y, marker=mk, s=28, facecolors="white",
                              edgecolors=color, linewidths=0.8,
                              transform=legend_ax.transAxes, clip_on=False)
        else:
            legend_ax.scatter(x, y, marker=mk, s=28, facecolors=color,
                              edgecolors=color, linewidths=0.8,
                              transform=legend_ax.transAxes, clip_on=False)

        legend_ax.text(x + 0.015, y, l, fontsize=5.5, va="center",
                       transform=legend_ax.transAxes)

    y -= row_height * 1.2

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
