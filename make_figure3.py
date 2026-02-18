"""
Figure 3: Bulk Fe K-edge XANES depth profiles by geochemical phase class.

Caption:
    Depth profiles of Fe phase proportions in bulk sediment determined by
    linear combination fitting (LCF) of Fe K-edge XANES spectra.
    Horizontal bars show the fractional contribution of each geochemical
    phase class at seven depths (1, 3, 5, 7, 9, 11, and 15 cm) below
    the sediment-water interface for (a) Station 15, (b) Station 27,
    and (c) Station 5. Individual mineral fractions from LCF are
    aggregated into three classes: Fe(II) silicate (hornblende),
    Fe(III) oxyhydroxide (6-line ferrihydrite), and Fe(III)
    phyllosilicate (ferrosmectite). Ferrosmectite dominates at all
    stations and depths (65-93%), with hornblende as the principal
    secondary phase, increasing with depth at Station 15. Ferrihydrite
    is detected only at the shallowest depths at Stations 5 and 27,
    consistent with reductive dissolution below the sediment-water
    interface. Fits used non-negative least-squares with up to 4
    reference phases selected from the set identified by PCA/target
    transformation analysis.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import string
from pathlib import Path

# ---------- Config ----------
DPI = 300
TARGET_WIDTH_MM = 180
TARGET_WIDTH_IN = TARGET_WIDTH_MM / 25.4

BULK_DIR = Path("bulk")
OUT_DIR = Path(".")

# Phase grouping
PHASE_GROUPS = {
    "Fe(III) oxyhydroxide":   ["6L-Fhy", "Goethite"],
    "Fe(III) phyllosilicate": ["Ferrosmectite"],
    "Fe(II) phyllosilicate":  ["Biotite"],
    "Fe(II) silicate":        ["Hornblende"],
    "Fe sulfide":             ["Mackinawite (aged)", "Pyrrhotite"],
    "Fe(II) carbonate":       ["Siderite-s"],
    "Fe-Ti oxide":            ["Ilmenite"],
    "Fe(II) phosphate":       ["Vivianite"],
}

# Plot order (top phases shown individually; rest lumped as "Other")
PLOT_ORDER = [
    "Fe(II) silicate",
    "Fe(III) oxyhydroxide",
    "Fe(III) phyllosilicate",
]

GROUP_COLORS = {
    "Fe(III) oxyhydroxide":   "#d62728",
    "Fe(III) phyllosilicate": "#7fbfff",
    "Fe(II) phyllosilicate":  "#aec7e8",
    "Fe(II) silicate":        "#1f77b4",
    "Fe sulfide":             "#2ca02c",
    "Fe(II) carbonate":       "#ff7f0e",
    "Fe-Ti oxide":            "#9467bd",
    "Fe(II) phosphate":       "#8c564b",
}

STATION_LABELS = {
    "GP17_station5_19119":  "Station 5",
    "GP17_station15_19908": "Station 15",
    "GP17_station27_20892": "Station 27",
}

# ---------- Load data ----------
print("Loading bulk LCF results...")
df = pd.read_csv(BULK_DIR / "bulk_lcf_mineral_refs.csv")

# Aggregate into phase classes
meta_cols = ["filename", "station", "depth_cm", "r_factor", "chi_sq",
             "weight_sum", "n_refs", "components"]
df_grouped = df[[c for c in meta_cols if c in df.columns]].copy()
for group_name, members in PHASE_GROUPS.items():
    cols = [c for c in members if c in df.columns]
    df_grouped[group_name] = df[cols].sum(axis=1)

phase_cols = [g for g in PHASE_GROUPS.keys() if g in df_grouped.columns]
top_phases = [p for p in PLOT_ORDER if p in phase_cols]
other_phases = [c for c in phase_cols if c not in top_phases]

# Phase markers for scatter plot
PHASE_MARKERS = {
    "Fe(II) silicate":        "s",
    "Fe(III) oxyhydroxide":   "D",
    "Fe(III) phyllosilicate": "o",
}

# ---------- Build figure ----------
print("Building figure...")
stations = sorted(df_grouped["station"].unique())
n_stations = len(stations)
panel_labels = list(string.ascii_lowercase)

fig, axes = plt.subplots(1, n_stations, figsize=(TARGET_WIDTH_IN, 3.2),
                         sharey=True, dpi=DPI)
if n_stations == 1:
    axes = [axes]

for i, (ax, station) in enumerate(zip(axes, stations)):
    sdf = df_grouped[df_grouped["station"] == station].sort_values("depth_cm")
    depths = sdf["depth_cm"].values

    for col in top_phases:
        values = sdf[col].values * 100
        color = GROUP_COLORS.get(col, "gray")
        marker = PHASE_MARKERS.get(col, "o")
        ax.plot(values, depths, linestyle=':', color=color, linewidth=0.8)
        ax.scatter(values, depths, marker=marker, color=color,
                   s=25, zorder=3, label=col if i == 0 else None,
                   edgecolors="white", linewidths=0.3)

    # Other
    other_values = sdf[other_phases].sum(axis=1).values * 100
    if other_values.sum() > 0:
        ax.plot(other_values, depths, linestyle=':', color="gray", linewidth=0.8)
        ax.scatter(other_values, depths, marker="^", color="lightgray",
                   s=25, zorder=3, label="Other" if i == 0 else None,
                   edgecolors="gray", linewidths=0.3)

    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=7, direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Phase fraction (%)", fontsize=8)
    label = STATION_LABELS.get(station, station)
    ax.set_title(f"({panel_labels[i]})  {label}", fontsize=8, loc='left')

axes[0].set_ylabel("Depth (cm)", fontsize=8)

# Shared legend below
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center',
           ncol=len(handles), fontsize=6.5,
           bbox_to_anchor=(0.5, -0.01), frameon=False,
           handlelength=1.2)

fig.subplots_adjust(wspace=0.08, bottom=0.15)

# Save
for ext in ("png", "pdf"):
    outpath = OUT_DIR / f"figure_bulk_depth_profiles.{ext}"
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    print(f"Saved: {outpath}")

plt.close(fig)
