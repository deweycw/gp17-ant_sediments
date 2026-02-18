"""
Supplemental Figure: All 172 µ-XANES spectra stacked by group,
with group centroids highlighted.

Caption:
    All 172 Fe K-edge µ-XANES point spectra organized by spectral group.
    Colored lines show individual spectra and black lines show group
    centroids. Spectra are vertically offset for clarity. Groups were
    identified by k-means clustering (k=5) of PCA scores, with Group 3
    further subdivided into 3a and 3b by secondary k-means clustering
    based on differences in Fe sulfide speciation. The number of spectra
    in each group is indicated in parentheses.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os, glob

# ---------- Config ----------
DPI = 300
TARGET_WIDTH_MM = 180
TARGET_WIDTH_IN = TARGET_WIDTH_MM / 25.4

SPEC_DIR = "flattened-spectra"
REF_DIR = "FeK-standards/fluorescence/flattened"
PCA_DIR = "pca_results"

E_MIN, E_MAX, E_STEP = 7100, 7180, 0.2
E_GRID = np.arange(E_MIN, E_MAX + E_STEP / 2, E_STEP)

PLOT_E_MIN, PLOT_E_MAX = 7105, 7170

# Group colors and display order
GROUP_COLORS = {
    "1":  '#1f77b4',
    "2":  '#ff7f0e',
    "3a": '#2ca02c',
    "3b": '#82d682',
    "4":  '#d62728',
    "5":  '#9467bd',
}
GROUP_ORDER = ["1", "2", "3a", "3b", "4", "5"]
GROUP_LABELS = {
    "1": "Group 1", "2": "Group 2", "3a": "Group 3a",
    "3b": "Group 3b", "4": "Group 4", "5": "Group 5",
}

# Most frequently represented reference standards, ordered to align
# with sample groups: Fe(II) silicates → carbonate → Fe(III) → sulfides
REF_STANDARDS = [
    ("Biotite",            "Biotite",              '#1f77b4'),   # Grp 1  (n=88)
    ("Siderite-s",         "Siderite",             '#8c564b'),   # Grp 1, 4 (n=49)
    ("Augite",             "Augite",               '#2ca02c'),   # Grp 2, 3a (n=42)
    ("6L-Fhy",             "Ferrihydrite (6L)",    '#ff7f0e'),   # Grp 2 (n=19)
    ("Mackinawite (aged)", "Mackinawite (aged)",   '#9467bd'),   # Grp 3a-5 (n=64)
    ("Pyrrhotite",         "Pyrrhotite",           '#d62728'),   # Grp 5  (n=17)
]


# ---------- Helpers ----------
def load_csv_spectrum(filepath):
    data = np.loadtxt(filepath, comments='#', delimiter=',')
    return data[:, 0], data[:, 1]


def interp_to_grid(energy, mu, grid=E_GRID):
    return np.interp(grid, energy, mu)


# ---------- Load data ----------
print("Loading cluster assignments...")
clusters_df = pd.read_csv(os.path.join(PCA_DIR, "cluster_assignments.csv"))

# Sub-cluster cluster 3
c3_mask = clusters_df["cluster"] == 3
c3_df = clusters_df[c3_mask].copy()
pc_cols = [c for c in clusters_df.columns if c.startswith("PC")]
if len(c3_df) > 1:
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    c3_labels = km.fit_predict(c3_df[pc_cols].values)

    lcf_df = pd.read_csv(os.path.join(PCA_DIR, "lcf_individual.csv"))
    c3_lcf = lcf_df[lcf_df["cluster"] == 3].copy()
    c3_lcf["sub_label"] = c3_labels
    sub0_pyrr = c3_lcf[c3_lcf["sub_label"] == 0]["Pyrrhotite"].mean()
    sub1_pyrr = c3_lcf[c3_lcf["sub_label"] == 1]["Pyrrhotite"].mean()
    label_map = {0: "3a", 1: "3b"} if sub0_pyrr >= sub1_pyrr else {1: "3a", 0: "3b"}

    sub3_lookup = {}
    for i, (_, row) in enumerate(c3_df.iterrows()):
        sub3_lookup[row["spectrum"]] = label_map[c3_labels[i]]

# Build group assignment for all spectra
clusters_df["group"] = clusters_df["cluster"].astype(str)
for idx in clusters_df[c3_mask].index:
    spec = clusters_df.loc[idx, "spectrum"]
    clusters_df.loc[idx, "group"] = sub3_lookup.get(spec, "3a")

print(f"Group counts:\n{clusters_df['group'].value_counts().sort_index()}")

# ---------- Load all spectra ----------
print("Loading spectra...")
e_mask = (E_GRID >= PLOT_E_MIN) & (E_GRID <= PLOT_E_MAX)
e_plot = E_GRID[e_mask]

spectra_by_group = {g: [] for g in GROUP_ORDER}
for _, row in clusters_df.iterrows():
    spec_name = row["spectrum"]
    grp = row["group"]
    if grp not in spectra_by_group:
        continue

    spec_fp = os.path.join(SPEC_DIR, spec_name + ".csv")
    if not os.path.exists(spec_fp):
        matches = glob.glob(os.path.join(SPEC_DIR, spec_name + "*"))
        if matches:
            spec_fp = matches[0]
        else:
            continue

    e_data, mu_data = load_csv_spectrum(spec_fp)
    mu_grid = interp_to_grid(e_data, mu_data)
    spectra_by_group[grp].append(mu_grid[e_mask])

for g in GROUP_ORDER:
    print(f"  {GROUP_LABELS[g]}: {len(spectra_by_group[g])} spectra")

# ---------- Load reference standards ----------
print("Loading reference standards...")
ref_spectra = {}
for fname, display_name, color in REF_STANDARDS:
    fp = os.path.join(REF_DIR, fname + ".csv")
    if os.path.exists(fp):
        e, mu = load_csv_spectrum(fp)
        ref_spectra[display_name] = (interp_to_grid(e, mu)[e_mask], color)
        print(f"  {display_name}")
    else:
        print(f"  WARNING: {fp} not found")

# ---------- Build figure ----------
print("Building figure...")

fig, ax = plt.subplots(1, 1, figsize=(TARGET_WIDTH_IN * 0.42, 5.5), dpi=DPI)

offset = 0.0
OFFSET_STEP = 0.9
group_y_positions = {}

for grp in GROUP_ORDER:
    specs = spectra_by_group[grp]
    if not specs:
        continue
    color = GROUP_COLORS[grp]

    for mu in specs:
        ax.plot(e_plot, mu + offset, '-', color=color, lw=0.5, alpha=0.6)

    mean_spec = np.mean(specs, axis=0)
    ax.plot(e_plot, mean_spec + offset, 'k-', lw=1.5, alpha=0.9)

    group_y_positions[grp] = offset + np.max(mean_spec) * 0.5
    offset += OFFSET_STEP

# Group labels inside plot area
for grp in GROUP_ORDER:
    if grp in group_y_positions:
        n = len(spectra_by_group[grp])
        ax.text(PLOT_E_MAX - 5, group_y_positions[grp],
                f"{GROUP_LABELS[grp]} (n={n})", fontsize=6,
                va='center', ha='right', color=GROUP_COLORS[grp],
                fontweight='bold')

# Legend
ax.legend(
    handles=[
        plt.Line2D([0], [0], color='gray', lw=0.5, alpha=0.6, label='Individual spectrum'),
        plt.Line2D([0], [0], color='k', lw=1.5, label='Group centroid'),
    ],
    loc='lower right', fontsize=5.5, frameon=True, fancybox=False,
    edgecolor='gray', handlelength=1.5)

ax.set_xlabel("Energy (eV)", fontsize=8)
ax.set_ylabel("Flattened µ(E) + offset", fontsize=8)
ax.set_xlim(PLOT_E_MIN, PLOT_E_MAX)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=7, direction='in')
ax.set_yticks([])

fig.subplots_adjust(left=0.06, right=0.97, top=0.98, bottom=0.07)

# Save
OUT_DIR = "supplemental_figures"
os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(os.path.join(OUT_DIR, "figure_xanes_stacked.png"), dpi=DPI,
            bbox_inches='tight', facecolor='white', pad_inches=0.05)
fig.savefig(os.path.join(OUT_DIR, "figure_xanes_stacked.pdf"), dpi=DPI,
            bbox_inches='tight', facecolor='white', pad_inches=0.05)
plt.close(fig)
print("Saved: supplemental_figures/figure_xanes_stacked.png and .pdf")
