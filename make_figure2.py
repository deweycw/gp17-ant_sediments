"""
Figure 2: Representative Fe K-edge µ-XANES spectra with LCF fits.
6-panel figure (2 columns x 3 rows) illustrating the range of Fe speciation.
LCF uses pre-edge-weighted NNLS (×2, 7108–7118 eV) with 5% minimum component threshold.

Caption:
    Representative Fe K-edge µ-XANES spectra with linear combination fitting
    (LCF) results showing the range of Fe speciation observed across 172
    microprobe point analyses. Black solid lines show measured spectra, red
    dashed lines show LCF fits, colored lines show individual reference mineral
    contributions scaled by their fitted weights (with phase group and
    percentage labeled), and gray lines show fit residuals offset below. LCF
    used non-negative least squares with pre-edge weighting (×2, 7108–7118 eV)
    to improve sensitivity to Fe oxidation state, and a 5% minimum component
    threshold. Spectra were selected to illustrate the dominant Fe phases:
    (a) Fe(II) phyllosilicate + carbonate + oxyhydroxide;
    (b) Fe(III) phyllosilicate with Fe(II) silicate;
    (c) Fe(II) phyllosilicate with Fe sulfide and carbonate;
    (d) green rust with Fe(II) phosphate;
    (e) Fe(II) carbonate with Fe sulfide;
    (f) Fe sulfide with Fe(II) phosphate.
    Spectral groups (1–5, with group 3 subdivided into 3a and 3b) were
    identified by k-means clustering of PCA scores; measurement locations are
    indicated in Figure 1.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from itertools import combinations
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

# Pre-edge weighting (must match 03_lcf_microprobe.ipynb)
PRE_EDGE_RANGE = (7108, 7118)  # eV
PRE_EDGE_WEIGHT = 2.0

# Plot range
PLOT_E_MIN, PLOT_E_MAX = 7105, 7170

# Map panel spectrum prefixes (spectra visible on the 4 maps in Figure 1)
MAP_PREFIXES = [
    "FeXANES_GT5_flaky2_FeXRD_",
    "FeXANES_GT5_flaky2_Fe_",
    "FeXANES_GT5_flakysmooth2_Fe_",
    "FeXANES_GT15_rectangles_Fe_",
    "FeXANES_GT15_FeTiXRD_striated2_",
    "FeXANES_GT15_Fe_striated2_",
]

# Fixed spectra matching Figure 1 selections (group -> spectrum name)
SELECTED_SPECTRA = {
    1:    "FeXANES_GT15_rectangles_Fe_6.001",
    2:    "FeXANES_GT5_flaky2_FeXRD_12.001",
    "3a": "FeXANES_GT5_flakysmooth2_Fe_11.001",
    "3b": "FeXANES_GT5_flaky2_Fe_20.001",
    4:    "FeXANES_GT15_FeTiXRD_striated2_2.001",
    5:    "FeXANES_GT5_flakysmooth2_Fe_19.001",
}

# Reference standard files
REF_NAMES = [
    "2L-Fhy on sand", "2L-Fhy", "6L-Fhy", "Augite", "Biotite",
    "FeS", "Ferrosmectite", "Goethite on sand", "Goethite",
    "Green Rust - Carbonate", "Green Rust - Chloride",
    "Green Rust - Sulfate", "Hematite on sand", "Hematite",
    "Hornblende", "Ilmenite", "Jarosite", "Lepidocrocite",
    "Mackinawite (aged)", "Mackinawite", "Maghemite", "Nontronite",
    "Pyrite", "Pyrrhotite", "Schwertmannite", "Siderite-n",
    "Siderite-s", "Vivianite",
]

# Phase grouping: mineral name -> phase group label
PHASE_GROUPS = {
    "6L-Fhy":             "Fe(III) oxyhydroxide",
    "2L-Fhy":             "Fe(III) oxyhydroxide",
    "2L-Fhy on sand":     "Fe(III) oxyhydroxide",
    "Goethite":           "Fe(III) oxyhydroxide",
    "Goethite on sand":   "Fe(III) oxyhydroxide",
    "Lepidocrocite":      "Fe(III) oxyhydroxide",
    "Schwertmannite":     "Fe(III) oxyhydroxide",
    "Hematite":           "Fe(III) oxide",
    "Hematite on sand":   "Fe(III) oxide",
    "Maghemite":          "Fe(III) oxide",
    "Ferrosmectite":      "Fe(III) phyllosilicate",
    "Nontronite":         "Fe(III) phyllosilicate",
    "Biotite":            "Fe(II) phyllosilicate",
    "Hornblende":         "Fe(II) silicate",
    "Augite":             "Fe(II) silicate",
    "Mackinawite (aged)": "Fe sulfide",
    "Mackinawite":        "Fe sulfide",
    "Pyrrhotite":         "Fe sulfide",
    "Pyrite":             "Fe sulfide",
    "FeS":                "Fe sulfide",
    "Siderite-s":         "Fe(II) carbonate",
    "Siderite-n":         "Fe(II) carbonate",
    "Ilmenite":           "Fe-Ti oxide",
    "Vivianite":          "Fe(II) phosphate",
    "Jarosite":           "Fe(III) sulfate",
    "Green Rust - Carbonate": "Green rust",
    "Green Rust - Chloride":  "Green rust",
    "Green Rust - Sulfate":   "Green rust",
}

def phase_label(ref_name):
    """Map a reference mineral name to its phase group label."""
    return PHASE_GROUPS.get(ref_name, ref_name)

# Colors for phase groups (consistent across panels)
PHASE_COLORS = {
    "Fe(III) oxyhydroxide":   '#ff7f0e',  # orange
    "Fe(III) oxide":          '#d62728',   # dark red (avoid for fit line)
    "Fe(III) phyllosilicate": '#2ca02c',   # green
    "Fe(II) phyllosilicate":  '#1f77b4',   # blue
    "Fe(II) silicate":        '#17becf',   # cyan
    "Fe sulfide":             '#9467bd',   # purple
    "Fe(II) carbonate":       '#8c564b',   # brown
    "Fe-Ti oxide":            '#e377c2',   # pink
    "Fe(II) phosphate":       '#bcbd22',   # olive
    "Fe(III) sulfate":        '#7f7f7f',   # gray
    "Green rust":             '#aec7e8',   # light blue
}

# Fallback colors
COMP_COLORS = [
    '#1f77b4', '#2ca02c', '#9467bd', '#8c564b', '#e377c2',
    '#7f7f7f', '#bcbd22', '#17becf', '#ff7f0e', '#aec7e8',
]

# Map spectrum-name prefix → Figure 1 panel label
SPEC_TO_PANEL = [
    ("FeXANES_GT5_flaky2_FeXRD_",       "(a)"),
    ("FeXANES_GT5_flaky2_Fe_",          "(a)"),
    ("FeXANES_GT5_flakysmooth2_Fe_",    "(b)"),
    ("FeXANES_GT15_rectangles_Fe_",     "(c)"),
    ("FeXANES_GT15_FeTiXRD_striated2_", "(d)"),
    ("FeXANES_GT15_Fe_striated2_",      "(d)"),
]


def map_panel_label(spectrum_name):
    """Return 'Fig. 1X, pt. N' string for a spectrum from the Figure 1 maps."""
    for prefix, panel in SPEC_TO_PANEL:
        if spectrum_name.startswith(prefix):
            # Extract point number: everything after the prefix, before .001
            rest = spectrum_name[len(prefix):]
            pt = rest.replace(".001", "").strip("_. ")
            if not pt:
                pt = "1"
            return f"Fig. 1{panel[1]}, pt. {pt}"
    return ""


# ---------- Helpers ----------
def load_csv_spectrum(filepath):
    """Load a two-column CSV spectrum with # comments."""
    data = np.loadtxt(filepath, comments='#', delimiter=',')
    return data[:, 0], data[:, 1]


def interp_to_grid(energy, mu, grid=E_GRID):
    """Interpolate spectrum onto common energy grid."""
    return np.interp(grid, energy, mu)


def is_map_spectrum(name):
    """Check if spectrum name comes from one of the 4 Figure 1 maps."""
    for prefix in MAP_PREFIXES:
        if name.startswith(prefix):
            return True
    return False


def short_name(spectrum_name):
    """Create a short display name from the full spectrum filename."""
    s = spectrum_name.replace("FeXANES_", "").replace(".001", "")
    s = s.replace("GT15_FeTiXRD_", "").replace("GT15_", "").replace("GT5_", "")
    s = s.replace("Fe_", "").replace("_Fe", "")
    return s


# ---------- Load data ----------
print("Loading cluster assignments and LCF results...")
clusters_df = pd.read_csv(os.path.join(PCA_DIR, "cluster_assignments.csv"))
lcf_df = pd.read_csv(os.path.join(PCA_DIR, "lcf_individual.csv"))

# Merge
df = clusters_df.merge(lcf_df, on=["spectrum", "cluster"], how="inner")

# Sub-cluster cluster 3 into 3a and 3b using k-means on PC scores
c3_mask = df["cluster"] == 3
c3_df = df[c3_mask].copy()
pc_cols = [c for c in df.columns if c.startswith("PC")]
if len(c3_df) > 1:
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    c3_labels = km.fit_predict(c3_df[pc_cols].values)
    c3_df["sub_cluster"] = c3_labels

    # Determine which sub-cluster is 3a (higher Pyrrhotite) vs 3b (higher Mackinawite aged)
    sub0_pyrrhotite = c3_df[c3_df["sub_cluster"] == 0]["Pyrrhotite"].mean()
    sub1_pyrrhotite = c3_df[c3_df["sub_cluster"] == 1]["Pyrrhotite"].mean()
    if sub0_pyrrhotite >= sub1_pyrrhotite:
        label_map = {0: "3a", 1: "3b"}
    else:
        label_map = {1: "3a", 0: "3b"}
    c3_df["group"] = c3_df["sub_cluster"].map(label_map)
else:
    c3_df["group"] = "3a"

# Build full group column
df["group"] = df["cluster"].astype(str)
df.loc[c3_mask, "group"] = c3_df["group"].values

print(f"Group counts:\n{df['group'].value_counts().sort_index()}")

# ---------- Use fixed spectra from Figure 1 ----------
print("\nUsing selected spectra from Figure 1...")
ref_columns = [c for c in lcf_df.columns if c not in
               ["spectrum", "cluster", "r_factor", "r_factor_w", "chi_sq", "weight_sum", "n_refs"]]

selected = {}
for grp in [1, 2, "3a", "3b", 4, 5]:
    spec_name = SELECTED_SPECTRA[grp]
    row = df[df["spectrum"] == spec_name]
    if row.empty:
        print(f"  Group {grp}: {spec_name} NOT FOUND in data")
        continue
    row = row.iloc[0]
    selected[grp] = row
    print(f"  Group {grp}: {row['spectrum']} (R={row['r_factor']:.4f}, n_refs={int(row['n_refs'])})")

# ---------- Load reference standards ----------
print("\nLoading reference standards...")
ref_spectra = {}
for name in REF_NAMES:
    fp = os.path.join(REF_DIR, name + ".csv")
    if os.path.exists(fp):
        e, mu = load_csv_spectrum(fp)
        ref_spectra[name] = interp_to_grid(e, mu)
    else:
        print(f"  WARNING: {fp} not found")

# ---------- Build Figure 2 ----------
print("\nBuilding Figure 2...")

fig, axes = plt.subplots(3, 2, figsize=(TARGET_WIDTH_IN, 8.5), dpi=DPI)
fig.subplots_adjust(hspace=0.15, wspace=0.25, left=0.08, right=0.97, top=0.98, bottom=0.06)

group_order = [1, 2, "3a", "3b", 4, 5]
panel_labels_list = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

for idx, grp in enumerate(group_order):
    row_idx = idx // 2
    col_idx = idx % 2
    ax = axes[row_idx, col_idx]

    info = selected[grp]
    spec_name = info["spectrum"]

    # Load the measured spectrum
    spec_fp = os.path.join(SPEC_DIR, spec_name + ".csv")
    if not os.path.exists(spec_fp):
        # Try without the .001 part
        alt = spec_name.replace(".001", "") + ".001.csv"
        spec_fp = os.path.join(SPEC_DIR, alt)
    if not os.path.exists(spec_fp):
        # Search
        matches = glob.glob(os.path.join(SPEC_DIR, spec_name + "*"))
        if matches:
            spec_fp = matches[0]

    e_data, mu_data = load_csv_spectrum(spec_fp)
    mu_grid = interp_to_grid(e_data, mu_data)

    # Full combinatorial weighted NNLS (matches 03_lcf_microprobe.ipynb)
    b = mu_grid
    all_ref_names = [rn for rn in REF_NAMES if rn in ref_spectra]
    ref_mat = np.column_stack([ref_spectra[rn] for rn in all_ref_names])

    # Build pre-edge weight vector
    w_vec = np.ones(len(E_GRID))
    pre_mask = (E_GRID >= PRE_EDGE_RANGE[0]) & (E_GRID <= PRE_EDGE_RANGE[1])
    w_vec[pre_mask] = PRE_EDGE_WEIGHT
    b_w = b * w_vec

    MAX_REFS = 3
    MIN_COMPONENT_FRAC = 0.05  # drop components below 5%
    best = {"r_factor_w": np.inf}
    for n_ref in range(1, MAX_REFS + 1):
        for combo in combinations(range(len(all_ref_names)), n_ref):
            A = ref_mat[:, list(combo)]
            A_w = A * w_vec[:, np.newaxis]
            weights, _ = nnls(A_w, b_w)
            w_sum = weights.sum()
            if w_sum == 0:
                continue
            # Skip if any component is below the minimum fraction
            if n_ref > 1 and np.any((weights / w_sum) < MIN_COMPONENT_FRAC):
                continue
            fitted = A @ weights
            residual_tmp = b - fitted
            residual_w = residual_tmp * w_vec
            r_factor_w = np.sum(np.abs(residual_w)) / np.sum(np.abs(b_w))
            if r_factor_w < best["r_factor_w"]:
                best = {
                    "combo": combo,
                    "weights": weights,
                    "fitted": fitted,
                    "r_factor_w": r_factor_w,
                }

    fit = best["fitted"]
    raw_weights = best["weights"]
    combo_names = [all_ref_names[i] for i in best["combo"]]
    weight_sum = raw_weights.sum()
    components = []
    for rn, rw in zip(combo_names, raw_weights):
        if rw > 1e-6:
            comp = rw * ref_spectra[rn]
            pct = rw / weight_sum * 100 if weight_sum > 0 else 0
            components.append((rn, rw, pct, comp))
    components.sort(key=lambda x: x[1], reverse=True)

    residual = mu_grid - fit

    # Unweighted R-factor (for display)
    r_factor = np.sum(np.abs(residual)) / np.sum(np.abs(mu_grid))

    # Plot range mask
    e_mask = (E_GRID >= PLOT_E_MIN) & (E_GRID <= PLOT_E_MAX)
    e_plot = E_GRID[e_mask]

    # Compute residual offset (below data)
    data_min = mu_grid[e_mask].min()
    resid_max = np.abs(residual[e_mask]).max()
    resid_offset = data_min - resid_max - 0.08

    # Plot data
    ax.plot(e_plot, mu_grid[e_mask], 'k-', lw=1.2, label='Data', zorder=5)
    # Plot fit
    ax.plot(e_plot, fit[e_mask], 'r--', lw=1.0, label='LCF fit', zorder=4)

    # Plot individual components using phase-group colors
    for ci, (ref_name, w, pct, comp) in enumerate(components):
        pg = phase_label(ref_name)
        color = PHASE_COLORS.get(pg, COMP_COLORS[ci % len(COMP_COLORS)])
        ax.plot(e_plot, comp[e_mask], '-', color=color, lw=0.7, alpha=0.8, zorder=3)

    # Plot residual
    ax.plot(e_plot, residual[e_mask] + resid_offset, '-', color='gray', lw=0.7, zorder=2)
    ax.axhline(y=resid_offset, color='gray', lw=0.3, ls=':', zorder=1)

    # Component legend with colored lines
    comp_lines = []
    for ref_name, w, pct, comp in components:
        pg = phase_label(ref_name)
        color = PHASE_COLORS.get(pg, COMP_COLORS[0])
        comp_lines.append(plt.Line2D([0], [0], color=color, lw=1.5,
                                      label=f"{pg} {pct:.0f}%"))

    # Add data, fit, and residual to legend
    legend_handles = [
        plt.Line2D([0], [0], color='k', lw=1.2, label='Data'),
        plt.Line2D([0], [0], color='r', lw=1.0, ls='--', label='LCF fit'),
    ] + comp_lines + [
        plt.Line2D([0], [0], color='gray', lw=0.7, label='Residual'),
    ]

    ax.legend(handles=legend_handles, loc='upper right', fontsize=5,
              frameon=True, fancybox=False, edgecolor='gray',
              handlelength=1.5, handletextpad=0.4, borderpad=0.3,
              labelspacing=0.3)

    # Panel label
    ax.text(0.02, 0.97, panel_labels_list[idx],
            transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='top', ha='left')

    # Axis formatting — despine
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if row_idx == 2:
        ax.set_xlabel("Energy (eV)", fontsize=8)
    if col_idx == 0:
        ax.set_ylabel("Flattened µ(E)", fontsize=8)

    ax.tick_params(labelsize=7, direction='in', top=False, right=False)
    ax.set_xlim(PLOT_E_MIN, PLOT_E_MAX)
    # Extend y-axis upward to make room for legend
    ymin, ymax = ax.get_ylim()
    extra = 0.15 * (ymax - ymin)
    if grp == 4:  # panel (e) needs a bit more room
        extra += 0.05
    ax.set_ylim(ymin, ymax + extra)

# Save
fig.savefig("figure_xanes_lcf.png", dpi=DPI, bbox_inches='tight',
            facecolor='white', pad_inches=0.05)
fig.savefig("figure_xanes_lcf.pdf", dpi=DPI, bbox_inches='tight',
            facecolor='white', pad_inches=0.05)
plt.close(fig)
print("\nFigure 2 saved: figure_xanes_lcf.png and .pdf")
