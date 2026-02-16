"""
Figure 2: Representative point XANES with LCF fits for each group.
6-panel figure (2 columns x 3 rows), one panel per group.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import nnls
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

# Plot range
PLOT_E_MIN, PLOT_E_MAX = 7105, 7170

# Map panel spectrum prefixes (spectra visible on the 4 maps in Figure 1)
MAP_PREFIXES = [
    "FeXANES_GT15_FeTiXRD_striated2_",
    "FeXANES_GT15_Fe_striated2_",
    "FeXANES_GT15_rectangles_Fe_",
    "FeXANES_GT5_flaky_nodule_Fe",
    "FeXANES_GT15_flakydark_Fe_",
]

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
    ("FeXANES_GT15_FeTiXRD_striated2_", "(a)"),
    ("FeXANES_GT15_Fe_striated2_",      "(a)"),
    ("FeXANES_GT15_rectangles_Fe_",     "(b)"),
    ("FeXANES_GT5_flaky_nodule_Fe",     "(c)"),
    ("FeXANES_GT15_flakydark_Fe_",      "(d)"),
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

# ---------- Select representative spectra ----------
print("\nSelecting representative spectra...")
ref_columns = [c for c in lcf_df.columns if c not in
               ["spectrum", "cluster", "r_factor", "chi_sq", "weight_sum", "n_refs"]]

selected = {}
for grp in [1, 2, "3a", "3b", 4, 5]:
    grp_str = str(grp)
    grp_df = df[df["group"] == grp_str].copy()

    # Filter to map spectra first
    map_mask = grp_df["spectrum"].apply(is_map_spectrum)
    candidates = grp_df[map_mask] if map_mask.any() else grp_df

    # Prefer 3+ component fits
    multi_comp = candidates[candidates["n_refs"] >= 3]
    if len(multi_comp) >= 1:
        candidates = multi_comp

    # Prefer spectra with R-factor <= 0.025 (good fits)
    good_fits = candidates[candidates["r_factor"] <= 0.025]
    if len(good_fits) >= 1:
        candidates = good_fits

    # Pick spectrum closest to median R-factor
    median_r = candidates["r_factor"].median()
    idx = (candidates["r_factor"] - median_r).abs().idxmin()
    row = candidates.loc[idx]
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
fig.subplots_adjust(hspace=0.35, wspace=0.25, left=0.08, right=0.97, top=0.97, bottom=0.06)

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

    # Get the active references and their weights from LCF results
    active_refs = []
    for ref_name in ref_columns:
        w = info[ref_name]
        if w > 0.005:  # threshold
            active_refs.append((ref_name, w))
    active_refs.sort(key=lambda x: x[1], reverse=True)

    # Reconstruct fit from stored weights
    weight_sum = info["weight_sum"]
    fit = np.zeros_like(E_GRID)
    components = []
    for ref_name, w in active_refs:
        if ref_name in ref_spectra:
            comp = w * ref_spectra[ref_name]
            fit += comp
            pct = w / weight_sum * 100 if weight_sum > 0 else 0
            components.append((ref_name, w, pct, comp))

    residual = mu_grid - fit

    # R-factor
    r_factor = info["r_factor"]

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

    # Component text annotation using phase group names
    comp_text_parts = []
    for ref_name, w, pct, comp in components:
        pg = phase_label(ref_name)
        comp_text_parts.append(f"{pg} {pct:.0f}%")
    comp_text = "\n".join(comp_text_parts)

    # Text annotation with R-factor and components
    ax.text(0.98, 0.97, f"R = {r_factor:.4f}\n{comp_text}",
            transform=ax.transAxes, fontsize=5.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray', lw=0.3))

    # Panel label and title — cross-reference to Figure 1 map panel + point
    panel_ref = map_panel_label(spec_name)
    title_str = f"Group {grp}"
    if panel_ref:
        title_str += f"  ({panel_ref})"
    ax.text(0.02, 0.97, f"{panel_labels_list[idx]}",
            transform=ax.transAxes, fontsize=9, fontweight='bold', va='top', ha='left')
    ax.set_title(title_str, fontsize=7.5, pad=4)

    # Axis formatting
    if row_idx == 2:
        ax.set_xlabel("Energy (eV)", fontsize=8)
    else:
        ax.set_xticklabels([])
    if col_idx == 0:
        ax.set_ylabel("Flattened µ(E)", fontsize=8)

    ax.tick_params(labelsize=7, direction='in', top=True, right=True)
    ax.set_xlim(PLOT_E_MIN, PLOT_E_MAX)

# Save
fig.savefig("figure_xanes_lcf.png", dpi=DPI, bbox_inches='tight',
            facecolor='white', pad_inches=0.05)
fig.savefig("figure_xanes_lcf.pdf", dpi=DPI, bbox_inches='tight',
            facecolor='white', pad_inches=0.05)
plt.close(fig)
print("\nFigure 2 saved: figure_xanes_lcf.png and .pdf")
