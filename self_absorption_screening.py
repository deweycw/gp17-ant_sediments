"""
Self-Absorption Screening for Fe K-edge µ-XANES Spectra
Steps 2–6: Spectral features, SA flagging, diagnostics, LCF refit, summary.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from scipy.signal import savgol_filter
from itertools import combinations
from pathlib import Path
import os, warnings
warnings.filterwarnings("ignore")

# ---------- Config ----------
SPEC_DIR = Path("flattened-spectra")
REF_DIR = Path("FeK-standards/fluorescence/flattened")
PCA_DIR = Path("pca_results")
OUT_DIR = Path("self_absorption_screening")
OUT_DIR.mkdir(exist_ok=True)

E_MIN, E_MAX, E_STEP = 7100, 7180, 0.2
E_GRID = np.arange(E_MIN, E_MAX + E_STEP / 2, E_STEP)

# Reference names
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

# Cluster colors
CLUSTER_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728', 5: '#9467bd'}
CLUSTER_LABELS = {1: 'Grp 1', 2: 'Grp 2', 3: 'Grp 3', 4: 'Grp 4', 5: 'Grp 5'}

DPI = 300


# ---------- Helpers ----------
def load_csv_spectrum(filepath):
    data = np.loadtxt(filepath, comments='#', delimiter=',')
    return data[:, 0], data[:, 1]

def interp_to_grid(energy, mu, grid=E_GRID):
    return np.interp(grid, energy, mu)

def find_spec_file(spec_name):
    """Find the CSV file for a spectrum name."""
    fp = SPEC_DIR / (spec_name + ".csv")
    if fp.exists():
        return fp
    # Try alternate patterns
    base = spec_name.replace(".001", "")
    for p in SPEC_DIR.glob(base + "*"):
        return p
    return None


# ==========================================================
# STEP 2: Spectral feature screening
# ==========================================================
print("=" * 60)
print("STEP 2: Computing spectral features")
print("=" * 60)

# Load data
clusters_df = pd.read_csv(PCA_DIR / "cluster_assignments.csv")
lcf_df = pd.read_csv(PCA_DIR / "lcf_individual.csv")

results = []

for _, row in clusters_df.iterrows():
    spec_name = row["spectrum"]
    cluster = row["cluster"]

    fp = find_spec_file(spec_name)
    if fp is None:
        print(f"  WARNING: spectrum file not found for {spec_name}")
        continue

    energy, mu = load_csv_spectrum(fp)
    mu_grid = interp_to_grid(energy, mu)

    # White line intensity: max in 7125-7135 eV
    wl_mask = (E_GRID >= 7125) & (E_GRID <= 7135)
    wl_intensity = mu_grid[wl_mask].max()
    wl_energy = E_GRID[wl_mask][np.argmax(mu_grid[wl_mask])]

    # Edge step: mean in 7150-7170 eV
    post_mask = (E_GRID >= 7150) & (E_GRID <= 7170)
    edge_step = mu_grid[post_mask].mean()

    # White line ratio
    wl_ratio = wl_intensity / edge_step if edge_step > 0 else 0

    # Edge energy E0: energy where mu = 0.5 (on the rising edge, 7115-7135)
    edge_mask = (E_GRID >= 7115) & (E_GRID <= 7135)
    e_edge = E_GRID[edge_mask]
    mu_edge = mu_grid[edge_mask]
    # Find first crossing of 0.5
    crossings = np.where(np.diff(np.sign(mu_edge - 0.5)))[0]
    if len(crossings) > 0:
        i = crossings[0]
        # Linear interpolation
        e0 = e_edge[i] + (0.5 - mu_edge[i]) / (mu_edge[i+1] - mu_edge[i]) * (e_edge[i+1] - e_edge[i])
    else:
        e0 = np.nan

    # Also compute E0 from derivative maximum
    deriv = np.gradient(mu_grid, E_GRID)
    deriv_smooth = savgol_filter(deriv, window_length=11, polyorder=3)
    edge_deriv_mask = (E_GRID >= 7118) & (E_GRID <= 7135)
    e0_deriv = E_GRID[edge_deriv_mask][np.argmax(deriv_smooth[edge_deriv_mask])]

    # Pre-edge centroid: 7110-7118 eV
    pe_mask = (E_GRID >= 7110) & (E_GRID <= 7118)
    e_pe = E_GRID[pe_mask]
    mu_pe = mu_grid[pe_mask]
    # Subtract a linear baseline (endpoints)
    baseline = np.interp(e_pe, [e_pe[0], e_pe[-1]], [mu_pe[0], mu_pe[-1]])
    mu_pe_sub = mu_pe - baseline
    mu_pe_sub = np.maximum(mu_pe_sub, 0)
    if mu_pe_sub.sum() > 0:
        preedge_centroid = np.average(e_pe, weights=mu_pe_sub)
    else:
        preedge_centroid = np.nan

    # Get R-factor from LCF
    lcf_row = lcf_df[lcf_df["spectrum"] == spec_name]
    r_factor = lcf_row["r_factor"].values[0] if len(lcf_row) > 0 else np.nan

    results.append({
        "spectrum": spec_name,
        "cluster": cluster,
        "wl_intensity": wl_intensity,
        "wl_energy": wl_energy,
        "e0": e0,
        "e0_deriv": e0_deriv,
        "preedge_centroid": preedge_centroid,
        "wl_ratio": wl_ratio,
        "edge_step": edge_step,
        "r_factor": r_factor,
    })

features_df = pd.DataFrame(results)
features_df.to_csv(OUT_DIR / "spectral_features.csv", index=False)
print(f"Computed features for {len(features_df)} spectra")
print(f"\nPer-cluster feature summary:")
for c in sorted(features_df["cluster"].unique()):
    sub = features_df[features_df["cluster"] == c]
    print(f"  Cluster {c} (n={len(sub)}): "
          f"WLR={sub['wl_ratio'].mean():.2f}±{sub['wl_ratio'].std():.2f}, "
          f"E0={sub['e0'].mean():.1f}±{sub['e0'].std():.1f}, "
          f"E0d={sub['e0_deriv'].mean():.1f}±{sub['e0_deriv'].std():.1f}, "
          f"PE={sub['preedge_centroid'].mean():.1f}±{sub['preedge_centroid'].std():.1f}, "
          f"R={sub['r_factor'].mean():.4f}±{sub['r_factor'].std():.4f}")


# ==========================================================
# STEP 3: Identify self-absorbed spectra
# ==========================================================
print("\n" + "=" * 60)
print("STEP 3: Flagging self-absorbed spectra")
print("=" * 60)

# Merge with Fe intensity
fe_df = pd.read_csv(OUT_DIR / "fe_intensity.csv")
merged = features_df.merge(fe_df[["spectrum", "fe_counts"]], on="spectrum", how="left")

# Compute thresholds
# Cluster 2 = Fe(III) oxyhydroxide reference group
c2 = merged[merged["cluster"] == 2]
c2_wlr_mean = c2["wl_ratio"].mean()
c2_wlr_std = c2["wl_ratio"].std()
wlr_threshold = c2_wlr_mean - 1.0 * c2_wlr_std

# Per-cluster median R-factor
cluster_median_r = merged.groupby("cluster")["r_factor"].median()

# Fe counts top quartile
fe_q75 = merged["fe_counts"].quantile(0.75)

print(f"Thresholds:")
print(f"  Cluster 2 WLR: {c2_wlr_mean:.2f} ± {c2_wlr_std:.2f}")
print(f"  WLR threshold (C2 mean - 1σ): {wlr_threshold:.2f}")
print(f"  Fe counts Q75: {fe_q75:.0f}")
print(f"  Cluster median R-factors: {dict(cluster_median_r.round(4))}")

flags = []
for _, row in merged.iterrows():
    score = 0
    criteria = []

    # Criterion 1: Edge energy consistent with Fe(III)
    c1 = (not np.isnan(row["e0"]) and row["e0"] > 7122) or \
         (not np.isnan(row["preedge_centroid"]) and row["preedge_centroid"] > 7113.5)
    if c1:
        score += 1
        criteria.append("E0/PE")

    # Criterion 2: White line suppressed relative to Fe(III)
    c2_flag = row["wl_ratio"] < wlr_threshold
    if c2_flag:
        score += 1
        criteria.append("WLR")

    # Criterion 3: Poor LCF fit
    c_median = cluster_median_r.get(row["cluster"], 0)
    c3 = row["r_factor"] > c_median
    if c3:
        score += 1
        criteria.append("R-factor")

    # Criterion 4: High Fe counts
    c4 = not np.isnan(row.get("fe_counts", np.nan)) and row["fe_counts"] > fe_q75
    if c4:
        score += 1
        criteria.append("Fe-counts")

    if score >= 4:
        flag = "likely"
    elif score >= 2:
        flag = "possible"
    else:
        flag = "ok"

    flags.append({
        "spectrum": row["spectrum"],
        "cluster": row["cluster"],
        "sa_flag": flag,
        "sa_score": score,
        "criteria_met": "+".join(criteria) if criteria else "",
    })

flags_df = pd.DataFrame(flags)
flags_df.to_csv(OUT_DIR / "sa_flags.csv", index=False)

print(f"\nSelf-absorption flags:")
for flag_val in ["likely", "possible", "ok"]:
    sub = flags_df[flags_df["sa_flag"] == flag_val]
    if len(sub) > 0:
        by_cluster = sub.groupby("cluster").size().to_dict()
        print(f"  {flag_val:10s}: {len(sub):3d} spectra  {by_cluster}")

# Merge all data for plotting
all_data = merged.merge(flags_df[["spectrum", "sa_flag", "sa_score", "criteria_met"]], on="spectrum")
all_data.to_csv(OUT_DIR / "full_screening_data.csv", index=False)


# ==========================================================
# STEP 4: Diagnostic plots
# ==========================================================
print("\n" + "=" * 60)
print("STEP 4: Creating diagnostic plots")
print("=" * 60)

# --- Plot 1: White line ratio vs E0, colored by cluster ---
fig, ax = plt.subplots(figsize=(7, 5), dpi=DPI)
for c in sorted(all_data["cluster"].unique()):
    sub = all_data[all_data["cluster"] == c]
    ok = sub[sub["sa_flag"] == "ok"]
    ax.scatter(ok["e0_deriv"], ok["wl_ratio"], c=CLUSTER_COLORS[c],
               label=CLUSTER_LABELS[c], s=30, alpha=0.7, edgecolors="none")

# Mark flagged spectra
for flag_val, marker, sz in [("possible", "s", 50), ("likely", "X", 70)]:
    sub = all_data[all_data["sa_flag"] == flag_val]
    if len(sub) > 0:
        ax.scatter(sub["e0_deriv"], sub["wl_ratio"], c=[CLUSTER_COLORS[c] for c in sub["cluster"]],
                   marker=marker, s=sz, edgecolors="red", linewidths=1.2,
                   label=f"{flag_val} SA", zorder=6)

ax.axhline(y=wlr_threshold, color='gray', ls='--', lw=0.8, label=f'WLR threshold ({wlr_threshold:.2f})')
ax.set_xlabel("Edge energy E0 (eV, derivative max)", fontsize=10)
ax.set_ylabel("White line ratio", fontsize=10)
ax.set_title("White Line Ratio vs. Edge Energy", fontsize=11)
ax.legend(fontsize=7, loc="upper left")
ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "plot1_wlr_vs_e0.png", dpi=DPI)
plt.close(fig)
print("  Saved plot1_wlr_vs_e0.png")

# --- Plot 2: Fe counts vs R-factor ---
fig, ax = plt.subplots(figsize=(7, 5), dpi=DPI)
for c in sorted(all_data["cluster"].unique()):
    sub = all_data[all_data["cluster"] == c]
    ok = sub[sub["sa_flag"] == "ok"]
    ax.scatter(ok["fe_counts"], ok["r_factor"], c=CLUSTER_COLORS[c],
               label=CLUSTER_LABELS[c], s=30, alpha=0.7, edgecolors="none")

for flag_val, marker, sz in [("possible", "s", 50), ("likely", "X", 70)]:
    sub = all_data[all_data["sa_flag"] == flag_val]
    if len(sub) > 0:
        ax.scatter(sub["fe_counts"], sub["r_factor"], c=[CLUSTER_COLORS[c] for c in sub["cluster"]],
                   marker=marker, s=sz, edgecolors="red", linewidths=1.2,
                   label=f"{flag_val} SA", zorder=6)

ax.axvline(x=fe_q75, color='gray', ls='--', lw=0.8, label=f'Fe Q75 ({fe_q75:.0f})')
ax.set_xlabel("Fe Kα counts", fontsize=10)
ax.set_ylabel("LCF R-factor", fontsize=10)
ax.set_title("Fe Counts vs. LCF R-factor", fontsize=11)
ax.legend(fontsize=7, loc="upper right")
ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "plot2_fe_vs_rfactor.png", dpi=DPI)
plt.close(fig)
print("  Saved plot2_fe_vs_rfactor.png")

# --- Plot 3: WLR vs Fe counts ---
fig, ax = plt.subplots(figsize=(7, 5), dpi=DPI)
for c in sorted(all_data["cluster"].unique()):
    sub = all_data[all_data["cluster"] == c]
    ok = sub[sub["sa_flag"] == "ok"]
    ax.scatter(ok["fe_counts"], ok["wl_ratio"], c=CLUSTER_COLORS[c],
               label=CLUSTER_LABELS[c], s=30, alpha=0.7, edgecolors="none")

for flag_val, marker, sz in [("possible", "s", 50), ("likely", "X", 70)]:
    sub = all_data[all_data["sa_flag"] == flag_val]
    if len(sub) > 0:
        ax.scatter(sub["fe_counts"], sub["wl_ratio"], c=[CLUSTER_COLORS[c] for c in sub["cluster"]],
                   marker=marker, s=sz, edgecolors="red", linewidths=1.2,
                   label=f"{flag_val} SA", zorder=6)

ax.axhline(y=wlr_threshold, color='gray', ls='--', lw=0.8)
ax.axvline(x=fe_q75, color='gray', ls='--', lw=0.8)
ax.set_xlabel("Fe Kα counts", fontsize=10)
ax.set_ylabel("White line ratio", fontsize=10)
ax.set_title("White Line Ratio vs. Fe Counts", fontsize=11)
ax.legend(fontsize=7, loc="upper right")
ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "plot3_wlr_vs_fe.png", dpi=DPI)
plt.close(fig)
print("  Saved plot3_wlr_vs_fe.png")

# --- Plot 4: WLR histograms by cluster ---
fig, ax = plt.subplots(figsize=(7, 4), dpi=DPI)
for c in sorted(all_data["cluster"].unique()):
    sub = all_data[all_data["cluster"] == c]
    ax.hist(sub["wl_ratio"], bins=20, alpha=0.5, color=CLUSTER_COLORS[c],
            label=CLUSTER_LABELS[c], edgecolor="white", linewidth=0.5)
ax.axvline(x=wlr_threshold, color='red', ls='--', lw=1, label=f'SA threshold')
ax.set_xlabel("White line ratio", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.set_title("White Line Ratio Distribution by Cluster", fontsize=11)
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "plot4_wlr_histogram.png", dpi=DPI)
plt.close(fig)
print("  Saved plot4_wlr_histogram.png")

# --- Plot 5: Representative spectra comparison ---
# Load cluster centroids (compute from all spectra in each cluster)
print("  Computing cluster centroids for comparison plot...")
cluster_spectra = {}
for c in [2, 4]:
    c_names = clusters_df[clusters_df["cluster"] == c]["spectrum"].values
    c_mus = []
    for sn in c_names:
        fp = find_spec_file(sn)
        if fp:
            e, mu = load_csv_spectrum(fp)
            c_mus.append(interp_to_grid(e, mu))
    cluster_spectra[c] = np.mean(c_mus, axis=0)

likely_sa = all_data[all_data["sa_flag"] == "likely"]
if len(likely_sa) == 0:
    likely_sa = all_data[all_data["sa_flag"] == "possible"].head(6)

n_plot = min(6, len(likely_sa))
if n_plot > 0:
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), dpi=DPI)
    axes = axes.flatten()
    plot_mask = (E_GRID >= 7105) & (E_GRID <= 7170)
    e_plot = E_GRID[plot_mask]

    for i in range(n_plot):
        ax = axes[i]
        row = likely_sa.iloc[i]
        fp = find_spec_file(row["spectrum"])
        if fp is None:
            continue
        e, mu = load_csv_spectrum(fp)
        mu_grid = interp_to_grid(e, mu)

        ax.plot(e_plot, mu_grid[plot_mask], 'k-', lw=1.2, label='Spectrum')
        ax.plot(e_plot, cluster_spectra[2][plot_mask], 'r-', lw=0.8, alpha=0.7, label='Grp 2 centroid')
        ax.plot(e_plot, cluster_spectra[4][plot_mask], 'b-', lw=0.8, alpha=0.7, label='Grp 4 centroid')

        short = row["spectrum"].replace("FeXANES_", "").replace(".001", "")
        ax.set_title(f"C{row['cluster']}: {short}\nWLR={row['wl_ratio']:.2f}, E0={row['e0_deriv']:.1f}",
                     fontsize=6.5)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=5.5)
        if i >= 3:
            ax.set_xlabel("Energy (eV)", fontsize=7)

    for i in range(n_plot, 6):
        axes[i].axis("off")

    fig.suptitle("Flagged Spectra vs. Cluster 2 (Fe(III)) and Cluster 4 Centroids", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot5_flagged_spectra.png", dpi=DPI)
    plt.close(fig)
    print("  Saved plot5_flagged_spectra.png")
else:
    print("  No flagged spectra to plot for Plot 5")


# ==========================================================
# STEP 5: LCF test with self-absorbed reference
# ==========================================================
print("\n" + "=" * 60)
print("STEP 5: LCF refit with self-absorbed references")
print("=" * 60)

# Load reference spectra
print("  Loading reference spectra...")
ref_spectra = {}
for name in REF_NAMES:
    fp = REF_DIR / (name + ".csv")
    if fp.exists():
        e, mu = load_csv_spectrum(fp)
        ref_spectra[name] = interp_to_grid(e, mu)

# Generate self-absorbed ferrihydrite references
fhy_6L = ref_spectra.get("6L-Fhy")
sa_alphas = [0.1, 0.2, 0.3, 0.5, 0.8]
sa_refs = {}
if fhy_6L is not None:
    for alpha in sa_alphas:
        mu_sa = fhy_6L / (1 + alpha * fhy_6L)
        # Renormalize so post-edge = 1.0
        post_mask = (E_GRID >= 7150) & (E_GRID <= 7170)
        post_mean = mu_sa[post_mask].mean()
        if post_mean > 0:
            mu_sa = mu_sa / post_mean
        sa_refs[f"6L-Fhy_SA_{alpha}"] = mu_sa
    print(f"  Generated {len(sa_refs)} self-absorbed Fhy references")

# Also try Option B: most self-absorbed-looking cluster 2 spectrum
# (highest Fe counts + lowest WLR in cluster 2)
c2_data = all_data[all_data["cluster"] == 2].copy()
if len(c2_data) > 0:
    c2_data["sa_proxy"] = c2_data["fe_counts"] / c2_data["wl_ratio"]
    worst_c2 = c2_data.sort_values("sa_proxy", ascending=False).iloc[0]
    fp = find_spec_file(worst_c2["spectrum"])
    if fp:
        e, mu = load_csv_spectrum(fp)
        sa_refs["empirical_SA_C2"] = interp_to_grid(e, mu)
        print(f"  Empirical SA reference: {worst_c2['spectrum']} "
              f"(Fe={worst_c2['fe_counts']:.0f}, WLR={worst_c2['wl_ratio']:.2f})")

# LCF function for a single spectrum
def lcf_fit(mu_data, ref_dict, max_refs=3):
    """NNLS LCF with all combinations of 1-max_refs references."""
    ref_names_list = list(ref_dict.keys())
    n_refs = len(ref_names_list)

    # Build reference matrix on the fitting range
    fit_mask = (E_GRID >= 7110) & (E_GRID <= 7170)
    b = mu_data[fit_mask]

    ref_matrix = np.array([ref_dict[n][fit_mask] for n in ref_names_list])

    best = {"r_factor": np.inf}

    for n_ref in range(1, min(max_refs + 1, n_refs + 1)):
        for combo in combinations(range(n_refs), n_ref):
            A = ref_matrix[list(combo)].T
            weights, _ = nnls(A, b)
            fitted = A @ weights
            residual = b - fitted
            r_factor = np.sum(np.abs(residual)) / np.sum(np.abs(b))

            if r_factor < best["r_factor"]:
                best = {
                    "r_factor": r_factor,
                    "refs": [ref_names_list[i] for i in combo],
                    "weights": weights,
                    "weight_sum": weights.sum(),
                    "fitted": fitted,
                }

    return best

# Refit flagged spectra
flagged = all_data[all_data["sa_flag"].isin(["likely", "possible"])].copy()
print(f"\n  Refitting {len(flagged)} flagged spectra...")

refit_results = []
for _, row in flagged.iterrows():
    fp = find_spec_file(row["spectrum"])
    if fp is None:
        continue

    e, mu = load_csv_spectrum(fp)
    mu_grid = interp_to_grid(e, mu)

    # Original fit (original refs only)
    orig_fit = lcf_fit(mu_grid, ref_spectra, max_refs=3)

    # SA fit (original refs + self-absorbed refs)
    all_refs = {**ref_spectra, **sa_refs}
    sa_fit = lcf_fit(mu_grid, all_refs, max_refs=3)

    improvement = (orig_fit["r_factor"] - sa_fit["r_factor"]) / orig_fit["r_factor"] * 100

    # Check which SA ref was used
    sa_ref_used = [r for r in sa_fit["refs"] if r.startswith("6L-Fhy_SA") or r == "empirical_SA_C2"]
    best_sa_alpha = ""
    if sa_ref_used:
        best_sa_alpha = sa_ref_used[0]

    sa_confirmed = improvement > 20 and len(sa_ref_used) > 0

    refit_results.append({
        "spectrum": row["spectrum"],
        "cluster": row["cluster"],
        "sa_flag": row["sa_flag"],
        "r_factor_original": orig_fit["r_factor"],
        "r_factor_sa": sa_fit["r_factor"],
        "improvement_pct": improvement,
        "best_sa_alpha": best_sa_alpha,
        "sa_confirmed": sa_confirmed,
        "orig_refs": ", ".join(orig_fit["refs"]),
        "sa_refs": ", ".join(sa_fit["refs"]),
    })

    if sa_confirmed:
        print(f"    CONFIRMED: {row['spectrum']} "
              f"R: {orig_fit['r_factor']:.4f} -> {sa_fit['r_factor']:.4f} "
              f"({improvement:.1f}%) using {best_sa_alpha}")

refit_df = pd.DataFrame(refit_results)
refit_df.to_csv(OUT_DIR / "lcf_refit.csv", index=False)
print(f"  Saved lcf_refit.csv ({len(refit_df)} entries)")


# ==========================================================
# STEP 6: Summary report
# ==========================================================
print("\n" + "=" * 60)
print("STEP 6: Summary Report")
print("=" * 60)

n_total = len(all_data)
n_likely = len(flags_df[flags_df["sa_flag"] == "likely"])
n_possible = len(flags_df[flags_df["sa_flag"] == "possible"])
n_confirmed = refit_df["sa_confirmed"].sum() if len(refit_df) > 0 else 0

print(f"""
Self-absorption screening results:
  Total spectra: {n_total}
  Likely self-absorbed: {n_likely}""")

for flag_val in ["likely", "possible"]:
    sub = flags_df[flags_df["sa_flag"] == flag_val]
    if len(sub) > 0:
        by_cluster = sub.groupby("cluster").size()
        detail = ", ".join([f"{n} in cluster {c}" for c, n in by_cluster.items()])
        print(f"    {flag_val}: {len(sub)} ({detail})")

print(f"  Confirmed by LCF refit: {n_confirmed}")

if n_confirmed > 0:
    confirmed_clusters = refit_df[refit_df["sa_confirmed"]].groupby("cluster").size()
    print(f"  Confirmed by cluster: {dict(confirmed_clusters)}")

# Detailed breakdown of criteria met
print(f"\n  Criteria breakdown for flagged spectra:")
for flag_val in ["likely", "possible"]:
    sub = flags_df[flags_df["sa_flag"] == flag_val]
    if len(sub) > 0:
        print(f"    {flag_val} ({len(sub)}):")
        criteria_counts = sub["criteria_met"].value_counts()
        for c, n in criteria_counts.items():
            print(f"      {c}: {n}")

# Final merged output
full_results = all_data.merge(
    refit_df[["spectrum", "r_factor_original", "r_factor_sa", "improvement_pct",
              "best_sa_alpha", "sa_confirmed"]],
    on="spectrum", how="left"
)
full_results.to_csv(OUT_DIR / "full_results.csv", index=False)
print(f"\n  Saved full_results.csv ({len(full_results)} rows)")

# Recommendation
if n_likely + n_confirmed > 0:
    remove_set = set()
    if n_confirmed > 0:
        remove_set.update(refit_df[refit_df["sa_confirmed"]]["spectrum"].values)
    remove_set.update(flags_df[flags_df["sa_flag"] == "likely"]["spectrum"].values)
    affected = flags_df[flags_df["spectrum"].isin(remove_set)].groupby("cluster").size()
    print(f"\n  Recommendation: Review {len(remove_set)} spectra before re-running PCA")
    print(f"  Affected clusters: {dict(affected)}")
else:
    print(f"\n  Recommendation: No spectra require removal based on current thresholds.")
    if n_possible > 0:
        print(f"  However, {n_possible} spectra are flagged as 'possible' — review the diagnostic plots.")

print("\nDone.")
