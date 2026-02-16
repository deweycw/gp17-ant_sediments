# Self-Absorption Screening for Fe K-edge µ-XANES Spectra

## Objective

Screen 172 Fe K-edge µ-XANES spectra (collected in fluorescence mode at GSECARS, APS sector 13) for self-absorption artifacts. Self-absorption in concentrated Fe phases suppresses the white line and broadens the absorption edge, potentially making Fe(III) oxyhydroxides appear spectrally similar to Fe(II) phases. The primary concern was whether cluster 4 (assigned as siderite + mackinawite by LCF) is actually self-absorbed ferrihydrite.

## Methods

### Fe Kα fluorescence intensity (Step 1)

Fe Kα counts were extracted at each XANES point location by matching area masks in the HDF5 µ-XRF map files to the corresponding ROI maps. 171 of 172 spectra were successfully matched (1 missing area: `GT15_FeTiXRD_striated2_4`).

### Spectral feature metrics (Step 2)

Four spectral indicators were computed from the flattened/normalized XANES spectra:

| Metric | Definition |
|--------|-----------|
| White line ratio (WLR) | max µ(E) in 7125–7135 eV / mean µ(E) in 7150–7170 eV |
| White line energy | Energy of WLR maximum |
| Edge energy (E0) | (1) Energy at µ = 0.5 on the rising edge; (2) Maximum of smoothed first derivative in 7118–7135 eV |
| Pre-edge centroid | Intensity-weighted mean energy of baseline-subtracted pre-edge peak (7110–7118 eV) |

### Flagging criteria (Step 3)

A spectrum was flagged as **possible** self-absorption if it met 2 or more of:

1. **Edge energy or pre-edge consistent with Fe(III):** E0 > 7122 eV or pre-edge centroid > 7113.5 eV
2. **Suppressed white line:** WLR < cluster 2 mean − 1σ (threshold = 1.40)
3. **Poor LCF fit:** R-factor > cluster median
4. **High Fe concentration:** Fe Kα counts > 75th percentile (51,576 counts)

A spectrum was flagged as **likely** if all 4 criteria were met.

### LCF refit with self-absorbed references (Step 5)

Flagged spectra were refit using the original 28 reference spectra plus:

- **Synthetic SA references:** 6-line ferrihydrite with empirical self-absorption dampening: µ\_SA(E) = µ(E) / (1 + α·µ(E)), renormalized to unit edge step, for α = 0.1, 0.2, 0.3, 0.5, 0.8
- **Empirical SA reference:** The cluster 2 spectrum with the highest Fe-counts-to-WLR ratio (`FeXANES_GT5_white_gray_particles_Fe_2.001`)

A spectrum was **confirmed** as self-absorbed if the R-factor improved by >20% when a self-absorbed reference was included in the best 3-component fit.

## Results

### Cluster-level spectral feature summary

| Cluster | n  | Fe Kα (mean) | WLR        | E0 deriv (eV) | Pre-edge (eV) | R-factor    |
|---------|----|-------------|------------|----------------|----------------|-------------|
| 1       | 51 | 31,319      | 1.41±0.09  | 7119.9±1.6     | 7111.1±0.9     | 0.014±0.006 |
| 2       | 36 | 24,116      | 1.48±0.08  | 7123.1±1.3     | 7112.7±0.5     | 0.011±0.003 |
| 3       | 26 | 46,458      | 1.31±0.07  | 7118.5±0.3     | —              | 0.021±0.006 |
| 4       | 33 | 76,013      | 1.19±0.04  | 7119.7±1.4     | 7112.2±0.7     | 0.020±0.010 |
| 5       | 26 | 44,753      | 1.29±0.08  | 7121.5±1.4     | 7112.9±0.2     | 0.015±0.006 |

### Flagging results

| Flag     | Count | By cluster                                       |
|----------|-------|--------------------------------------------------|
| Likely   | 0     | —                                                |
| Possible | 86    | C1: 16, C2: 9, C3: 14, C4: 32, C5: 15           |
| OK       | 86    | C1: 35, C2: 27, C3: 12, C4: 1, C5: 11           |

Most "possible" flags were triggered by WLR + R-factor (36 spectra) or WLR + Fe-counts (23 spectra). No spectra met all 4 criteria simultaneously.

### LCF refit confirmation

8 spectra were confirmed as self-absorbed (R-factor improved >20% with SA reference):

| Spectrum | Cluster | Fe Kα | WLR  | E0d (eV) | R original | R with SA | Improvement | SA reference |
|----------|---------|-------|------|-----------|------------|-----------|-------------|--------------|
| GT5_white_gray_particles_Fe_2 | 2 | 53,224 | 1.28 | 7121.4 | 0.0117 | 0.0000 | 100% | 6L-Fhy_SA_0.8 |
| GT5_white_gray_particles_Fe_3 | 5 | 49,735 | 1.25 | 7121.6 | 0.0140 | 0.0074 | 47% | 6L-Fhy_SA_0.8 |
| GT5_flaky_nodule_Fe_7 | 5 | 43,237 | 1.27 | 7122.2 | 0.0135 | 0.0071 | 48% | empirical_SA_C2 |
| GT5_flaky_nodule_Fe_8 | 2 | 38,636 | 1.30 | 7122.2 | 0.0132 | 0.0090 | 32% | empirical_SA_C2 |
| GT5_concentric_gray3_Fe_11 | 5 | 55,485 | 1.29 | 7121.6 | 0.0129 | 0.0093 | 28% | 6L-Fhy_SA_0.8 |
| GT5_flaky2_Fe_28 | 2 | 31,172 | 1.36 | 7122.2 | 0.0066 | 0.0050 | 25% | empirical_SA_C2 |
| GT15_FeTiXRD_striated2_5 | 5 | 96,538 | 1.18 | 7121.8 | 0.0217 | 0.0164 | 24% | 6L-Fhy_SA_0.8 |
| GT15_Fe_5 | 5 | 112,630 | 1.18 | 7121.6 | 0.0220 | 0.0172 | 22% | 6L-Fhy_SA_0.8 |

All confirmed spectra are in **cluster 2 (3 spectra) or cluster 5 (5 spectra)** — none in cluster 4.

## Interpretation

### Cluster 4 is not self-absorbed Fe(III)

The hypothesis that cluster 4 represents self-absorbed ferrihydrite is **not supported**:

1. **Edge energy is too low.** Cluster 4 has E0 (derivative) = 7119.7±1.4 eV, firmly in the Fe(II) range. Self-absorbed Fe(III) retains its edge position at ~7123 eV; only the white line amplitude is suppressed. Cluster 2 (genuine Fe(III)) has E0 = 7123.1±1.3 eV — the ~3.4 eV gap is too large to be a self-absorption artifact.

2. **Pre-edge centroid is consistent with Fe(II).** Cluster 4 pre-edge centroid = 7112.2 eV, lower than cluster 2 (7112.7 eV). Self-absorption does not shift the pre-edge position.

3. **LCF refit shows no improvement.** When self-absorbed ferrihydrite references (empirical and synthetic with α = 0.1–0.8) were included as candidate references, no cluster 4 spectrum showed >20% R-factor improvement. The Fe(II) references (siderite, mackinawite) remain the best fit.

4. **Low WLR is expected for these phases.** Siderite, mackinawite, and biotite inherently have weak Fe K-edge white lines compared to ferrihydrite. The WLR of ~1.19 for cluster 4 is consistent with these Fe(II) mineral references, not with dampened Fe(III).

### Mild self-absorption in clusters 2 and 5

The 8 confirmed SA spectra in clusters 2 and 5 likely reflect mild self-absorption in Fe(III) oxyhydroxide phases and Fe-Ti oxides (ilmenite) at moderate Fe concentrations. These spectra have:
- E0 > 7121 eV (consistent with Fe(III) or mixed Fe(III)/Fe(II))
- Moderate Fe Kα counts (31k–113k)
- Fits that improve substantially with a dampened 6L-Fhy or empirical SA reference

The `white_gray_particles_Fe_2` spectrum (cluster 2, 100% improvement) was used as the empirical SA reference itself, so its perfect self-fit is expected.

## Recommendation

- **Cluster 4 assignments are robust.** No removal or reclassification is warranted.
- **8 spectra in clusters 2 and 5** show evidence of mild self-absorption. These could be:
  - Excluded from PCA and re-assigned manually
  - Retained with a note that their LCF compositions may underestimate Fe(III) oxyhydroxide contribution
  - Refit with self-absorption-corrected references for more accurate phase quantification
- The high Fe Kα counts in cluster 4 (mean 76k, max 131k) do increase self-absorption risk, but the spectral features unambiguously indicate Fe(II) character. The concentrated Fe is genuine siderite/mackinawite.

## Output files

| File | Description |
|------|-------------|
| `fe_intensity.csv` | Fe Kα fluorescence counts at each XANES point (from µ-XRF maps) |
| `spectral_features.csv` | WLR, edge energy, pre-edge centroid, R-factor for all 172 spectra |
| `sa_flags.csv` | Self-absorption flag (likely/possible/ok) and criteria met |
| `lcf_refit.csv` | LCF refit results for flagged spectra with SA references |
| `full_results.csv` | Merged dataset with all features, flags, and refit results |
| `full_screening_data.csv` | Intermediate merged dataset (features + flags + Fe counts) |
| `plot1_wlr_vs_e0.png` | White line ratio vs. edge energy (derivative), colored by cluster |
| `plot2_fe_vs_rfactor.png` | Fe Kα counts vs. LCF R-factor |
| `plot3_wlr_vs_fe.png` | White line ratio vs. Fe Kα counts |
| `plot4_wlr_histogram.png` | WLR distribution by cluster |
| `plot5_flagged_spectra.png` | Flagged spectra overlaid with cluster 2 and 4 centroids |
