# GP17-ANT Sediments Project

## Project Overview
Fe K-edge µ-XANES and µ-XRF analysis of Antarctic marine sediments from GEOTRACES GP17-ANT cruise. Three stations (5, 15, 27) with depth profiles. Data collected at GSECARS (APS sector 13) in fluorescence mode.

## Key Architecture
- **172 point XANES spectra** → PCA (5 components) → k-means (k=5) → LCF against 28 Fe mineral references
- **Cluster 3** is sub-divided into **3a** (pyrrhotite-rich) and **3b** (mackinawite-rich) via k-means on PC scores
- Sub-cluster assignment uses Pyrrhotite LCF fraction to determine which sub-label is 3a vs 3b
- **Bulk spectra**: station × depth LCF with grouped phase classes, stacked-area depth profiles
- **HDF5 maps**: µ-XRF element maps with XANES point areas stored in `xrmmap/areas/`

## Notebook Structure (split from monolithic `pca-clustering-2.ipynb`)
| Notebook | Purpose |
|----------|---------|
| `01_pca_clustering.ipynb` | Data loading, PCA, k-means, quality screening, export `cluster_assignments.csv` |
| `02_target_transform.ipynb` | Test references against PCA model |
| `03_lcf_microprobe.ipynb` | LCF centroids, individual spectra, sub-clustering → `lcf_individual.csv` |
| `04_lcf_bulk.ipynb` | Bulk XANES LCF with depth profiles |
| `05_xfm_maps.ipynb` | Interactive XFM tricolor maps (ipywidgets) |
| `figures.ipynb` | **Publication figures 1 & 2** — combined, shared data loading, cross-referenced |
| `xrf_correlation.ipynb` | Interactive element correlations (ipywidgets) |

Notebooks 02-05 and figures.ipynb re-run minimal pipeline steps to get required variables.
The original `pca-clustering-2.ipynb` is preserved but superseded by the split notebooks.

## Scripts (standalone, `matplotlib.use("Agg")`)
- `make_figure1.py` / `make_figure2.py` — publication figures (headless)
- `xrf_correlation.py` — pixel-level and cluster-specific XRF correlations
- `self_absorption_screening.py` — SA screening for fluorescence XANES

## Key Data Files
- `pca_results/cluster_assignments.csv` — spectrum, cluster (1-5), PC scores
- `pca_results/lcf_individual.csv` — per-spectrum LCF weights, R-factor
- `bulk/bulk_lcf_mineral_refs.csv` — bulk LCF mineral-level results
- `bulk/bulk_lcf_grouped.csv` — bulk LCF grouped by phase class
- `self_absorption_screening/SUMMARY.md` — SA screening results and interpretation

## HDF5 Map Structure
- ROI maps: `f["xrmmap/roimap/sum_cor"][:, 1:-1, idx]` (note column slicing)
- ROI names: `f["xrmmap/roimap/sum_name"][:]` (bytes, need decode)
- Areas: `f["xrmmap/areas/<name>"][:]` → boolean mask
- Positions: `f["xrmmap/positions/pos"][:, 1:-1, 0/1]` → x/y in mm
- Spectrum naming: `FeXANES_<area_name>.001`

## Important Findings
- **Cluster 4 is NOT self-absorbed Fe(III)** — confirmed Fe(II) (siderite + mackinawite) by edge energy, pre-edge, and LCF refit
- 8 spectra in clusters 2 & 5 show mild self-absorption (Fe(III) oxyhydroxides)

## Style Notes
- Publication figures: 180mm width, 300 DPI, save both PNG and PDF
- Cluster colors: `{1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728', 5: '#9467bd'}`
- Cluster markers: `{1: 'o', 2: 's', '3a': '^', '3b': 'v', 4: 'D', 5: 'p'}`
- XFM maps default RGB: Fe Ka (red), Ca Ka (green), K Ka (blue)
- Tick labels on density scatter plots use scientific notation (`ticklabel_format`)
