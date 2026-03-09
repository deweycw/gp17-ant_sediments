"""
Box-and-whisker plots of fitted R (Reff + delr) for each scattering path,
with individual sample points overlaid.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ---------- Config ----------
DPI = 300
TARGET_WIDTH_MM = 180
TARGET_WIDTH_IN = TARGET_WIDTH_MM / 25.4
OUT_DIR = Path("bulk")
CSV_PATH = Path("bulk/feffit_parameters.csv")

SHELLS = ["O203", "Fe302", "Al328"]
LABELS = {"O203": "Fe–O", "Fe302": "Fe–Fe", "Al328": "Fe–Al"}
COLORS = {"O203": "#1f77b4", "Fe302": "#d62728", "Al328": "#2ca02c"}

STATIONS = [5, 15, 27]
STATION_LABELS = {5: "Stn 5", 15: "Stn 15", 27: "Stn 27"}


def parse_station(sample):
    import re
    m = re.search(r"station(\d+)", sample)
    return int(m.group(1)) if m else None


def main():
    df = pd.read_csv(CSV_PATH)
    df["station"] = df["sample"].apply(parse_station)

    fig, axes = plt.subplots(
        1, len(SHELLS),
        figsize=(TARGET_WIDTH_IN, TARGET_WIDTH_IN * 0.35),
    )

    for ax, shell in zip(axes, SHELLS):
        data = [df.loc[df["station"] == stn, f"R_{shell}"] for stn in STATIONS]
        tick_labels = [STATION_LABELS[s] for s in STATIONS]
        c = COLORS[shell]

        bp = ax.boxplot(
            data,
            tick_labels=tick_labels,
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=1),
            capprops=dict(linewidth=1),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(c)
            patch.set_alpha(0.4)
            patch.set_edgecolor(c)

        # Overlay individual points
        for i, stn in enumerate(STATIONS):
            vals = df.loc[df["station"] == stn, f"R_{shell}"]
            ax.scatter(
                [i + 1] * len(vals), vals,
                color=c, edgecolor="black", linewidth=0.5,
                s=20, zorder=3,
            )

        ax.set_title(LABELS[shell])
        ax.set_ylabel("R (Å)")
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.005))

    fig.tight_layout()

    for fmt in ("png", "pdf"):
        out = OUT_DIR / f"R_boxplots.{fmt}"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"Saved to {OUT_DIR}/R_boxplots.{{png,pdf}}")


if __name__ == "__main__":
    main()
