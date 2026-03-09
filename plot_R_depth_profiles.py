"""
Plot fitted R (Reff + delr) vs depth for each scattering path,
one panel per station/core. Error bars from delr uncertainty.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
from pathlib import Path

# ---------- Config ----------
DPI = 300
TARGET_WIDTH_MM = 180
TARGET_WIDTH_IN = TARGET_WIDTH_MM / 25.4
OUT_DIR = Path("bulk")
CSV_PATH = Path("bulk/feffit_parameters.csv")

SHELLS = {
    "O203":  {"label": "Fe–O",  "color": "#1f77b4"},
    "Fe302": {"label": "Fe–Fe", "color": "#d62728"},
    "Al328": {"label": "Fe–Al", "color": "#2ca02c"},
}

STATION_LABELS = {
    5:  "Station 5",
    15: "Station 15",
    27: "Station 27",
}


def parse_depth(sample):
    m = re.search(r"_(\d+)cm", sample)
    return int(m.group(1)) if m else None


def parse_station(sample):
    m = re.search(r"station(\d+)", sample)
    return int(m.group(1)) if m else None


def main():
    df = pd.read_csv(CSV_PATH)
    df["depth_cm"] = df["sample"].apply(parse_depth)
    df["station"] = df["sample"].apply(parse_station)

    stations = sorted(df["station"].unique())
    n_stations = len(stations)

    fig, axes = plt.subplots(
        1, n_stations,
        figsize=(TARGET_WIDTH_IN, TARGET_WIDTH_IN * 0.4),
        sharey=True,
    )
    if n_stations == 1:
        axes = [axes]

    for ax, stn in zip(axes, stations):
        sub = df[df["station"] == stn].sort_values("depth_cm")

        for shell, props in SHELLS.items():
            ax.errorbar(
                sub[f"R_{shell}"],
                sub["depth_cm"],
                xerr=sub[f"R_{shell}_uncert"],
                fmt="o-",
                color=props["color"],
                label=props["label"],
                capsize=3,
                markersize=4,
                linewidth=1,
            )

        ax.set_title(STATION_LABELS.get(stn, f"Station {stn}"))
        ax.set_xlabel("R (Å)")
        ax.invert_yaxis()

    axes[0].set_ylabel("Depth (cm)")
    axes[-1].legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        out = OUT_DIR / f"R_depth_profiles.{fmt}"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"Saved to {OUT_DIR}/R_depth_profiles.{{png,pdf}}")


if __name__ == "__main__":
    main()
