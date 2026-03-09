"""
Parse feffit report files and extract fit parameters into a CSV.

Reads all .txt files in bulk/feffit_reports/ and outputs
bulk/feffit_parameters.csv with one row per sample.
"""

import re
import csv
from pathlib import Path

REPORT_DIR = Path(__file__).parent / "feffit_reports"
OUTPUT_CSV = Path(__file__).parent / "feffit_parameters.csv"

# Shells present in all reports
SHELLS = ["O203", "Fe302", "Al328"]


def parse_sample_name(filename):
    """Extract sample name from report filename.

    e.g. 'GP17_station5_19119_1cm_015_A_fl_avg_csv_Mar-07_23_55.txt'
      -> 'GP17_station5_19119_1cm'
    """
    # Match up to the depth (e.g. _1cm, _11cm, _15cm)
    m = re.match(r"(GP17_station\d+_\d+_\d+cm)", filename)
    if m:
        return m.group(1)
    return filename


def parse_report(filepath):
    """Parse a single feffit report file and return a dict of parameters."""
    text = filepath.read_text()
    result = {}

    # --- Statistics ---
    for key, pattern in [
        ("chi_sq", r"chi_square\s+=\s+([\d.eE+-]+)"),
        ("red_chi_sq", r"reduced chi_square\s+=\s+([\d.eE+-]+)"),
        ("r_factor", r"r-factor\s+=\s+([\d.eE+-]+)"),
    ]:
        m = re.search(pattern, text)
        if m:
            result[key] = float(m.group(1))

    # --- Parameters section ---
    # Match lines like:  N_O203  =  5.4500000 +/- 0.5840223  (init= ...)
    # or fixed:          s02     =  0.7000000 (fixed)
    param_pattern = re.compile(
        r"^\s+([\w]+)\s+=\s+([\d.eE+-]+)\s+\+/-\s+([\d.eE+-]+)", re.MULTILINE
    )
    fixed_pattern = re.compile(
        r"^\s+([\w]+)\s+=\s+([\d.eE+-]+)\s+\(fixed\)", re.MULTILINE
    )

    params = {}
    for m in param_pattern.finditer(text):
        name = m.group(1)
        params[name] = (float(m.group(2)), float(m.group(3)))

    for m in fixed_pattern.finditer(text):
        name = m.group(1)
        params[name] = (float(m.group(2)), None)  # no uncertainty for fixed

    # s02
    if "s02" in params:
        result["s02"] = params["s02"][0]

    # e0
    if "e0" in params:
        result["e0"] = params["e0"][0]
        result["e0_uncert"] = params["e0"][1]

    # Per-shell parameters
    for shell in SHELLS:
        for prefix in ["N", "delr", "sigma2"]:
            key = f"{prefix}_{shell}"
            if key in params:
                result[key] = params[key][0]
                result[f"{key}_uncert"] = params[key][1]

    # --- Reff from [[Paths]] section ---
    # Match lines like:  reff   =  3.2788000
    # Need to associate each reff with its path name
    # Path headers look like: = Path 'Fe_Al328' Fe K Edge
    path_blocks = re.split(r"= Path '(Fe_\w+)'", text)
    # path_blocks: [preamble, name1, block1, name2, block2, ...]
    for i in range(1, len(path_blocks) - 1, 2):
        path_name = path_blocks[i]  # e.g. 'Fe_Al328'
        block = path_blocks[i + 1]
        # Extract shell suffix from path name
        shell = path_name.replace("Fe_", "")  # e.g. 'Al328'
        if shell in SHELLS:
            reff_m = re.search(r"reff\s+=\s+([\d.eE+-]+)", block)
            if reff_m:
                reff = float(reff_m.group(1))
                result[f"Reff_{shell}"] = reff
                # R = Reff + delr; uncertainty on R equals uncertainty on delr
                delr_key = f"delr_{shell}"
                if delr_key in result:
                    result[f"R_{shell}"] = reff + result[delr_key]
                    result[f"R_{shell}_uncert"] = result.get(f"{delr_key}_uncert")

    return result


def main():
    reports = sorted(REPORT_DIR.glob("*.txt"))
    if not reports:
        print(f"No report files found in {REPORT_DIR}")
        return

    rows = []
    for rpt in reports:
        params = parse_report(rpt)
        params["sample"] = parse_sample_name(rpt.name)
        rows.append(params)

    # Sort by station then depth
    def sort_key(row):
        m = re.match(r"GP17_station(\d+)_\d+_(\d+)cm", row["sample"])
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return (999, 999)

    rows.sort(key=sort_key)

    # Build column order
    columns = ["sample", "s02", "e0", "e0_uncert"]
    for shell in SHELLS:
        columns.append(f"N_{shell}")
        columns.append(f"N_{shell}_uncert")
        columns.append(f"delr_{shell}")
        columns.append(f"delr_{shell}_uncert")
        columns.append(f"sigma2_{shell}")
        columns.append(f"sigma2_{shell}_uncert")
        columns.append(f"Reff_{shell}")
        columns.append(f"R_{shell}")
        columns.append(f"R_{shell}_uncert")
    columns.extend(["chi_sq", "red_chi_sq", "r_factor"])

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
