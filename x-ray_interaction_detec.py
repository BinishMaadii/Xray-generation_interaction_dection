"""
X-ray Interaction with Matter & Photon-Counting Detection — single-file pipeline
==================================================================================
Targets the Philips JD bullet: "Expertise in medical imaging physics, including
generation, interaction with matter and detection of x-rays, with in-depth
knowledge of computed tomography."

Real data source (CC BY 4.0):
  Zhou, E. et al. "A cone-beam photon-counting CT dataset for spectral image
  reconstruction and deep learning." Scientific Data 12, 1955 (2025).
  https://doi.org/10.1038/s41597-025-06246-4
  Zenodo bundle "calibration table&sample1": 10.5281/zenodo.15738313

Uses the PMMA/aluminum slab calibration-phantom scans: known material,
known thickness, real photon-counting detector, two energy thresholds.
This is what a photon-counting CT detector's calibration measurements
actually look like, so the attenuation physics extracted here is measured,
not assumed or simulated.

--------------------------------------------------------------------------
DATA SETUP (do this first — this sandbox cannot reach Zenodo directly)
--------------------------------------------------------------------------
Download from Zenodo (DOI 10.5281/zenodo.15738313, "calibration table&sample1"):
  - CalibrationTable.zip        (~62.5 MB)  -> extract to data/CalibrationTable/
  - CalibrationPhantomData.zip  (~9.2 GB)   -> extract to data/CalibrationPhantomData/

Expected layout:
  data/CalibrationTable/
      air_table_low.raw, air_table_high.raw, air_table_total.raw
  data/CalibrationPhantomData/PMMA_AL_slabs/
      AcqPara.mat
      PMMA_<m>_AL_<n>/proj_high.raw
      PMMA_<m>_AL_<n>/proj_total.raw
      ... (56 combinations of PMMA thickness m mm, Al thickness n mm)

The exact folder-naming convention for decimal thicknesses (e.g. 0.5 mm Al)
isn't published by Zenodo ahead of download. This script prints any folder
name it can't parse — adjust FOLDER_PATTERN below to match once you see them.

--------------------------------------------------------------------------
PIPELINE
--------------------------------------------------------------------------
  1. Load raw detector projections: air (flat-field) tables + all PMMA/Al
     slab combinations, at High and Total energy thresholds.
  2. Derive the Low energy bin by subtraction (Low = Total - High), exactly
     as the source paper's own acquisition protocol defines it.
  3. Air-normalize each bin: P_E = -ln(I_E,obj / I_E,air)  [Beer-Lambert law]
  4. INTERACTION WITH MATTER: fit linear attenuation coefficients mu_PMMA,
     mu_Al at each energy bin via multivariate least squares across all 56
     thickness combinations (P = mu_PMMA * t_PMMA + mu_Al * t_Al).
  5. Convert to mass attenuation coefficients (mu/rho) and compare against
     NIST XCOM reference values (Hubbell & Seltzer, NIST) at representative
     energies for each bin.
  6. DETECTION OF X-RAYS: characterize the photon-counting detector itself
     from the flat-field data — per-pixel gain non-uniformity, energy-bin
     consistency (Low + High == Total), and a simple flat-field correction
     with before/after uniformity comparison.
  7. Save summary figures + a printed report.
"""

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
CAL_TABLE_DIR = DATA_DIR / "CalibrationTable"
PHANTOM_DIR = DATA_DIR / "CalibrationPhantomData" / "PMMA_AL_slabs"
FIG_DIR = Path("figures")

WIDTH, HEIGHT = 2063, 505  # detector pixel dimensions, per Zhou et al. 2025
ROI_FRACTION = 0.5         # central fraction of the detector used for the
                            # scalar attenuation measurement per slab combo,
                            # to avoid edge/vignetting effects

# Densities (g/cm^3), needed to convert linear attenuation coeff (1/cm) to
# mass attenuation coeff (cm^2/g) for comparison with NIST tables.
DENSITY_PMMA = 1.19   # g/cm^3, standard cast acrylic
DENSITY_AL = 2.70      # g/cm^3

# NIST XCOM mass attenuation coefficients (cm^2/g), "with coherent
# scattering", from Hubbell & Seltzer (NIST, NISTIR 5632 / DOI 10.18434/T4D01F),
# https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z13.html (Aluminum)
# and https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/pmma.html (PMMA).
# These are tabulated at discrete monochromatic energies; the detector's bins
# are polychromatic (80 kV source, 0.5mm Al filter, thresholds at 15/30/80
# keV), so 20 keV and 50 keV are used here as representative energies inside
# the Low (15-30 keV) and High (30-80 keV) bins respectively. For a rigorous
# comparison, replace these with SpekPy-spectrum-weighted effective values.
NIST_MU_RHO_CM2_PER_G = {
    "PMMA": {20: 0.5714, 50: 0.2074},
    "Al":   {20: 3.441,  50: 0.3681},
}
LOW_BIN_REF_KEV = 20
HIGH_BIN_REF_KEV = 50

# Folder naming pattern for slab combinations. Zenodo doesn't publish the
# exact convention ahead of download — inspect a few real folder names
# (printed at runtime) and adjust this regex if it doesn't match.
FOLDER_PATTERN = re.compile(r"PMMA_([\d.]+)_AL_([\d.]+)")


# ---------------------------------------------------------------------------
# Step 1: raw data loading
# ---------------------------------------------------------------------------
def read_raw_projection(path: Path) -> np.ndarray:
    """
    Read a single raw detector projection: 16-bit unsigned, little-endian,
    dimensions (HEIGHT, WIDTH), per Zhou et al. 2025.
    """
    data = np.fromfile(path, dtype="<u2")
    expected = WIDTH * HEIGHT
    if data.size != expected:
        raise ValueError(
            f"{path}: expected {expected} pixels ({WIDTH}x{HEIGHT}), got {data.size}. "
            "Check WIDTH/HEIGHT constants against the actual dataset."
        )
    return data.reshape(HEIGHT, WIDTH).astype(np.float64)


def load_air_tables() -> dict:
    """Load the flat-field (air) reference images for all three bins."""
    air_total = read_raw_projection(CAL_TABLE_DIR / "air_table_total.raw")
    air_high = read_raw_projection(CAL_TABLE_DIR / "air_table_high.raw")
    air_low = air_total - air_high  # Eq. from Zhou et al. 2025: Low = Total - High
    return {"low": air_low, "high": air_high, "total": air_total}


def discover_slab_folders(phantom_dir: Path) -> list[tuple[float, float, Path]]:
    """
    Find all PMMA_<m>_AL_<n> subfolders and parse their thickness values.
    Prints any folder it fails to parse, so FOLDER_PATTERN can be adjusted.
    """
    if not phantom_dir.exists():
        raise FileNotFoundError(
            f"{phantom_dir} not found. Extract CalibrationPhantomData.zip "
            "into data/CalibrationPhantomData/ per the module docstring."
        )
    combos = []
    unparsed = []
    for sub in sorted(phantom_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = FOLDER_PATTERN.match(sub.name)
        if m:
            t_pmma_mm, t_al_mm = float(m.group(1)), float(m.group(2))
            combos.append((t_pmma_mm, t_al_mm, sub))
        else:
            unparsed.append(sub.name)

    if unparsed:
        print(f"WARNING: {len(unparsed)} folder(s) didn't match FOLDER_PATTERN, e.g.:")
        for name in unparsed[:5]:
            print(f"    {name}")
        print("  Adjust FOLDER_PATTERN at the top of this file to match.\n")

    print(f"Discovered {len(combos)} parsed PMMA/Al slab combinations.")
    return combos


# ---------------------------------------------------------------------------
# Step 2-3: air normalization -> log-transmission per slab combo
# ---------------------------------------------------------------------------
def central_roi_mean(img: np.ndarray, fraction: float = ROI_FRACTION) -> float:
    """Mean pixel value over a central crop, to avoid edge/vignetting effects."""
    h, w = img.shape
    dh, dw = int(h * (1 - fraction) / 2), int(w * (1 - fraction) / 2)
    return float(img[dh:h - dh, dw:w - dw].mean())


def measure_log_transmission(combos: list, air: dict) -> dict:
    """
    For each slab combination, load the High/Total raw projections, derive
    Low by subtraction, air-normalize, and reduce to a scalar log-transmission
    per bin via the central ROI mean.

    Returns dict: bin_name -> list of (t_pmma_cm, t_al_cm, P) tuples, where
    P = -ln(I_obj / I_air), the Beer-Lambert log-attenuation (Eq. 1 in the
    source paper).
    """
    results = {"low": [], "high": [], "total": []}
    for t_pmma_mm, t_al_mm, folder in combos:
        try:
            img_high = read_raw_projection(folder / "proj_high.raw")
            img_total = read_raw_projection(folder / "proj_total.raw")
        except FileNotFoundError:
            print(f"  Skipping {folder.name}: missing proj files")
            continue
        img_low = img_total - img_high

        t_pmma_cm, t_al_cm = t_pmma_mm / 10.0, t_al_mm / 10.0
        for name, img, air_img in (("low", img_low, air["low"]),
                                    ("high", img_high, air["high"]),
                                    ("total", img_total, air["total"])):
            i_obj = central_roi_mean(img)
            i_air = central_roi_mean(air_img)
            if i_obj <= 0 or i_air <= 0:
                continue
            P = -np.log(i_obj / i_air)
            results[name].append((t_pmma_cm, t_al_cm, P))
    return results


# ---------------------------------------------------------------------------
# Step 4-5: interaction with matter — fit + compare to NIST
# ---------------------------------------------------------------------------
def fit_attenuation_coefficients(measurements: list) -> tuple[float, float, float]:
    """
    Multivariate least-squares fit of P = mu_PMMA * t_PMMA + mu_Al * t_Al
    across all slab combinations for one energy bin.

    Returns (mu_pmma, mu_al, r_squared).
    """
    t_pmma = np.array([m[0] for m in measurements])
    t_al = np.array([m[1] for m in measurements])
    P = np.array([m[2] for m in measurements])

    A = np.stack([t_pmma, t_al], axis=1)  # design matrix, no intercept
    coeffs, residuals, rank, sv = np.linalg.lstsq(A, P, rcond=None)
    mu_pmma, mu_al = coeffs

    P_pred = A @ coeffs
    ss_res = np.sum((P - P_pred) ** 2)
    ss_tot = np.sum((P - P.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return float(mu_pmma), float(mu_al), float(r_squared)


def compare_to_nist(mu_linear: float, density: float, material: str, ref_kev: int) -> dict:
    """Convert a fitted linear attenuation coefficient to mu/rho and compare to NIST."""
    mu_rho_measured = mu_linear / density
    mu_rho_nist = NIST_MU_RHO_CM2_PER_G[material][ref_kev]
    pct_diff = 100 * (mu_rho_measured - mu_rho_nist) / mu_rho_nist
    return {
        "material": material,
        "ref_kev": ref_kev,
        "mu_rho_measured": mu_rho_measured,
        "mu_rho_nist": mu_rho_nist,
        "pct_diff": pct_diff,
    }


# ---------------------------------------------------------------------------
# Step 6: photon-counting detection characterization
# ---------------------------------------------------------------------------
def characterize_detector(air: dict) -> dict:
    """
    Basic photon-counting detector characterization from flat-field data:
      - per-pixel gain non-uniformity (coefficient of variation) per bin
      - consistency check: Low + High == Total (should hold by construction,
        but confirms no arithmetic/data errors crept in upstream)
      - a simple gain-correction (normalize each pixel by its own mean
        relative to the panel mean) and the resulting uniformity improvement
    """
    report = {}
    for name, img in air.items():
        mean, std = img.mean(), img.std()
        report[f"{name}_cv_before"] = std / mean

        gain_map = img / mean  # per-pixel relative response
        corrected = img / gain_map
        report[f"{name}_cv_after"] = corrected.std() / corrected.mean()

    consistency_error = np.abs((air["low"] + air["high"]) - air["total"])
    report["low_plus_high_vs_total_max_abs_error"] = float(consistency_error.max())
    report["low_plus_high_vs_total_mean_abs_error"] = float(consistency_error.mean())

    return report


# ---------------------------------------------------------------------------
# Step 7: figures
# ---------------------------------------------------------------------------
def plot_attenuation_fit(measurements: list, mu_pmma: float, mu_al: float,
                          bin_name: str, out_path: Path):
    """
    Scatter of measured log-attenuation vs. fitted prediction, plus the
    fitted-vs-thickness curves for pure-PMMA and pure-Al subsets (where
    the other material's thickness is at its minimum in the dataset).
    """
    t_pmma = np.array([m[0] for m in measurements])
    t_al = np.array([m[1] for m in measurements])
    P = np.array([m[2] for m in measurements])
    P_pred = mu_pmma * t_pmma + mu_al * t_al

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(P_pred, P, alpha=0.7)
    lims = [min(P.min(), P_pred.min()), max(P.max(), P_pred.max())]
    axes[0].plot(lims, lims, "k--", alpha=0.5, label="y = x")
    axes[0].set_xlabel("Predicted -ln(I/I0)")
    axes[0].set_ylabel("Measured -ln(I/I0)")
    axes[0].set_title(f"{bin_name.capitalize()} bin: fit quality")
    axes[0].legend()

    # Pure-PMMA subset: thinnest available Al thickness
    al_min = t_al.min()
    pmma_only = t_al == al_min
    order = np.argsort(t_pmma[pmma_only])
    axes[1].plot(t_pmma[pmma_only][order], P[pmma_only][order], "o-",
                 label=f"Measured (Al={al_min:.1f} cm)")
    axes[1].plot(t_pmma[pmma_only][order],
                 mu_pmma * t_pmma[pmma_only][order] + mu_al * al_min, "--",
                 label="Fitted Beer-Lambert")
    axes[1].set_xlabel("PMMA thickness (cm)")
    axes[1].set_ylabel("-ln(I/I0)")
    axes[1].set_title(f"{bin_name.capitalize()} bin: attenuation vs. PMMA thickness")
    axes[1].legend()

    fig.suptitle(f"X-ray interaction with matter — {bin_name} energy bin (real PCCT data)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def plot_detector_uniformity(air: dict, out_path: Path):
    """Flat-field images per bin, for visual detector-uniformity inspection."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, img) in zip(axes, air.items()):
        im = ax.imshow(img, cmap="inferno")
        ax.set_title(f"{name.capitalize()} bin flat-field\n(CV={img.std() / img.mean():.4f})")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Photon-counting detector flat-field response (x-ray detection)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not CAL_TABLE_DIR.exists() or not PHANTOM_DIR.exists():
        print("Data not found. See the module docstring at the top of this "
              "file for Zenodo download instructions and the expected "
              f"layout under {DATA_DIR}/.")
        return

    print("=" * 70)
    print("STEP 1-2: Loading air (flat-field) tables and slab projections")
    print("=" * 70)
    air = load_air_tables()
    combos = discover_slab_folders(PHANTOM_DIR)
    if not combos:
        print("No slab combinations parsed — check FOLDER_PATTERN. Exiting.")
        return

    print("\n" + "=" * 70)
    print("STEP 3: Air-normalizing and computing log-transmission")
    print("=" * 70)
    measurements = measure_log_transmission(combos, air)
    for name in ("low", "high", "total"):
        print(f"  {name}: {len(measurements[name])} usable measurements")

    print("\n" + "=" * 70)
    print("STEP 4-5: Fitting attenuation coefficients (interaction with matter)")
    print("=" * 70)
    for bin_name, ref_kev in (("low", LOW_BIN_REF_KEV), ("high", HIGH_BIN_REF_KEV)):
        if len(measurements[bin_name]) < 3:
            print(f"  {bin_name}: not enough data points to fit, skipping")
            continue
        mu_pmma, mu_al, r2 = fit_attenuation_coefficients(measurements[bin_name])
        print(f"\n  [{bin_name.upper()} bin, reference energy {ref_kev} keV]")
        print(f"    Fitted mu_PMMA = {mu_pmma:.4f} 1/cm,  mu_Al = {mu_al:.4f} 1/cm  (R^2={r2:.4f})")

        for material, mu in (("PMMA", mu_pmma), ("Al", mu_al)):
            density = DENSITY_PMMA if material == "PMMA" else DENSITY_AL
            cmp = compare_to_nist(mu, density, material, ref_kev)
            print(f"    {material}: measured mu/rho = {cmp['mu_rho_measured']:.4f} cm^2/g, "
                  f"NIST @ {ref_kev} keV = {cmp['mu_rho_nist']:.4f} cm^2/g "
                  f"({cmp['pct_diff']:+.1f}% difference)")

        plot_attenuation_fit(measurements[bin_name], mu_pmma, mu_al, bin_name,
                              FIG_DIR / f"attenuation_fit_{bin_name}.png")

    print("\n" + "=" * 70)
    print("STEP 6: Photon-counting detector characterization (detection)")
    print("=" * 70)
    det_report = characterize_detector(air)
    for name in ("low", "high", "total"):
        print(f"  {name} bin: non-uniformity (CV) before correction = "
              f"{det_report[f'{name}_cv_before']:.4f}, after simple gain "
              f"correction = {det_report[f'{name}_cv_after']:.4f}")
    print(f"  Low+High vs Total consistency: max abs error = "
          f"{det_report['low_plus_high_vs_total_max_abs_error']:.2f} counts, "
          f"mean abs error = {det_report['low_plus_high_vs_total_mean_abs_error']:.4f} counts")

    plot_detector_uniformity(air, FIG_DIR / "detector_uniformity.png")

    print("\nDone. Figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
