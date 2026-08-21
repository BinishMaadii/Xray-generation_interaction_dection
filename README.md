# X-ray Interaction & Photon-Counting CT Analysis Pipeline

A single-file Python pipeline for processing raw spectral projection data from a real photon-counting CT (PCCT) detector. The code extracts material-specific X-ray attenuation physics and characterizes photon-counting detector properties using real calibration phantom scans.

---

## Technical Context

This repository demonstrates practical physics and data analysis for spectral CT applications:
* **X-ray Interaction with Matter:** Measures linear ($\mu$) and mass attenuation coefficients ($\mu/\rho$) for PMMA and Aluminum across energy bins and validates results against NIST XCOM reference data.
* **Photon-Counting Detection:** Analyzes per-pixel gain non-uniformity, evaluates bin conservation ($\text{Low} + \text{High} = \text{Total}$), and performs flat-field calibrations.

---

## Dataset & Sources

The pipeline processes public real-world experimental data (CC BY 4.0):
* **Paper:** Zhou, E. et al. *"A cone-beam photon-counting CT dataset for spectral image reconstruction and deep learning."* *Scientific Data* 12, 1955 (2025). [DOI: 10.1038/s41597-025-06246-4](https://doi.org/10.1038/s41597-025-06246-4)
* **Data Repository:** Zenodo bundle `"calibration table&sample1"`. [DOI: 10.5281/zenodo.15738313](https://doi.org/10.5281/zenodo.15738313)

---

## Requirements

* Python 3.9+
* Required packages:
  ```bash
  pip install numpy matplotlib

-------

├── main.py
├── data/
│   ├── CalibrationTable/
│   │   ├── air_table_low.raw
│   │   ├── air_table_high.raw
│   │   └── air_table_total.raw
│   └── CalibrationPhantomData/
│       └── PMMA_AL_slabs/
│           ├── AcqPara.mat
│           ├── PMMA_<m>_AL_<n>/
│           │   ├── proj_high.raw
│           │   └── proj_total.raw
│           └── ... (56 slab combination subfolders)
└── figures/   (generated automatically at runtime)
