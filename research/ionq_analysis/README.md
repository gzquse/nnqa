# IonQ Stress Test Analysis

This directory contains scripts, data, and figures for analyzing IonQ Forte-1 stress test results for polynomial degrees 1-35.

## Directory Structure

```
ionq_analysis/
├── README.md              # This file
├── scripts/               # Analysis scripts
│   ├── plot_ionq_stress_test.py      # Generate figures
│   └── retrieve_ionq_stress_test.py  # Retrieve job results
├── data/                  # HDF5 result files (*.h5)
├── figures/               # Generated plots (PDF/PNG)
└── results/               # Raw job manifests and metadata
```

## Quick Start

### 1. Retrieve Results

After submitting jobs to IonQ Forte-1, retrieve the completed results:

```bash
cd scripts
python retrieve_ionq_stress_test.py
```

This will:
- Read job IDs from `results/job_manifest.json`
- Retrieve completed jobs from IonQ
- Process and save results to `data/*.h5`

### 2. Generate Plots

Generate publication-quality figures:

```bash
cd scripts
python plot_ionq_stress_test.py --all
```

This creates:
- `figures/fig1_scaling_analysis.pdf` - 4-panel scaling analysis (RMSE, Correlation, Pass Rate, Circuit Resources)
- `figures/fig2_recovery_grid.pdf` - Recovery grid showing theoretical vs measured for each degree

## Scripts

### `retrieve_ionq_stress_test.py`

Retrieves and processes IonQ stress test job results.

**Requirements:**
- IonQ API key set in environment or `research/ionq/ionq_config.py`
- Completed jobs listed in `results/job_manifest.json`

**Output:**
- HDF5 files in `data/` directory: `stress_deg{degree}_trial{trial:02d}.h5`

### `plot_ionq_stress_test.py`

Generates publication-quality figures from stress test results.

**Options:**
- `--all` - Generate all figures (default if no specific figure requested)
- `--fig1` - Generate Figure 1: Scaling analysis
- `--fig2` - Generate Figure 2: Recovery grid
- `--input DIR` - Input directory with H5 files (default: `../data`)
- `--output DIR` - Output directory for figures (default: `../figures`)
- `--no-display` - Don't display plots (default: True)

**Output:**
- PDF and PNG versions of all figures in `figures/` directory

## Data Format

### HDF5 Result Files

Each result file (`stress_deg{d}_trial{t}.h5`) contains:

**Data:**
- `x_values` - Input x values
- `theoretical` - Theoretical polynomial values
- `classical_pred` - Classical predictions
- `measured` - Quantum measurement results
- `measured_err` - Measurement error bars
- `true_coefficients` - True polynomial coefficients
- `learned_coefficients` - Neural network learned coefficients

**Metadata:**
- `degree` - Polynomial degree
- `trial` - Trial number
- `polynomial_name` - Name (e.g., "Icosic")
- `backend` - Backend name
- `shots` - Number of shots per circuit
- `metrics` - Computed metrics (RMSE, correlation, pass rate)
- `circuit_info` - Circuit resource information

## Tested Degrees

The stress test evaluates polynomial degrees:
- **Low degrees**: 1, 5, 10, 15
- **High degrees**: 20, 25, 30, 35

**Total**: 8 degrees, 40 circuits, ~40,960 shots

## Results Summary

| Degree | RMSE   | Correlation | Pass Rate | Qubits | Depth |
|--------|--------|-------------|-----------|--------|-------|
| 1      | 0.008  | 0.9999      | 100%      | 1      | 2     |
| 5      | 0.017  | 0.9997      | 100%      | 5      | 10    |
| 10     | 0.004  | 0.9997      | 100%      | 10     | 20    |
| 15     | 0.002  | 0.9994      | 100%      | 15     | 30    |
| 20     | 0.001  | 1.0000      | 100%      | 20     | 40    |
| 25     | 0.001  | 0.9998      | 100%      | 25     | 50    |
| 30     | 0.001  | 0.9999      | 100%      | 30     | 60    |
| 35     | 0.005  | 0.948       | 100%      | 35     | 70    |

**Key Findings:**
- Excellent accuracy (correlation ≥ 0.999) for degrees 1-30
- Performance boundary identified at degree 35 (correlation drops to 0.948)
- Linear resource scaling: O(d) qubits and gates
- All degrees achieve 100% pass rate

## Dependencies

- Python 3.8+
- NumPy
- Matplotlib
- Qiskit IonQ (`qiskit-ionq`)
- Custom modules from `research/ionq/`:
  - `ionq_config.py` - Configuration and polynomial definitions
  - `submit_ionq_native.py` - Neural network and circuit building functions
  - `toolbox.Util_H5io4` - HDF5 I/O utilities
