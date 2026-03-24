# IonQ Native Polynomial Experiments

This folder contains scripts for running native quantum polynomial approximation experiments on IonQ backends.

## Overview

The native polynomial approach uses quantum arithmetic primitives (multiplication and weighted sum) to compute polynomial powers directly on the quantum circuit, rather than pre-computing classically.

## Files

- `ionq_config.py` - Configuration for IonQ backends and polynomial definitions
- `submit_ionq_native.py` - Main script to run experiments on IonQ (local simulator or cloud)
- `plot_ionq_results.py` - Generate publication-quality figures from results
- `results/` - Directory for H5 result files
- `figures/` - Directory for generated plots

## Quick Start

### 1. Run experiments on IonQ local simulator

```bash
# Run all degrees (1-6) with 3 trials each
python submit_ionq_native.py --backend simulator --trials 3 --execute

# Run specific degrees
python submit_ionq_native.py --backend simulator --degrees 1,2,3 --trials 3 --execute

# Dry run to check circuits
python submit_ionq_native.py --backend simulator --degrees 1 --trials 1
```

### 2. Generate plots

```bash
# Generate all figures
python plot_ionq_results.py --all

# Generate specific figures
python plot_ionq_results.py --fig1  # Degree scaling
python plot_ionq_results.py --fig2  # Recovery grid
python plot_ionq_results.py --fig3  # Error distribution
python plot_ionq_results.py --fig4  # Correlation summary
python plot_ionq_results.py --tables  # LaTeX tables
```

## Quantum Circuit Design

For a polynomial F(x) = a0 + a1*x + a2*x^2 + ... + ad*x^d:

1. **Qubit 0**: Encodes x via Ry(arccos(x)), giving <Z_0> = x
2. **Qubit k (k >= 1)**: Stores x^(k+1) computed via quantum multiplication
   - Encode x via Ry
   - Apply RZ(pi/2)
   - Apply CNOT from qubit k-1 to qubit k
   - Result: <Z_k> = x^(k+1)

3. **Measurement**: All qubits are measured, and the polynomial is reconstructed:
   F(x) = a0 + a1*<Z_0> + a2*<Z_1> + ... + ad*<Z_{d-1}>

## Configuration

Edit `ionq_config.py` to modify:
- Polynomial coefficients
- Number of trials
- Number of shots
- Number of sample points
- Training hyperparameters

## IonQ Backends

- `simulator` - IonQ ideal simulator (local)
- `harmony` - IonQ Harmony 11-qubit system (cloud)
- `aria-1` - IonQ Aria 25-qubit system (cloud)

## Output Format

Results are saved in HDF5 format with:
- `x_values` - Input x values
- `theoretical` - Expected polynomial values
- `classical_pred` - Neural network predictions
- `measured` - Quantum circuit measurements
- `measured_err` - Measurement uncertainties

Metadata includes:
- Polynomial degree and name
- Backend and provider info
- Metrics (RMSE, correlation, pass rate)
- Configuration parameters
