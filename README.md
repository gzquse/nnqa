# Neural-Native Quantum Arithmetic (NNQA)

A toolkit for mapping trained neural networks directly to quantum circuits using quantum arithmetic primitives.

---

## Overview

This project implements a workflow to:

1. **Train** classical neural networks on polynomial/function approximation tasks
2. **Extract** learned weights and map them to quantum circuit angles
3. **Build** quantum circuits using quantum arithmetic primitives:
   - Weighted Sum: `w*x0 + (1-w)*x1`
   - Multiplication: `x0 * x1`
4. **Evaluate** and verify that quantum circuits reproduce NN behavior

## Project Structure

```
neuro_synthesis/
|-- main.sh                 # Main execution script
|-- config.sh               # Configuration file
|-- requirements.txt        # Python dependencies
|-- README.md               # This file
|-- nn_to_quantum.py        # NN to Quantum demo script
|-- pl_recovery_metrics.py  # Polynomial recovery metrics with error bars
|-- pl_sum.py               # IBM job result analysis
|-- theo_sum.py             # Theoretical expressivity bounds plots
|
|-- nnqa/                   # Main package
|   |-- __init__.py         # Package exports
|   |-- models.py           # Neural network architectures
|   |-- quantum_circuits.py # Quantum circuit builders
|   |-- mapper.py           # NN to Quantum translation
|   |-- trainer.py          # Training utilities
|   |-- evaluator.py        # Evaluation and benchmarking
|   |-- main.py             # Training entry point
|   |-- eval.py             # Evaluation entry point
|
|-- qc/                     # Pre-built quantum circuits
|   |-- circuits/           # QPY circuit files
|   |-- experiments/        # Experiment data (H5 files)
|
|-- results/                # Output directory
|-- toolbox/                # Utility modules
|
|-- cloud_job/              # IBM Quantum Cloud submission
|   |-- submit_nnqa_ibmq.py   # Submit jobs to IBM
|   |-- retrieve_nnqa_ibmq.py # Retrieve job results
|   |-- plot_nnqa_accuracy.py # Plot accuracy with error bars
|   |-- batch_nnqa_ibmq.sh    # Batch job script
|   |-- env.template          # Credentials template
|   |-- out/                  # Job outputs (jobs/, meas/, post/, plots/)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Training with Default Configuration

```bash
./main.sh
```

### 3. Run with Custom Configuration

Edit `config.sh` and then run:

```bash
./main.sh
```

Or use command-line arguments directly:

```bash
python -m nnqa.main --degree 3 --epochs 200 --target-type polynomial
```

## Configuration Options

Edit `config.sh` to customize:

### Model Parameters
- `MODEL_TYPE`: `polynomial` or `deep`
- `DEGREE`: Polynomial degree (default: 3)
- `HIDDEN_DIMS`: Hidden layer dimensions for deep model

### Target Function
- `TARGET_TYPE`: `polynomial`, `sin`, `cos`, `gaussian`
- `TARGET_COEFFS`: Polynomial coefficients

### Training Parameters
- `EPOCHS`: Number of training epochs
- `LR`: Learning rate
- `SAMPLES`: Number of training samples

### Quantum Parameters
- `SHOTS`: Quantum simulation shots

## Examples

### Train Polynomial NN

```bash
python -m nnqa.main \
    --model-type polynomial \
    --degree 3 \
    --target-type polynomial \
    --target-coeffs 0.1 0.3 -0.2 0.5 \
    --epochs 200
```

### Train Deep NN for Trigonometric Function

```bash
python -m nnqa.main \
    --model-type deep \
    --hidden-dims 32 32 \
    --target-type sin \
    --epochs 300
```

### Evaluate Trained Model

```bash
python -m nnqa.eval \
    --model-path results/run_default/final_model.pt \
    --shots 8192
```

### Comparison benchmarks (sine + VQA + polynomial domain sweep)

`research/scripts/nnqa_comparison_benchmarks.py` runs, in order: (1) deep `DeepPolynomialNN` on a fixed sine target with classical vs mapped-quantum RMSE vs the analytic sine; (2) the same sine target with a two-qubit variational ansatz (statevector MSE training, finite-shot evaluation) vs mapped NNQA at equal shots per test ``x``; (3) ``PolynomialNN`` plus ``quantum_polynomial_direct`` vs ground truth for degrees 1..N on test grids ``[-0.5,0.5]`` and ``[-0.9,0.9]``. Full definitions and hyperparameters are in the module docstring.

```bash
python research/scripts/nnqa_comparison_benchmarks.py --output-dir results/nnqa_comparison_benchmarks
python research/scripts/nnqa_comparison_benchmarks.py --quick
```

Writes ``comparison_benchmark_report.json`` under ``--output-dir``.

## Quantum Arithmetic Primitives

Quantum Arithmetic Primitives:

### Weighted Sum (Tag 0)
Computes `y = w*x0 + (1-w)*x1`

```
      +-----------+ barrier               +----------+     +-----------+
q0_0: | Ry(th[0]) |---||------*-----------| Ry(a/2)  |--X--| Ry(-a/2)  |--M--
      +-----------+   ||      |           +----------+  |  +-----------+     
                      ||  +---+---+                     |                    
q0_1: | Ry(th[1]) |---||--| Rz(pi/2)|-X-----------------*--------------------
      +-----------+   ||  +---------+ |                                      
```

### Multiplication (Tag 1)
Computes `y = x0 * x1`

```
      +-----------+ barrier               
q0_0: | Ry(th[0]) |---||------*-----------
      +-----------+   ||      |           
                      ||  +---+---+       
q0_1: | Ry(th[1]) |---||--| Rz(pi/2)|-X---
      +-----------+   ||  +---------+     
```

## Output Files

After training, find results in `results/run_<identifier>/`:

- `config.json`: Run configuration
- `training_history.json`: Loss curves and metrics
- `training_metadata.json`: Training summary
- `final_model.pt`: Trained model weights
- `benchmark_results.json`: NN vs Quantum comparison
- `evaluation_results.json`: Detailed evaluation data
- `checkpoint_epoch_*.pt`: Intermediate checkpoints

## API Usage

```python
from nnqa import create_trainer, Evaluator

# Create and train model
trainer = create_trainer(
    model_type='polynomial',
    degree=3,
    target_type='polynomial',
    coefficients=[0.1, 0.3, -0.2, 0.5],
    output_dir='my_results'
)
history = trainer.train(epochs=200)

# Evaluate with quantum simulation
evaluator = Evaluator(trainer.model, shots=4096)
results = evaluator.run_benchmark()
```

## Polynomial Recovery Metrics

The `pl_recovery_metrics.py` script generates publication-quality plots showing quantum polynomial recovery with error bars. It uses the recovery methodology from `ehands_2q.py`:

- **Theoretical EV (tEV)**: Computed from the polynomial formula
- **Measured EV (mEV)**: `1 - 2*mprob` where `mprob = n1/(n0+n1)`
- **Recovery Error**: `delta = tEV - mEV`
- **Status Thresholds**: PASS (`|delta| < 0.03`), POOR (`|delta| < 0.10`), FAIL (`|delta| >= 0.10`)

### Usage

```bash
# Basic usage with default polynomial F(x) = 0.1 + 0.3x - 0.1x^2 + 0.2x^3
python pl_recovery_metrics.py

# Custom polynomial coefficients [a0, a1, a2, ...]
python pl_recovery_metrics.py --polynomial 0.0 0.5 0.0 0.25

# More trials and points for smoother error bars
python pl_recovery_metrics.py --n-points 20 --n-trials 20 --shots 4096

# Include operation comparison plot (weighted sum and multiplication)
python pl_recovery_metrics.py --operation-comparison

# Save as PNG to results directory
python pl_recovery_metrics.py --format png --output-dir ./results
```

### Output Plots

1. **Polynomial Recovery Plot** (`polynomial_recovery.pdf`)
   - Left panel: Theoretical curve vs quantum measured points with error bars
   - Right panel: Recovery error analysis with colored threshold regions (PASS/POOR/FAIL)

2. **Operation Comparison Plot** (`operation_comparison.pdf`) - optional
   - Compares weighted sum and multiplication quantum operations
   - Shows theoretical vs measured values with error bars

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | `.` | Output directory for figures |
| `--format` | `pdf` | Output format (pdf, png, svg) |
| `--n-points` | `15` | Number of test points |
| `--n-trials` | `10` | Number of trials per point for statistics |
| `--shots` | `2048` | Shots per quantum circuit execution |
| `--polynomial` | `0.1 0.3 -0.1 0.2` | Polynomial coefficients |
| `--operation-comparison` | False | Generate operation comparison plot |

## IBM Quantum Cloud Submission

The `cloud_job/` directory provides scripts for submitting NNQA experiments to IBM Quantum hardware.

### Available Backends (Heron Processors)

| Backend | Qubits | Processor | Notes |
|---------|--------|-----------|-------|
| `ibm_boston` | 156 | Heron r3 | Recommended (latest) |
| `ibm_pittsburgh` | 156 | Heron r3 | Recommended (latest) |
| `ibm_fez` | 156 | Heron r2 | |
| `ibm_marrakesh` | 156 | Heron r2 | |
| `ibm_kingston` | 156 | Heron r2 | |
| `ibm_torino` | 133 | Heron r1 | |
| `ibm_miami` | 120 | Nighthawk r1 | |

List available backends:
```bash
cd cloud_job
python list_backends.py
```

### Setup Credentials

1. Copy the template and add your credentials:

```bash
cp cloud_job/env.template .env
```

2. Edit `.env` with your IBM Quantum credentials:

```bash
# .env file contents:
IBM_QUANTUM_TOKEN=your_api_token_here
QISKIT_IBM_CHANNEL=ibm_cloud
QISKIT_IBM_INSTANCE=crn:v1:bluemix:public:quantum-computing:us-east:a/your_account:your_instance::
```

**IMPORTANT**: Never commit `.env` to git! It's already in `.gitignore`.

### Quick Start

```bash
# Navigate to cloud_job directory
cd cloud_job
```

### Submit a Single Job

```bash
# Test locally first (no -E flag) - validates circuit without submitting
python submit_nnqa_ibmq.py --backend ibm_fez --numSample 5

# Submit to IBM cloud (with -E flag)
python submit_nnqa_ibmq.py -E \
    --backend ibm_fez \
    --testType polynomial \
    --polynomial 0.1 0.3 -0.1 0.2 \
    --numSample 5 \
    --numShot 4096 \
    --expName my_test_001

# Submit with custom experiment name
python submit_nnqa_ibmq.py -E \
    --backend ibm_marrakesh \
    --testType weighted_sum \
    --numSample 10 \
    --numShot 8192 \
    --expName ws_marrakesh_001
```

### Retrieve Results

```bash
# Wait for job completion and retrieve results
python retrieve_nnqa_ibmq.py --basePath cloud_job/out --expName my_test_001

# With custom timeout (default 3600 seconds)
python retrieve_nnqa_ibmq.py --basePath cloud_job/out --expName my_test_001 --timeout 7200
```

### Plot Accuracy with Error Bars

```bash
# Single experiment
python plot_nnqa_accuracy.py --basePath cloud_job/out --expName my_test_001

# Multiple experiments comparison
python plot_nnqa_accuracy.py --basePath cloud_job/out \
    --expName my_test_001 ws_marrakesh_001 \
    -p a b

# Save as PNG instead of PDF
python plot_nnqa_accuracy.py --basePath cloud_job/out --expName my_test_001 --format png
```

### Batch Operations

Use the batch script for multiple experiments:

```bash
cd cloud_job

# Submit all jobs (ibm_fez + ibm_marrakesh, all test types)
./batch_nnqa_ibmq.sh submit

# Retrieve all results
./batch_nnqa_ibmq.sh retrieve

# Generate all plots
./batch_nnqa_ibmq.sh plot

# Check status summary
./batch_nnqa_ibmq.sh status
```

### Output Structure

```
cloud_job/out/
|-- jobs/        # Submitted job H5 files (before results)
|-- meas/        # Measurement results H5 files
|-- post/        # Postprocessed results with metrics
|-- plots/       # Generated accuracy plots (PDF/PNG)
```

### Supported Test Types

| Type | Description | Formula |
|------|-------------|---------|
| `polynomial` | Polynomial evaluation | `F(x) = a0 + a1*x + a2*x^2 + ...` |
| `weighted_sum` | Linear combination | `y = w*x0 + (1-w)*x1` |
| `multiplication` | Product | `y = x0 * x1` |

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `-E` | False | Execute/submit job (required for actual submission) |
| `-b, --backend` | `ibm_fez` | IBM backend name |
| `-t, --testType` | `polynomial` | Test type |
| `-p, --polynomial` | `0.1 0.3 -0.1 0.2` | Polynomial coefficients |
| `-i, --numSample` | 10 | Number of test samples |
| `-n, --numShot` | 4096 | Shots per circuit |
| `--expName` | auto | Experiment name |
| `--useRC` | True | Enable randomized compilation |
| `--useDD` | False | Enable dynamical decoupling |

### Error Mitigation Options

| Option | Flag | Description |
|--------|------|-------------|
| Randomized Compilation | `--useRC` | Twirling for error averaging (recommended, enabled by default) |
| Dynamical Decoupling | `--useDD` | XX sequence for idle qubit decoherence |

### Example Workflow

```bash
# 1. Setup
cp cloud_job/env.template .env
# Edit .env with your credentials

# 2. Submit job
cd cloud_job
python submit_nnqa_ibmq.py -E --backend ibm_fez --numSample 5 --expName demo_001

# 3. Check job status on IBM Quantum Dashboard
# https://quantum.cloud.ibm.com/jobs

# 4. Retrieve results (after job completes)
python retrieve_nnqa_ibmq.py --basePath cloud_job/out --expName demo_001

# 5. Plot accuracy with error bars
python plot_nnqa_accuracy.py --basePath cloud_job/out --expName demo_001

# 6. View results
ls cloud_job/out/plots/
```

## Citation

If you use this code, please cite:

```bibtex
@software{neuro_synthesis,
  title = {Neural-Native Quantum Arithmetic},
  year = {2026}
}
```

## License

MIT License
