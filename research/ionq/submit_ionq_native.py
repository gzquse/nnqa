#!/usr/bin/env python3
"""
Submit Native Polynomial Jobs to IonQ
=====================================

Runs native quantum polynomial approximation experiments on IonQ backends.
Uses quantum multiplication primitives to compute polynomial powers natively.

Based on reference/submit_multXY_job.py structure adapted for native polynomial.

Usage:
    # Run on IonQ local simulator (ideal):
    python submit_ionq_native.py --backend simulator --trials 3 --execute
    
    # Run all degrees with specific config:
    python submit_ionq_native.py --backend simulator --degrees 1,2,3 --trials 3 --shots 8192 --execute
    
    # Dry run to check circuits:
    python submit_ionq_native.py --backend simulator --degrees 1 --trials 1

Adapted from a public qcrank-style quantum-control reference pattern.
"""

import sys
import os
import hashlib
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pprint import pprint
from time import time, localtime
from datetime import datetime
import argparse

# Add parent paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

# Try to import IonQ provider (optional)
try:
    from qiskit_ionq import IonQProvider
    IONQ_AVAILABLE = True
except ImportError:
    IONQ_AVAILABLE = False
    print("Warning: qiskit-ionq not available, using Qiskit Aer only")

from toolbox.Util_IOfunc import dateT2Str
from toolbox.Util_H5io4 import write4_data_hdf5, read4_data_hdf5

from ionq_config import (
    get_ionq_api_key,
    IONQ_BACKENDS,
    POLYNOMIALS,
    LOCAL_SIMULATOR_CONFIG,
    CLOUD_CONFIG,
    evaluate_polynomial,
    validate_polynomials,
)


# ==============================================================================
# NEURAL NETWORK FOR POLYNOMIAL APPROXIMATION
# ==============================================================================

class PolynomialNN(nn.Module):
    """Neural network that learns polynomial coefficients."""
    
    def __init__(self, degree=3):
        super().__init__()
        self.degree = degree
        self.coefficients = nn.Parameter(torch.zeros(degree + 1))
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.squeeze(-1)
        result = torch.zeros_like(x)
        x_power = torch.ones_like(x)
        for i in range(self.degree + 1):
            result = result + self.coefficients[i] * x_power
            x_power = x_power * x
        return result
    
    def get_coefficients(self):
        return self.coefficients.detach().cpu().numpy()


def train_nn(model, target_func, config, verbose=False):
    """Train neural network on target function."""
    optimizer = optim.Adam(model.parameters(), lr=config['train_lr'])
    loss_fn = nn.MSELoss()
    
    X_train = torch.linspace(-0.95, 0.95, config['train_samples'])
    y_train = torch.tensor([target_func(x.item()) for x in X_train], dtype=torch.float32)
    
    for epoch in range(config['train_epochs']):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()
        
        if verbose and epoch % 50 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")
    
    return loss.item()


# ==============================================================================
# QUANTUM CIRCUIT BUILDING (NATIVE POLYNOMIAL)
# ==============================================================================

def data_to_angle(x):
    """Convert x in [-1,1] to rotation angle. After Ry(theta), <Z> = cos(theta) = x."""
    x = np.clip(x, -1 + 1e-7, 1 - 1e-7)
    return np.arccos(x)


def build_native_polynomial_circuit(x, coefficients, add_barriers=True):
    """
    Build native quantum circuit for polynomial evaluation using quantum arithmetic.
    
    Uses quantum multiplication to compute powers natively on the quantum circuit.
    For degree d polynomial, uses d qubits minimum (one per power).
    
    Strategy:
    - Qubit 0: stores x
    - Qubit 1: stores x^2 (computed via quantum multiplication: x * x)
    - Qubit 2: stores x^3 (computed via quantum multiplication: x^2 * x)
    - Qubit k: stores x^(k+1) (computed via quantum multiplication)
    
    After circuit execution, each qubit k encodes x^(k+1) in its expectation value <Z_k>.
    Terms are then combined using the coefficients.
    
    Parameters:
        x: Input value in [-1, 1]
        coefficients: Polynomial coefficients [a0, a1, a2, ..., ad]
        add_barriers: Whether to add barriers between operations (for clarity)
    
    Returns:
        qc: QuantumCircuit implementing the polynomial
        theoretical: Expected polynomial value
    """
    degree = len(coefficients) - 1
    coeffs = np.array(coefficients)
    
    # Theoretical value for comparison
    theoretical = sum(coeffs[i] * (x ** i) for i in range(len(coeffs)))
    theoretical_clipped = np.clip(theoretical, -1 + 1e-6, 1 - 1e-6)
    
    # Number of qubits: one per power x, x^2, ..., x^degree
    n_qubits = max(1, degree)
    
    qr = QuantumRegister(n_qubits, 'q')
    cr = ClassicalRegister(n_qubits, 'c')
    qc = QuantumCircuit(qr, cr)
    
    # ========================================================================
    # STEP 1: Encode input x on qubit 0
    # ========================================================================
    if degree >= 1:
        qc.ry(data_to_angle(x), 0)
        # After this: <Z_0> = x
    
    # ========================================================================
    # STEP 2: Compute powers using quantum multiplication
    # ========================================================================
    # Quantum multiplication circuit:
    # 1. Encode x on new qubit k
    # 2. Apply RZ(pi/2) on qubit k
    # 3. Apply CNOT from qubit (k-1) to qubit k
    # Result: <Z_k> = x^(k-1) * x = x^k
    
    # x^2 = x * x
    if degree >= 2:
        qc.ry(data_to_angle(x), 1)
        if add_barriers:
            qc.barrier()
        qc.rz(np.pi/2, 1)
        qc.cx(0, 1)
        # Now: <Z_1> = x^2
    
    # x^3 = x^2 * x
    if degree >= 3:
        qc.ry(data_to_angle(x), 2)
        if add_barriers:
            qc.barrier()
        qc.rz(np.pi/2, 2)
        qc.cx(1, 2)
        # Now: <Z_2> = x^3
    
    # x^4 = x^3 * x
    if degree >= 4:
        qc.ry(data_to_angle(x), 3)
        if add_barriers:
            qc.barrier()
        qc.rz(np.pi/2, 3)
        qc.cx(2, 3)
        # Now: <Z_3> = x^4
    
    # Continue for higher degrees
    for k in range(4, degree):
        if k < n_qubits:
            qc.ry(data_to_angle(x), k)
            if add_barriers:
                qc.barrier()
            qc.rz(np.pi/2, k)
            qc.cx(k-1, k)
            # Now: <Z_k> = x^(k+1)
    
    # ========================================================================
    # STEP 3: Measure all qubits
    # ========================================================================
    for k in range(n_qubits):
        qc.measure(k, k)
    
    return qc, theoretical_clipped


# ==============================================================================
# RESULT PROCESSING
# ==============================================================================

def extract_polynomial_from_counts(counts, coefficients, total_shots):
    """
    Extract polynomial value from measurement counts.
    
    For each qubit k, compute <Z_k> = (n_0 - n_1) / total_shots
    Then compute F(x) = a0 + a1*<Z_0> + a2*<Z_1> + ... + ad*<Z_{d-1}>
    where <Z_k> approximates x^(k+1)
    """
    degree = len(coefficients) - 1
    n_qubits = max(1, degree)
    
    # Initialize powers: powers[0] = 1 (constant), powers[k] = x^k
    powers = np.zeros(degree + 1)
    powers[0] = 1.0  # x^0 = 1
    
    if total_shots == 0:
        return 0.0, 1.0
    
    # Extract <Z_k> for each qubit k = 0, 1, ..., degree-1
    for k in range(n_qubits):
        n0_k = 0  # Count when qubit k is 0
        n1_k = 0  # Count when qubit k is 1
        
        for bitstring, count in counts.items():
            # Clean bitstring
            bitstring_clean = bitstring.replace(' ', '')
            bitstring_padded = bitstring_clean.zfill(n_qubits)
            
            if k < len(bitstring_padded):
                # Qiskit format: rightmost bit is qubit 0 (LSB)
                bit_idx = len(bitstring_padded) - 1 - k
                bit_k = int(bitstring_padded[bit_idx])
                if bit_k == 0:
                    n0_k += count
                else:
                    n1_k += count
            else:
                n0_k += count
        
        # <Z_k> = (n0 - n1) / shots
        # Qubit k stores x^(k+1)
        x_power = (n0_k - n1_k) / total_shots
        powers[k+1] = x_power
    
    # Compute polynomial
    poly_value = sum(coefficients[j] * powers[j] for j in range(len(coefficients)))
    
    # Estimate error from shot noise
    shot_err = 1.0 / np.sqrt(total_shots)
    poly_err = sum(abs(coefficients[j]) * shot_err for j in range(1, len(coefficients)))
    
    return poly_value, poly_err


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Submit native polynomial jobs to IonQ'
    )
    
    parser.add_argument('-v', '--verb', type=int, default=1,
                       help='Verbosity level')
    parser.add_argument('--basePath', default='.',
                       help='Base path for output')
    
    # Experiment configuration
    parser.add_argument('--degrees', default='1,2,3,4,5,6',
                       help='Comma-separated list of polynomial degrees')
    parser.add_argument('--trials', type=int, default=3,
                       help='Number of trials per degree')
    parser.add_argument('--numSample', type=int, default=25,
                       help='Number of x-value samples')
    parser.add_argument('--numShot', type=int, default=8192,
                       help='Shots per circuit')
    
    # Backend configuration
    parser.add_argument('-b', '--backend', default='aer_ideal',
                       help='Backend: aer_ideal (local), ionq_simulator, ionq_harmony, ionq_aria-1, ionq_forte, ionq_qpu')
    
    # Execution flags
    parser.add_argument('-E', '--execute', action='store_true', default=False,
                       help='Execute circuits (otherwise dry run)')
    parser.add_argument('-B', '--noBarrier', action='store_true', default=False,
                       help='Remove barriers from circuits')
    parser.add_argument('--submit-only', action='store_true', default=False,
                       help='Submit job and exit without waiting for results')
    
    # Training configuration
    parser.add_argument('--trainEpochs', type=int, default=300,
                       help='NN training epochs')
    parser.add_argument('--trainLR', type=float, default=0.1,
                       help='NN training learning rate')
    
    args = parser.parse_args()
    
    # Parse degrees
    args.degree_list = [int(d.strip()) for d in args.degrees.split(',')]
    
    return args


def run_experiment(args):
    """Run the complete native polynomial experiment on IonQ."""
    
    print("=" * 70)
    print("NATIVE POLYNOMIAL APPROXIMATION - IonQ")
    print("=" * 70)
    
    # Validate polynomials
    if not validate_polynomials():
        print("ERROR: Polynomial validation failed!")
        return
    
    # Setup output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get backend
    print(f"\nSetting up backend: {args.backend}")
    
    use_ionq = False
    if args.backend == 'aer_ideal':
        backend = AerSimulator()
        backend_description = 'Qiskit Aer ideal simulator'
        print(f"Backend: {backend.name} - {backend_description}")
    elif args.backend.startswith('ionq_'):
        if not IONQ_AVAILABLE:
            print("ERROR: qiskit-ionq not installed. Use 'aer_ideal' backend instead.")
            return
        
        ionq_backend_name = args.backend.replace('ionq_', '')
        backend_info = IONQ_BACKENDS.get(ionq_backend_name, IONQ_BACKENDS['simulator'])
        
        try:
            provider = IonQProvider(get_ionq_api_key())
            backend = provider.get_backend(backend_info['name'])
            backend_description = backend_info['description']
            use_ionq = True
            print(f"Backend: {backend.name} - {backend_description}")
        except Exception as e:
            print(f"ERROR connecting to IonQ: {e}")
            print("Falling back to Qiskit Aer ideal simulator...")
            backend = AerSimulator()
            backend_description = 'Qiskit Aer ideal simulator (fallback)'
            print(f"Backend: {backend.name} - {backend_description}")
    else:
        # Default to Aer
        backend = AerSimulator()
        backend_description = 'Qiskit Aer ideal simulator'
        print(f"Backend: {backend.name} - {backend_description}")
    
    # Configuration
    config = {
        'trials': args.trials,
        'shots': args.numShot,
        'num_samples': args.numSample,
        'x_range': (-0.9, 0.9),
        'train_epochs': args.trainEpochs,
        'train_lr': args.trainLR,
        'train_samples': 200,
    }
    
    print(f"\nConfiguration:")
    print(f"  Degrees: {args.degree_list}")
    print(f"  Trials per degree: {config['trials']}")
    print(f"  Shots per circuit: {config['shots']}")
    print(f"  Samples per trial: {config['num_samples']}")
    print(f"  Execute: {args.execute}")
    
    # Run experiments for each degree
    all_results = []
    
    for degree in args.degree_list:
        print(f"\n{'='*70}")
        print(f"DEGREE {degree}: {POLYNOMIALS[degree]['name']}")
        print(f"{'='*70}")
        
        poly_info = POLYNOMIALS[degree]
        true_coeffs = np.array(poly_info['coefficients'])
        
        # Target function
        def target_func(x):
            return evaluate_polynomial(x, true_coeffs)
        
        for trial in range(config['trials']):
            print(f"\n--- Trial {trial + 1}/{config['trials']} ---")
            
            # Set seeds for reproducibility
            seed = 42 + trial * 100 + degree
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Train neural network
            model = PolynomialNN(degree=degree)
            final_loss = train_nn(model, target_func, config, verbose=(args.verb > 1))
            learned_coeffs = model.get_coefficients()
            
            print(f"  NN Training Loss: {final_loss:.6f}")
            if args.verb > 0:
                print(f"  Learned coeffs: {learned_coeffs}")
                print(f"  True coeffs:    {true_coeffs}")
            
            # Generate test points
            x_values = np.linspace(config['x_range'][0], config['x_range'][1], config['num_samples'])
            
            # Build circuits
            circuits = []
            theoretical = []
            
            for x in x_values:
                qc, y_theo = build_native_polynomial_circuit(
                    x, learned_coeffs, add_barriers=not args.noBarrier
                )
                circuits.append(qc)
                theoretical.append(y_theo)
            
            theoretical = np.array(theoretical)
            
            # Classical predictions
            classical_pred = np.array([
                evaluate_polynomial(x, learned_coeffs) for x in x_values
            ])
            
            # Show sample circuit
            if args.verb > 1 or (trial == 0 and degree <= 3):
                print(f"\n  Sample circuit (x={x_values[0]:.2f}):")
                print(circuits[0].draw('text', fold=100))
            
            if not args.execute:
                print(f"  [DRY RUN] Would submit {len(circuits)} circuits")
                continue
            
            # Execute circuits
            print(f"  Executing {len(circuits)} circuits...")
            T0 = time()
            
            # Transpile circuits
            circuits_t = transpile(circuits, backend, optimization_level=1)
            
            # Run job
            if use_ionq and args.submit_only:
                # For IonQ, we want to submit and get the job ID
                # backend.run() returns a job object (already submitted)
                job = backend.run(circuits_t, shots=config['shots'])
                print(f"  Job submitted! Job ID: {job.job_id()}")
                print("  --submit-only flag set, skipping result retrieval.")
                continue
            
            job = backend.run(circuits_t, shots=config['shots'])
            
            # Wait for results
            result = job.result()
            
            elaT = time() - T0
            print(f"  Execution time: {elaT:.1f}s")
            
            # Process results
            measured = np.zeros(len(x_values))
            measured_err = np.zeros(len(x_values))
            
            # Get counts - handle both Aer and IonQ result formats
            if use_ionq:
                counts_list = job.get_counts()
                if not isinstance(counts_list, list):
                    counts_list = [counts_list]
            else:
                # Aer simulator result format
                counts_list = [result.get_counts(i) for i in range(len(circuits))]
            
            for i, counts in enumerate(counts_list):
                total_shots = sum(counts.values())
                poly_val, poly_err = extract_polynomial_from_counts(
                    counts, learned_coeffs, total_shots
                )
                measured[i] = poly_val
                measured_err[i] = poly_err
            
            # Compute metrics
            quantum_rmse = np.sqrt(np.mean((measured - theoretical) ** 2))
            classical_rmse = np.sqrt(np.mean((classical_pred - theoretical) ** 2))
            quantum_corr = np.corrcoef(theoretical, measured)[0, 1]
            classical_corr = np.corrcoef(theoretical, classical_pred)[0, 1]
            
            abs_errors = np.abs(measured - theoretical)
            pass_rate = np.mean(abs_errors < 0.03)
            poor_rate = np.mean((abs_errors >= 0.03) & (abs_errors < 0.1))
            fail_rate = np.mean(abs_errors >= 0.1)
            
            print(f"  Quantum RMSE: {quantum_rmse:.4f}")
            print(f"  Correlation: {quantum_corr:.4f}")
            print(f"  Pass Rate: {pass_rate*100:.1f}%")
            
            # Save results
            exp_name = f"deg{degree}_trial{trial:02d}"
            result_file = os.path.join(output_dir, f"{exp_name}.h5")
            
            h5_data = {
                'x_values': x_values,
                'theoretical': theoretical,
                'classical_pred': classical_pred,
                'measured': measured,
                'measured_err': measured_err,
                'true_coefficients': true_coeffs,
                'learned_coefficients': learned_coeffs,
            }
            
            h5_meta = {
                'degree': degree,
                'trial': trial,
                'polynomial_name': poly_info['name'],
                'backend': args.backend,
                'backend_name': backend.name,
                'shots': config['shots'],
                'num_samples': config['num_samples'],
                'approach': 'native',
                'provider': 'ionq' if use_ionq else 'qiskit_aer',
                'execution_time': elaT,
                'metrics': {
                    'quantum_rmse': float(quantum_rmse),
                    'classical_rmse': float(classical_rmse),
                    'quantum_corr': float(quantum_corr),
                    'classical_corr': float(classical_corr),
                    'pass_rate': float(pass_rate),
                    'poor_rate': float(poor_rate),
                    'fail_rate': float(fail_rate),
                },
                'config': config,
                'timestamp': datetime.now().isoformat(),
            }
            
            write4_data_hdf5(h5_data, result_file, h5_meta)
            print(f"  Saved: {result_file}")
            
            all_results.append({
                'degree': degree,
                'trial': trial,
                'metrics': h5_meta['metrics'],
            })
    
    # Summary
    if args.execute and all_results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        for degree in args.degree_list:
            degree_results = [r for r in all_results if r['degree'] == degree]
            if degree_results:
                avg_rmse = np.mean([r['metrics']['quantum_rmse'] for r in degree_results])
                avg_corr = np.mean([r['metrics']['quantum_corr'] for r in degree_results])
                avg_pass = np.mean([r['metrics']['pass_rate'] for r in degree_results])
                
                print(f"Degree {degree}: RMSE={avg_rmse:.4f}, Corr={avg_corr:.4f}, Pass={avg_pass*100:.1f}%")
        
        print(f"\nResults saved to: {output_dir}")
        print("\nTo generate plots, run:")
        print(f"  python plot_ionq_results.py --input {output_dir}")


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
