#!/usr/bin/env python3
"""
Submit Research Batch to IBM Quantum Cloud
===========================================

Submits polynomial recovery experiments for degrees 1-6 to IBM Quantum.

Usage:
    python submit_cloud_batch.py --backend ibm_boston --trials 10
    python submit_cloud_batch.py --status
    python submit_cloud_batch.py --retrieve
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from time import time, sleep
from datetime import datetime
import argparse

# Add parent paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from research_config import (
    POLYNOMIALS, CLOUD_CONFIG, evaluate_polynomial, validate_polynomials
)

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from toolbox.Util_H5io4 import write4_data_hdf5, read4_data_hdf5


# ==============================================================================
# NEURAL NETWORK
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
    
    return loss.item()


# ==============================================================================
# QUANTUM CIRCUITS
# ==============================================================================

def data_to_angle(x):
    """Convert x in [-1,1] to rotation angle."""
    x = np.clip(x, -1 + 1e-7, 1 - 1e-7)
    return np.arccos(x)


def build_polynomial_circuit(x, coefficients):
    """
    Build native quantum circuit for polynomial evaluation using quantum arithmetic.
    
    Uses quantum multiplication to compute powers natively on the quantum circuit.
    For degree d polynomial, uses d+1 qubits minimum.
    
    Strategy:
    - Qubit 0: stores x
    - Qubit 1: stores x^2 (computed via quantum multiplication: x * x)
    - Qubit 2: stores x^3 (computed via quantum multiplication: x^2 * x)
    - Qubit k: stores x^k (computed via quantum multiplication: x^(k-1) * x)
    
    After circuit execution, each qubit k encodes x^k in its expectation value <Z_k>.
    Terms are then combined classically (hybrid approach) or using quantum weighted sums.
    """
    degree = len(coefficients) - 1
    coeffs = np.array(coefficients)
    
    # Theoretical value for comparison
    theoretical = sum(coeffs[i] * (x ** i) for i in range(len(coeffs)))
    theoretical_clipped = np.clip(theoretical, -1 + 1e-6, 1 - 1e-6)
    
    # Number of qubits needed: degree qubits for x, x^2, ..., x^degree (constant term a0 is classical)
    n_qubits = max(1, degree)  # For degree 0, we still need 1 qubit
    
    # We need to measure all qubits to extract all powers
    qr = QuantumRegister(n_qubits, 'q')
    cr = ClassicalRegister(n_qubits, 'c')  # Measure all qubits, one classical bit per qubit
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
    # Quantum multiplication: to compute x^k, we need:
    # 1. Encode x^(k-1) on qubit (k-1) [already done]
    # 2. Encode x on qubit k
    # 3. Apply RZ(π/2) on qubit k
    # 4. Apply CNOT from qubit (k-1) to qubit k
    # Result: <Z_k> = x^(k-1) * x = x^k
    
    # x^2 = x * x
    if degree >= 2:
        # Copy x to qubit 1
        qc.ry(data_to_angle(x), 1)
        qc.barrier()
        # Quantum multiplication: <Z_1> = x * x = x^2
        qc.rz(np.pi/2, 1)
        qc.cx(0, 1)
        # Now: <Z_1> = x^2
    
    # x^3 = x^2 * x
    if degree >= 3:
        # Copy x to qubit 2
        qc.ry(data_to_angle(x), 2)
        qc.barrier()
        # Quantum multiplication: <Z_2> = x^2 * x = x^3
        qc.rz(np.pi/2, 2)
        qc.cx(1, 2)
        # Now: <Z_2> = x^3
    
    # x^4 = x^3 * x
    if degree >= 4:
        qc.ry(data_to_angle(x), 3)
        qc.barrier()
        qc.rz(np.pi/2, 3)
        qc.cx(2, 3)
        # Now: <Z_3> = x^4
    
    # Continue for higher degrees
    for k in range(5, degree + 1):
        if k < n_qubits:
            qc.ry(data_to_angle(x), k)
            qc.barrier()
            qc.rz(np.pi/2, k)
            qc.cx(k-1, k)
            # Now: <Z_k> = x^k
    
    # ========================================================================
    # STEP 3: Measure all qubits to extract all powers
    # ========================================================================
    # Measure all qubits: <Z_0> = x, <Z_1> = x^2, ..., <Z_k> = x^(k+1)
    # Then combine classically: F(x) = a0 + a1*<Z_0> + a2*<Z_1> + ... + ad*<Z_(d-1)>
    # Map each qubit k to classical bit k
    for k in range(n_qubits):
        qc.measure(k, k)
    
    return qc, theoretical_clipped


# ==============================================================================
# IBM SERVICE
# ==============================================================================

def get_service():
    """Get IBM Quantum Runtime Service."""
    token = os.environ.get('IBM_QUANTUM_TOKEN')
    channel = os.environ.get('QISKIT_IBM_CHANNEL', 'ibm_cloud')
    instance = os.environ.get('QISKIT_IBM_INSTANCE')
    
    if not token:
        raise ValueError("IBM_QUANTUM_TOKEN not set. Check .env file.")
    
    return QiskitRuntimeService(channel=channel, token=token, instance=instance)


# ==============================================================================
# BATCH SUBMISSION
# ==============================================================================

def submit_degree_experiment(degree, trial, backend, config, output_dir):
    """Submit a single degree experiment to IBM Quantum."""
    
    poly_info = POLYNOMIALS[degree]
    true_coeffs = np.array(poly_info['coefficients'])
    
    # Train NN
    def target_func(x):
        return evaluate_polynomial(x, true_coeffs)
    
    model = PolynomialNN(degree=degree)
    seed = 42 + trial * 100 + degree
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    final_loss = train_nn(model, target_func, config, verbose=False)
    learned_coeffs = model.get_coefficients()
    
    # Generate test points
    x_range = config['x_range']
    x_values = np.linspace(x_range[0], x_range[1], config['num_samples'])
    
    # Build circuits
    circuits = []
    theoretical = []
    
    for x in x_values:
        qc, y_theo = build_polynomial_circuit(x, learned_coeffs)
        circuits.append(qc)
        theoretical.append(y_theo)
    
    theoretical = np.array(theoretical)
    
    # Classical predictions
    classical_pred = np.array([
        evaluate_polynomial(x, learned_coeffs) for x in x_values
    ])
    
    # Get service and backend
    service = get_service()
    backend_obj = service.backend(backend)
    
    # Transpile circuits
    circuits_t = transpile(circuits, backend_obj, optimization_level=1)
    
    # Submit job without custom options (use defaults)
    sampler = Sampler(backend_obj)
    job = sampler.run(circuits_t, shots=config['shots'])
    
    job_id = job.job_id()
    
    # Save submission metadata
    exp_name = f"deg{degree}_trial{trial:02d}"
    out_file = os.path.join(output_dir, f"{exp_name}_submitted.h5")
    
    h5_data = {
        'x_values': x_values,
        'theoretical': theoretical,
        'classical_pred': classical_pred,
        'true_coefficients': true_coeffs,
        'learned_coefficients': learned_coeffs,
    }
    
    h5_meta = {
        'degree': degree,
        'trial': trial,
        'polynomial_name': poly_info['name'],
        'job_id': job_id,
        'backend': backend,
        'shots': config['shots'],
        'num_samples': config['num_samples'],
        'approach': 'native',  # Native polynomial protocol (multi-qubit quantum arithmetic)
        'submitted_at': datetime.now().isoformat(),
        'status': 'submitted',
    }
    
    write4_data_hdf5(h5_data, out_file, h5_meta)
    
    print(f"  Deg {degree} Trial {trial}: Submitted job {job_id}")
    
    return job_id, exp_name


def submit_all_experiments(backend, config, output_dir):
    """Submit all experiments to IBM Quantum."""
    print("=" * 70)
    print(f"SUBMITTING TO IBM QUANTUM: {backend}")
    print("=" * 70)
    
    if not validate_polynomials():
        print("ERROR: Polynomial validation failed!")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    degrees = sorted(POLYNOMIALS.keys())
    n_trials = config['trials']
    total_jobs = len(degrees) * n_trials
    
    print(f"\nSubmitting {total_jobs} jobs ({len(degrees)} degrees x {n_trials} trials)")
    print("-" * 70)
    
    all_jobs = []
    
    for degree in degrees:
        print(f"\nDegree {degree} ({POLYNOMIALS[degree]['name']}):")
        
        for trial in range(n_trials):
            try:
                job_id, exp_name = submit_degree_experiment(
                    degree, trial, backend, config, output_dir
                )
                all_jobs.append({
                    'job_id': job_id,
                    'exp_name': exp_name,
                    'degree': degree,
                    'trial': trial,
                })
                
                # Small delay between submissions
                sleep(0.5)
                
            except Exception as e:
                print(f"  ERROR submitting deg {degree} trial {trial}: {e}")
    
    # Save job manifest
    manifest_file = os.path.join(output_dir, 'job_manifest.json')
    manifest = {
        'backend': backend,
        'submitted_at': datetime.now().isoformat(),
        'total_jobs': len(all_jobs),
        'jobs': all_jobs,
    }
    
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("\n" + "-" * 70)
    print(f"Submitted {len(all_jobs)} jobs to {backend}")
    print(f"Manifest saved to: {manifest_file}")
    
    return all_jobs


def check_job_status(output_dir):
    """Check status of submitted jobs."""
    manifest_file = os.path.join(output_dir, 'job_manifest.json')
    
    if not os.path.exists(manifest_file):
        print(f"No manifest found at: {manifest_file}")
        return
    
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
    
    service = get_service()
    
    print("=" * 70)
    print(f"JOB STATUS - {manifest['backend']}")
    print("=" * 70)
    
    status_counts = {}
    
    for job_info in manifest['jobs']:
        job_id = job_info['job_id']
        try:
            job = service.job(job_id)
            status = str(job.status())
        except Exception as e:
            status = f"ERROR: {e}"
        
        status_counts[status] = status_counts.get(status, 0) + 1
        print(f"  {job_info['exp_name']}: {status}")
    
    print("-" * 70)
    print("Summary:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")


def retrieve_all_results(output_dir):
    """Retrieve results from all completed jobs."""
    manifest_file = os.path.join(output_dir, 'job_manifest.json')
    
    if not os.path.exists(manifest_file):
        print(f"No manifest found at: {manifest_file}")
        return
    
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
    
    service = get_service()
    
    print("=" * 70)
    print("RETRIEVING RESULTS")
    print("=" * 70)
    
    completed = 0
    failed = 0
    
    for job_info in manifest['jobs']:
        job_id = job_info['job_id']
        exp_name = job_info['exp_name']
        
        # Load submitted data
        submitted_file = os.path.join(output_dir, f"{exp_name}_submitted.h5")
        if not os.path.exists(submitted_file):
            print(f"  {exp_name}: Missing submitted file")
            failed += 1
            continue
        
        data, meta = read4_data_hdf5(submitted_file, verb=0)
        
        try:
            job = service.job(job_id)
            status = str(job.status())
            
            if status != 'DONE':
                print(f"  {exp_name}: Status = {status}")
                continue
            
            # Get results
            result = job.result()
            
            n_samples = len(data['x_values'])
            measured = np.zeros(n_samples)
            measured_err = np.zeros(n_samples)
            
            # Get learned coefficients for polynomial evaluation
            learned_coeffs = data['learned_coefficients']
            degree = meta['degree']
            
            for i in range(n_samples):
                pub_result = result[i]
                counts = pub_result.data.c.get_counts()
                total_shots = sum(counts.values())
                
                if total_shots == 0:
                    measured[i] = 0.0
                    measured_err[i] = 1.0
                    continue
                
                # Extract expectation values <Z_k> for each qubit k
                # Qiskit bitstrings: for measure(q[k], c[k]), the bitstring has format
                # where bit at position k represents qubit k's measurement
                # Bitstring is typically in "big-endian" format: '000' means q0=0, q1=0, q2=0
                # But after measure(q[k], c[k]), bitstring[k] corresponds to qubit k
                n_qubits = degree  # Number of qubits measured (x, x^2, ..., x^degree)
                powers = np.zeros(degree + 1)  # powers[0] = constant (1), powers[k] = x^k
                powers[0] = 1.0  # Constant term x^0 = 1
                
                # Extract <Z_k> for each qubit k = 0, 1, ..., degree-1
                # Qubit 0 stores x, qubit 1 stores x^2, ..., qubit k stores x^(k+1)
                for k in range(n_qubits):
                    n0_k = 0  # Count when qubit k is 0
                    n1_k = 0  # Count when qubit k is 1
                    
                    for bitstring, count in counts.items():
                        # Bitstring format: for measure(q[k], c[k]), bitstring[k] is qubit k
                        # Remove any spaces and pad if needed
                        bitstring_clean = bitstring.replace(' ', '')
                        bitstring_padded = bitstring_clean.zfill(n_qubits)
                        
                        # Get bit k: in standard Qiskit format, position k from left (MSB) 
                        # or from right (LSB)? For explicit measure(q[k], c[k]), it's position k
                        if k < len(bitstring_padded):
                            # Try both orderings - typically it's reversed (LSB first)
                            # Actually, let's check: for measure(q0, c0), measure(q1, c1), ...
                            # the bitstring format depends on how Sampler returns it
                            # Common format: rightmost bit is qubit 0 (LSB)
                            bit_idx = len(bitstring_padded) - 1 - k  # Reverse indexing
                            bit_k = int(bitstring_padded[bit_idx])
                            if bit_k == 0:
                                n0_k += count
                            else:
                                n1_k += count
                        else:
                            n0_k += count  # Assume 0 if bitstring too short
                    
                    # <Z_k> = (n0_k - n1_k) / total_shots
                    # Qubit k stores x^(k+1), so <Z_k> = x^(k+1)
                    x_power = (n0_k - n1_k) / total_shots if total_shots > 0 else 0.0
                    powers[k+1] = x_power
                
                # Compute polynomial: F(x) = a0 + a1*x + a2*x^2 + ... + ad*x^d
                # where powers[k] = x^k (powers[0] = 1, powers[1] = x, powers[2] = x^2, ...)
                poly_value = sum(learned_coeffs[j] * powers[j] for j in range(len(learned_coeffs)))
                
                measured[i] = poly_value
                
                # Estimate error from shot noise
                # Shot noise per qubit: ~1/sqrt(shots)
                # Propagate through polynomial: sum of |coefficient| * shot_error
                shot_err = 1.0 / np.sqrt(total_shots) if total_shots > 0 else 1.0
                poly_err = sum(abs(learned_coeffs[j]) * shot_err for j in range(1, len(learned_coeffs)))
                measured_err[i] = poly_err
            
            # Compute metrics
            theoretical = data['theoretical']
            classical_pred = data['classical_pred']
            
            quantum_rmse = np.sqrt(np.mean((measured - theoretical) ** 2))
            classical_rmse = np.sqrt(np.mean((classical_pred - theoretical) ** 2))
            quantum_corr = np.corrcoef(theoretical, measured)[0, 1]
            classical_corr = np.corrcoef(theoretical, classical_pred)[0, 1]
            
            abs_errors = np.abs(measured - theoretical)
            pass_rate = np.mean(abs_errors < 0.03)
            poor_rate = np.mean((abs_errors >= 0.03) & (abs_errors < 0.1))
            fail_rate = np.mean(abs_errors >= 0.1)
            
            # Save complete results
            result_file = os.path.join(output_dir, f"{exp_name}.h5")
            
            h5_data = {
                'x_values': data['x_values'],
                'theoretical': theoretical,
                'classical_pred': classical_pred,
                'measured': measured,
                'measured_err': measured_err,
                'true_coefficients': data['true_coefficients'],
                'learned_coefficients': data['learned_coefficients'],
            }
            
            h5_meta = {
                'degree': meta['degree'],
                'trial': meta['trial'],
                'polynomial_name': meta['polynomial_name'],
                'job_id': job_id,
                'backend': meta['backend'],
                'approach': meta.get('approach', 'native'),  # Native polynomial protocol (multi-qubit quantum arithmetic)
                'metrics': {
                    'quantum_rmse': float(quantum_rmse),
                    'classical_rmse': float(classical_rmse),
                    'quantum_corr': float(quantum_corr),
                    'classical_corr': float(classical_corr),
                    'pass_rate': float(pass_rate),
                    'poor_rate': float(poor_rate),
                    'fail_rate': float(fail_rate),
                },
                'config': {
                    'shots': meta['shots'],
                    'num_samples': meta['num_samples'],
                    'backend': meta['backend'],
                },
                'status': 'completed',
            }
            
            write4_data_hdf5(h5_data, result_file, h5_meta)
            
            print(f"  {exp_name}: Q-RMSE={quantum_rmse:.4f}, Corr={quantum_corr:.4f}, Pass={pass_rate*100:.0f}%")
            completed += 1
            
        except Exception as e:
            print(f"  {exp_name}: ERROR - {e}")
            failed += 1
    
    print("-" * 70)
    print(f"Completed: {completed}, Failed/Pending: {failed}")


# ==============================================================================
# MAIN
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Submit research batch to IBM Quantum'
    )
    
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--submit', action='store_true',
                     help='Submit jobs to IBM cloud')
    mode.add_argument('--status', action='store_true',
                     help='Check status of submitted jobs')
    mode.add_argument('--retrieve', action='store_true',
                     help='Retrieve results from IBM cloud')
    
    parser.add_argument('--backend', default='ibm_boston',
                       help='IBM backend name')
    parser.add_argument('--trials', type=int, default=10,
                       help='Number of trials per configuration')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(script_dir, 'results', 'cloud')
    
    os.makedirs(output_dir, exist_ok=True)
    
    if args.submit:
        config = CLOUD_CONFIG.copy()
        config['trials'] = args.trials
        submit_all_experiments(args.backend, config, output_dir)
        
    elif args.status:
        check_job_status(output_dir)
        
    elif args.retrieve:
        retrieve_all_results(output_dir)

