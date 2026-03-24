#!/usr/bin/env python3
"""
Submit Direct Approach (1-qubit) to IBM Quantum Cloud
======================================================

Submits polynomial recovery experiments using direct approach:
- Classical polynomial computation
- Single-qubit encoding of result
- Measurement

Usage:
    python submit_direct_cloud_batch.py --submit --backend ibm_boston --trials 10
    python submit_direct_cloud_batch.py --status
    python submit_direct_cloud_batch.py --retrieve
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
# QUANTUM CIRCUITS - DIRECT APPROACH
# ==============================================================================

def data_to_angle(x):
    """Convert x in [-1,1] to rotation angle."""
    x = np.clip(x, -1 + 1e-7, 1 - 1e-7)
    return np.arccos(x)


def build_direct_circuit(x, coefficients):
    """
    Build direct approach circuit (1-qubit).
    
    Direct approach:
    1. Compute polynomial classically: y = F(x) = sum(a_i * x^i)
    2. Encode result into single qubit: Ry(arccos(y))|0⟩
    3. Measure: <Z> = y
    
    This demonstrates encoding/decoding fidelity with constant resources.
    """
    degree = len(coefficients) - 1
    coeffs = np.array(coefficients)
    
    # Classical polynomial evaluation
    theoretical = sum(coeffs[i] * (x ** i) for i in range(len(coeffs)))
    theoretical_clipped = np.clip(theoretical, -1 + 1e-6, 1 - 1e-6)
    
    # Single-qubit circuit
    qr = QuantumRegister(1, 'q')
    cr = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(qr, cr)
    
    # Encode result into qubit
    qc.ry(data_to_angle(theoretical_clipped), 0)
    
    # Measure
    qc.measure(0, 0)
    
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
    """Submit a single degree experiment to IBM Quantum (direct approach)."""
    
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
    
    # Build circuits (direct approach)
    circuits = []
    theoretical = []
    
    for x in x_values:
        qc, y_theo = build_direct_circuit(x, learned_coeffs)
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
    
    # Submit job
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
        'approach': 'direct',  # Direct approach (1-qubit encoding)
        'submitted_at': datetime.now().isoformat(),
        'status': 'submitted',
    }
    
    write4_data_hdf5(h5_data, out_file, h5_meta)
    
    print(f"  Deg {degree} Trial {trial}: Submitted job {job_id}")
    
    return job_id, exp_name


def submit_all_experiments(backend, config, output_dir):
    """Submit all experiments to IBM Quantum."""
    print("=" * 70)
    print(f"SUBMITTING DIRECT APPROACH TO IBM QUANTUM: {backend}")
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
        'approach': 'direct',
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
    print(f"JOB STATUS - {manifest['backend']} ({manifest.get('approach', 'unknown')} approach)")
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
    print("RETRIEVING RESULTS (Direct Approach)")
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
            
            for i in range(n_samples):
                pub_result = result[i]
                counts = pub_result.data.c.get_counts()
                n0 = counts.get('0', 0)
                n1 = counts.get('1', 0)
                total = n0 + n1
                
                mprob = n1 / total if total > 0 else 0.5
                measured[i] = 1 - 2 * mprob
                measured_err[i] = 2 * np.sqrt(mprob * (1 - mprob) / total) if total > 0 else 0
            
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
                'approach': meta.get('approach', 'direct'),
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Submit direct approach batch to IBM Quantum'
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
        output_dir = os.path.join(script_dir, 'results', 'direct')
    
    os.makedirs(output_dir, exist_ok=True)
    
    if args.submit:
        config = CLOUD_CONFIG.copy()
        config['trials'] = args.trials
        submit_all_experiments(args.backend, config, output_dir)
        
    elif args.status:
        check_job_status(output_dir)
        
    elif args.retrieve:
        retrieve_all_results(output_dir)
