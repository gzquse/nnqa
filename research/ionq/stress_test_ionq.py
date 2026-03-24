#!/usr/bin/env python3
"""
IonQ Stress Test: High-Degree Polynomial Evaluation
====================================================

Minimal-budget stress test for IonQ Forte-1 testing polynomial degrees up to 20.
Uses sparse sampling (degrees 1, 5, 10, 15, 20) with reduced shots and samples.

Usage:
    # Submit stress test jobs (submit-only mode)
    python stress_test_ionq.py --execute --submit-only
    
    # Dry run to check circuits
    python stress_test_ionq.py
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pprint import pprint
from time import time
from datetime import datetime
import argparse
import json

# Add parent paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

# Try to import IonQ provider
try:
    from qiskit_ionq import IonQProvider
    IONQ_AVAILABLE = True
except ImportError:
    IONQ_AVAILABLE = False
    print("Warning: qiskit-ionq not available, using Qiskit Aer only")

from toolbox.Util_IOfunc import dateT2Str
from toolbox.Util_H5io4 import write4_data_hdf5

from ionq_config import (
    get_ionq_api_key,
    IONQ_BACKENDS,
    POLYNOMIALS,
    evaluate_polynomial,
    validate_polynomials,
)
from submit_ionq_native import PolynomialNN, train_nn, build_native_polynomial_circuit, extract_polynomial_from_counts

# ==============================================================================
# STRESS TEST CONFIGURATION
# ==============================================================================

# Sparse degree sampling for stress test (up to 35 qubits)
STRESS_TEST_DEGREES = [1, 5, 10, 15, 20, 25, 30, 35]

# Minimal budget parameters
STRESS_TEST_CONFIG = {
    'trials': 1,  # Single trial per degree
    'shots': 1024,  # Reduced from 8192 (8× reduction)
    'num_samples': 5,  # Reduced from 25 (5× reduction)
    'x_range': (-0.9, 0.9),
    'train_epochs': 300,
    'train_lr': 0.1,
    'train_samples': 200,
}

# IonQ Forte-1 device limits (approximate)
FORTE1_LIMITS = {
    'max_qubits': 35,  # Extended to 35 qubits for stress test
    'max_depth': 1000,  # Approximate depth limit
}

# ==============================================================================
# CIRCUIT VALIDATION
# ==============================================================================

def validate_circuit_for_backend(qc, backend_name, verb=1):
    """
    Validate circuit fits within backend constraints.
    
    Returns:
        (is_valid, issues): tuple of (bool, list of issue strings)
    """
    issues = []
    
    n_qubits = qc.num_qubits
    depth = qc.depth()
    ops = qc.count_ops()
    n_2q = ops.get('cx', 0) + ops.get('cz', 0) + ops.get('ecr', 0)
    
    # Check qubit count
    if n_qubits > FORTE1_LIMITS['max_qubits']:
        issues.append(f"Too many qubits: {n_qubits} > {FORTE1_LIMITS['max_qubits']}")
    
    # Check depth
    if depth > FORTE1_LIMITS['max_depth']:
        issues.append(f"Circuit too deep: {depth} > {FORTE1_LIMITS['max_depth']}")
    
    if verb > 0:
        print(f"  Circuit: {n_qubits} qubits, depth={depth}, 2Q gates={n_2q}")
    
    is_valid = len(issues) == 0
    return is_valid, issues

# ==============================================================================
# COST ESTIMATION
# ==============================================================================

def estimate_cost(degrees, config):
    """Estimate total cost (shots) for stress test."""
    total_circuits = len(degrees) * config['trials'] * config['num_samples']
    total_shots = total_circuits * config['shots']
    
    # Compare to full test (6 degrees × 3 trials × 25 samples × 8192 shots)
    full_test_shots = 6 * 3 * 25 * 8192
    
    return {
        'degrees': len(degrees),
        'trials_per_degree': config['trials'],
        'samples_per_trial': config['num_samples'],
        'shots_per_circuit': config['shots'],
        'total_circuits': total_circuits,
        'total_shots': total_shots,
        'full_test_shots': full_test_shots,
        'reduction_factor': full_test_shots / total_shots if total_shots > 0 else 0,
        'reduction_percent': (1 - total_shots / full_test_shots) * 100 if full_test_shots > 0 else 0,
    }

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='IonQ stress test for high-degree polynomials'
    )
    
    parser.add_argument('-v', '--verb', type=int, default=1,
                       help='Verbosity level')
    parser.add_argument('--basePath', default='.',
                       help='Base path for output')
    
    # Stress test configuration
    parser.add_argument('--degrees', default='1,5,10,15,20,25,30,35',
                       help='Comma-separated list of polynomial degrees (default: 1,5,10,15,20,25,30,35)')
    parser.add_argument('--numSample', type=int, default=5,
                       help='Number of x-value samples (default: 5, minimal budget)')
    parser.add_argument('--numShot', type=int, default=1024,
                       help='Shots per circuit (default: 1024, minimal budget)')
    
    # Backend configuration
    parser.add_argument('-b', '--backend', default='ionq_forte-1',
                       help='Backend: ionq_forte-1 (default), ionq_simulator, etc.')
    
    # Execution flags
    parser.add_argument('-E', '--execute', action='store_true', default=False,
                       help='Execute circuits (otherwise dry run)')
    parser.add_argument('--submit-only', action='store_true', default=False,
                       help='Submit job and exit without waiting for results')
    parser.add_argument('--wait', action='store_true', default=False,
                       help='Wait for results (overrides --submit-only, default: submit-only mode)')
    parser.add_argument('-B', '--noBarrier', action='store_true', default=False,
                       help='Remove barriers from circuits')
    
    # Training configuration
    parser.add_argument('--trainEpochs', type=int, default=300,
                       help='NN training epochs')
    parser.add_argument('--trainLR', type=float, default=0.1,
                       help='NN training learning rate')
    
    args = parser.parse_args()
    
    # Parse degrees
    args.degree_list = [int(d.strip()) for d in args.degrees.split(',')]
    
    # Default behavior: submit-only unless --wait is specified
    if not args.wait:
        args.submit_only = True
    
    return args

def run_stress_test(args):
    """Run the stress test experiment on IonQ."""
    
    print("=" * 70)
    print("IONQ STRESS TEST: High-Degree Polynomial Evaluation")
    print("=" * 70)
    
    # Validate polynomials
    if not validate_polynomials():
        print("ERROR: Polynomial validation failed!")
        return
    
    # Setup output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'results', 'stress_test')
    os.makedirs(output_dir, exist_ok=True)
    
    # Cost estimation
    config = {
        'trials': 1,
        'shots': args.numShot,
        'num_samples': args.numSample,
        'x_range': (-0.9, 0.9),
        'train_epochs': args.trainEpochs,
        'train_lr': args.trainLR,
        'train_samples': 200,
    }
    
    cost_est = estimate_cost(args.degree_list, config)
    print(f"\nCost Estimation:")
    print(f"  Degrees: {cost_est['degrees']}")
    print(f"  Trials per degree: {cost_est['trials_per_degree']}")
    print(f"  Samples per trial: {cost_est['samples_per_trial']}")
    print(f"  Shots per circuit: {cost_est['shots_per_circuit']}")
    print(f"  Total circuits: {cost_est['total_circuits']}")
    print(f"  Total shots: {cost_est['total_shots']:,}")
    print(f"  Budget reduction: {cost_est['reduction_percent']:.1f}% vs full test")
    print(f"  (Full test would use: {cost_est['full_test_shots']:,} shots)")
    
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
        backend = AerSimulator()
        backend_description = 'Qiskit Aer ideal simulator'
        print(f"Backend: {backend.name} - {backend_description}")
    
    print(f"\nConfiguration:")
    print(f"  Degrees: {args.degree_list}")
    print(f"  Shots per circuit: {config['shots']}")
    print(f"  Samples per trial: {config['num_samples']}")
    print(f"  Execute: {args.execute}")
    print(f"  Submit-only: {args.submit_only}")
    
    # Track all jobs
    all_jobs = []
    
    # Run experiments for each degree
    for degree in args.degree_list:
        print(f"\n{'='*70}")
        print(f"DEGREE {degree}: {POLYNOMIALS[degree]['name']}")
        print(f"{'='*70}")
        
        if degree not in POLYNOMIALS:
            print(f"  ERROR: Degree {degree} not defined in POLYNOMIALS")
            continue
        
        poly_info = POLYNOMIALS[degree]
        true_coeffs = np.array(poly_info['coefficients'])
        
        # Target function
        def target_func(x):
            return evaluate_polynomial(x, true_coeffs)
        
        trial = 0  # Single trial
        print(f"\n--- Trial {trial + 1} ---")
        
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
            print(f"  Learned coeffs: {learned_coeffs[:5]}..." if len(learned_coeffs) > 5 else f"  Learned coeffs: {learned_coeffs}")
        
        # Generate test points
        x_values = np.linspace(config['x_range'][0], config['x_range'][1], config['num_samples'])
        
        # Build circuits
        circuits = []
        theoretical = []
        circuit_info = []
        
        for x in x_values:
            qc, y_theo = build_native_polynomial_circuit(
                x, learned_coeffs, add_barriers=not args.noBarrier
            )
            circuits.append(qc)
            theoretical.append(y_theo)
            
            # Validate circuit
            is_valid, issues = validate_circuit_for_backend(qc, args.backend, verb=args.verb)
            if not is_valid:
                print(f"  WARNING: Circuit validation failed for x={x:.2f}: {issues}")
            
            # Store circuit info
            ops = qc.count_ops()
            circuit_info.append({
                'qubits': qc.num_qubits,
                'depth': qc.depth(),
                '2q_gates': ops.get('cx', 0) + ops.get('cz', 0),
            })
        
        theoretical = np.array(theoretical)
        
        # Classical predictions
        classical_pred = np.array([
            evaluate_polynomial(x, learned_coeffs) for x in x_values
        ])
        
        # Circuit resource summary
        avg_qubits = np.mean([c['qubits'] for c in circuit_info])
        avg_depth = np.mean([c['depth'] for c in circuit_info])
        avg_2q = np.mean([c['2q_gates'] for c in circuit_info])
        print(f"  Circuit resources: {avg_qubits:.1f} qubits avg, depth={avg_depth:.1f}, 2Q gates={avg_2q:.1f}")
        
        # Show sample circuit
        if args.verb > 1 or (degree <= 3):
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
            # Submit and get job ID
            job = backend.run(circuits_t, shots=config['shots'])
            job_id = job.job_id()
            print(f"  Job submitted! Job ID: {job_id}")
            print("  --submit-only flag set, skipping result retrieval.")
            
            # Save job info
            all_jobs.append({
                'degree': degree,
                'trial': trial,
                'job_id': job_id,
                'backend': args.backend,
                'num_circuits': len(circuits),
                'shots': config['shots'],
            })
            continue
        
        # Wait for results (if not submit-only)
        job = backend.run(circuits_t, shots=config['shots'])
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
            counts_list = [result.get_counts(i) for i in range(len(circuits))]
        
        for i, counts in enumerate(counts_list):
            if i >= len(x_values):
                break
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
        exp_name = f"stress_deg{degree}_trial{trial:02d}"
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
            'circuit_info': circuit_info,
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
    
    # Save job manifest if submit-only
    if args.execute and args.submit_only and all_jobs:
        manifest_file = os.path.join(output_dir, 'job_manifest.json')
        manifest = {
            'backend': args.backend,
            'submitted_at': datetime.now().isoformat(),
            'total_jobs': len(all_jobs),
            'config': config,
            'cost_estimation': cost_est,
            'jobs': all_jobs,
        }
        
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print("\n" + "=" * 70)
        print("JOB SUBMISSION SUMMARY")
        print("=" * 70)
        print(f"Submitted {len(all_jobs)} jobs to {args.backend}")
        print(f"Manifest saved to: {manifest_file}")
        print("\nTo retrieve results later, run:")
        print(f"  python retrieve_stress_test.py")

if __name__ == '__main__':
    args = parse_args()
    run_stress_test(args)
