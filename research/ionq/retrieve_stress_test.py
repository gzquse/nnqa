#!/usr/bin/env python3
"""
Retrieve IonQ Stress Test Jobs
================================

Retrieves results from stress test job manifest and processes them.
Reads job IDs from the manifest file created by stress_test_ionq.py.

Usage:
    python retrieve_stress_test.py
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from qiskit_ionq import IonQProvider

# Add parent paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from toolbox.Util_H5io4 import write4_data_hdf5
from ionq_config import get_ionq_api_key, POLYNOMIALS, evaluate_polynomial
from submit_ionq_native import PolynomialNN, train_nn, build_native_polynomial_circuit, extract_polynomial_from_counts

def retrieve_stress_test_jobs():
    """Retrieve and process all stress test jobs from manifest."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results', 'stress_test')
    manifest_file = os.path.join(results_dir, 'job_manifest.json')
    
    if not os.path.exists(manifest_file):
        print(f"ERROR: Manifest file not found: {manifest_file}")
        print("Run stress_test_ionq.py first to submit jobs.")
        return
    
    # Load manifest
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
    
    print("=" * 70)
    print("RETRIEVING IONQ STRESS TEST JOBS")
    print("=" * 70)
    print(f"Manifest: {manifest_file}")
    print(f"Total jobs: {manifest['total_jobs']}")
    print(f"Backend: {manifest['backend']}")
    
    provider = IonQProvider(get_ionq_api_key())
    backend = provider.get_backend('ionq_qpu.forte-1')
    
    output_dir = results_dir
    os.makedirs(output_dir, exist_ok=True)
    
    config = manifest['config']
    
    for job_info in manifest['jobs']:
        degree = job_info['degree']
        job_id = job_info['job_id']
        trial = job_info['trial']
        
        print(f"\nProcessing Degree {degree} (Job ID: {job_id})")
        
        # 1. Reconstruct Context (Train NN)
        print("  Reconstructing experimental context...")
        poly_info = POLYNOMIALS[degree]
        true_coeffs = np.array(poly_info['coefficients'])
        
        def target_func(x):
            return evaluate_polynomial(x, true_coeffs)
        
        # Set seeds for reproducibility (matches stress_test_ionq.py)
        seed = 42 + trial * 100 + degree
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Train neural network
        model = PolynomialNN(degree=degree)
        train_nn(model, target_func, config, verbose=False)
        learned_coeffs = model.get_coefficients()
        
        print(f"  Learned coeffs: {learned_coeffs[:5]}..." if len(learned_coeffs) > 5 else f"  Learned coeffs: {learned_coeffs}")
        
        # Generate test points
        x_values = np.linspace(config['x_range'][0], config['x_range'][1], config['num_samples'])
        
        # Re-calculate theoretical values
        theoretical = []
        for x in x_values:
            _, y_theo = build_native_polynomial_circuit(x, learned_coeffs, add_barriers=True)
            theoretical.append(y_theo)
        theoretical = np.array(theoretical)
        
        # Classical predictions
        classical_pred = np.array([
            evaluate_polynomial(x, learned_coeffs) for x in x_values
        ])
        
        # 2. Retrieve Job
        print(f"  Retrieving job {job_id}...")
        try:
            job = backend.retrieve_job(job_id)
            status = job.status()
            print(f"  Job Status: {status}")
            
            if status.name not in ['DONE', 'COMPLETED']:
                print(f"  Job is not complete. Status: {status}")
                continue
            
            result = job.result()
            counts_list = job.get_counts()
            
            # Handle different return formats
            if isinstance(counts_list, dict):
                counts_list = [counts_list]
            
            if len(counts_list) != len(x_values):
                print(f"  WARNING: Mismatch in counts length. Expected {len(x_values)}, got {len(counts_list)}")
            
            # 3. Process Results
            measured = np.zeros(len(x_values))
            measured_err = np.zeros(len(x_values))
            
            for i, counts in enumerate(counts_list):
                if i >= len(x_values):
                    break
                
                total_shots = sum(counts.values())
                poly_val, poly_err = extract_polynomial_from_counts(
                    counts, learned_coeffs, total_shots
                )
                measured[i] = poly_val
                measured_err[i] = poly_err
            
            # 4. Save Results
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
                'backend': manifest['backend'],
                'backend_name': 'ionq_qpu.forte-1',
                'shots': config['shots'],
                'num_samples': config['num_samples'],
                'approach': 'native',
                'provider': 'ionq',
                'execution_time': 0,  # Unknown from retrieval
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
                'job_id': job_id,
            }
            
            write4_data_hdf5(h5_data, result_file, h5_meta)
            print(f"  Saved: {result_file}")
        
        except Exception as e:
            print(f"  ERROR retrieving/processing job: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("RETRIEVAL COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print("\nTo generate plots, run:")
    print(f"  python plot_stress_test.py --input {output_dir}")

if __name__ == '__main__':
    retrieve_stress_test_jobs()
