
#!/usr/bin/env python3
"""
Retrieve IonQ Native Polynomial Jobs
====================================

Retrieves results for specific IonQ job IDs, processes them, and saves to HDF5.
Reconstructs the experimental context (training data, coefficients) by re-running
the deterministic training process used in submission.

Usage:
    python retrieve_ionq_jobs.py
"""

import sys
import os
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

# Jobs to retrieve: fill in with your own job_id values after running submit_ionq_native.py
JOBS = [
    # Example entry (replace uuid and metadata):
    # {
    #     'degree': 1,
    #     'job_id': '<ionq-job-uuid>',
    #     'trial': 0,
    #     'backend_name': 'ionq_qpu.forte-1',
    # },
]

# Shared configuration (must match submit_ionq_native.py defaults)
CONFIG = {
    'trials': 1, # Was 1 in submission
    'shots': 8192,
    'num_samples': 25,
    'x_range': (-0.9, 0.9),
    'train_epochs': 300,
    'train_lr': 0.1,
    'train_samples': 200,
}

def retrieve_and_process():
    print("=" * 70)
    print("RETRIEVING IONQ JOBS")
    print("=" * 70)

    if not JOBS:
        print("JOBS list is empty. Add entries with your IonQ job_id values (see comments in this file).")
        return

    provider = IonQProvider(get_ionq_api_key())
    backend = provider.get_backend('ionq_qpu.forte-1') # Assuming both are on the same backend or accessible via provider
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    for job_info in JOBS:
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
            
        # Set seeds for reproducibility (matches submit_ionq_native.py)
        seed = 42 + trial * 100 + degree
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Train neural network
        model = PolynomialNN(degree=degree)
        train_nn(model, target_func, CONFIG, verbose=False)
        learned_coeffs = model.get_coefficients()
        
        print(f"  Learned coeffs: {learned_coeffs}")
        
        # Generate test points (matches submit_ionq_native.py)
        x_values = np.linspace(CONFIG['x_range'][0], CONFIG['x_range'][1], CONFIG['num_samples'])
        
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
            
            # IonQ retrieve_job sometimes returns a single dict if only 1 circuit, 
            # or a list if multiple. We submitted 25 circuits.
            if isinstance(counts_list, dict):
                counts_list = [counts_list]
                
            if len(counts_list) != len(x_values):
                print(f"  WARNING: Mismatch in counts length. Expected {len(x_values)}, got {len(counts_list)}")
                # Proceeding with caution, mapping index-wise
            
            # 3. Process Results
            measured = np.zeros(len(x_values))
            measured_err = np.zeros(len(x_values))
            
            for i, counts in enumerate(counts_list):
                if i >= len(x_values): break
                
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
            
            exp_name = f"deg{degree}_trial{trial:02d}_ionq"
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
                'backend': job_info['backend_name'],
                'backend_name': job_info['backend_name'],
                'shots': CONFIG['shots'],
                'num_samples': CONFIG['num_samples'],
                'approach': 'native',
                'provider': 'ionq',
                'execution_time': 0, # Unknown from retrieval
                'metrics': {
                    'quantum_rmse': float(quantum_rmse),
                    'classical_rmse': float(classical_rmse),
                    'quantum_corr': float(quantum_corr),
                    'classical_corr': float(classical_corr),
                    'pass_rate': float(pass_rate),
                    'poor_rate': float(poor_rate),
                    'fail_rate': float(fail_rate),
                },
                'config': CONFIG,
                'timestamp': datetime.now().isoformat(),
                'job_id': job_id
            }
            
            write4_data_hdf5(h5_data, result_file, h5_meta)
            print(f"  Saved: {result_file}")
            
        except Exception as e:
            print(f"  ERROR retrieving/processing job: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    retrieve_and_process()
