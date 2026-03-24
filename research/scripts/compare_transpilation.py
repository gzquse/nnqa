
#!/usr/bin/env python3
"""
Compare Transpilation: IBM Miami (Nighthawk) vs IBM Boston (Heron)
==================================================================

Compares the transpiled circuit depth and gate counts for Native Polynomial
circuits on different IBM Quantum architectures.

Nighthawk (Miami) has higher connectivity (square lattice, 218 couplers for 120 qubits)
vs Heron (Boston) which has standard heavy-hex/tunable coupler topology.

Hypothesis: Miami should require fewer SWAPs/2-qubit gates for the all-to-all 
connectivity required by the native polynomial circuit (controlled rotations).
"""

import sys
import os
import numpy as np
import torch
from qiskit import transpile
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService

# Add parent paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research.submit_cloud_batch import build_polynomial_circuit, PolynomialNN, train_nn
from research.research_config import POLYNOMIALS, CLOUD_CONFIG, evaluate_polynomial

def get_circuit(degree=5):
    """Generate a representative circuit for a specific degree."""
    poly_info = POLYNOMIALS[degree]
    true_coeffs = np.array(poly_info['coefficients'])
    
    # Quick train to get realistic coefficients
    def target_func(x):
        return evaluate_polynomial(x, true_coeffs)
    
    model = PolynomialNN(degree=degree)
    # Train briefly just to get valid-looking coeffs
    train_nn(model, target_func, {'train_lr': 0.1, 'train_samples': 50, 'train_epochs': 50})
    learned_coeffs = model.get_coefficients()
    
    # Build circuit for a random x
    qc, _ = build_polynomial_circuit(0.5, learned_coeffs)
    return qc

def compare_backends(degrees=[3, 4, 5, 6]):
    service = QiskitRuntimeService()
    
    # Use fake providers to avoid connecting to real backends if they are busy/unavailable
    # Or just use the real service to get configuration if we aren't running jobs
    
    # We will try to get the real backend object first
    # If not available, we can't do the comparison accurately without the backend properties
    
    # If accessing real backends fails, we can use fake backends to simulate the topology
    # if available, or just skip.
    # Given the previous error 'No backend matches the criteria', it's possible 
    # the user account doesn't have access to 'ibm_boston' specifically under that name
    # or is in a different hub/group/project.
    
    # Let's try listing available backends first to be sure
    available_backends = service.backends()
    available_names = [b.name for b in available_backends]
    print(f"Available backends: {available_names}")
    
    # Fallback mappings if specific names not found
    # We want a Heron device (heavy-hex) and Nighthawk (square/heavy-square)
    # If Miami not found, check available list.
    
    backends_to_compare = []
    if 'ibm_miami' in available_names:
        backends_to_compare.append('ibm_miami')
    if 'ibm_boston' in available_names:
        backends_to_compare.append('ibm_boston')
    
    # If empty, try to pick representative ones
    if not backends_to_compare:
        print("Requested backends not found. Using available ones.")
        # Pick top 2 by qubit count
        sorted_backends = sorted(available_backends, key=lambda x: x.num_qubits, reverse=True)
        backends_to_compare = [b.name for b in sorted_backends[:2]]
    
    print(f"Comparing backends: {backends_to_compare}")
    
    print(f"{'Degree':<8} | {'Backend':<12} | {'Original 2q':<12} | {'Transpiled 2q':<14} | {'Depth':<8} | {'Total Gates':<12}")
    print("-" * 85)
    
    results = {}
    
    for degree in degrees:
        print(f"Generating circuit for Degree {degree}...")
        qc = get_circuit(degree)
        # Decompose high-level gates to see 'ideal' 2-qubit count
        qc_decomp = qc.decompose()
        orig_ops = qc_decomp.count_ops()
        orig_2q = orig_ops.get('cx', 0) + orig_ops.get('cz', 0) + orig_ops.get('ecr', 0)
        
        results[degree] = {'original_2q': orig_2q}
        
        for b_name in backends_to_compare:
            try:
                # Retrieve backend
                backend = service.backend(b_name)
                
                # Generate pass manager for this backend
                # optimization_level=3 includes heavy optimization and mapping
                pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
                qc_transpiled = pm.run(qc)
                
                ops = qc_transpiled.count_ops()
                # Sum common 2-qubit gates
                transpiled_2q = ops.get('cx', 0) + ops.get('cz', 0) + ops.get('ecr', 0) + ops.get('rzx', 0)
                depth = qc_transpiled.depth()
                total = sum(ops.values())
                
                print(f"{degree:<8} | {b_name:<12} | {orig_2q:<12} | {transpiled_2q:<14} | {depth:<8} | {total:<12}")
                
                if b_name not in results[degree]:
                    results[degree][b_name] = {}
                results[degree][b_name] = {
                    '2q': transpiled_2q,
                    'depth': depth,
                    'total': total
                }
                
            except Exception as e:
                print(f"Error with {b_name}: {e}")

    return results

if __name__ == "__main__":
    compare_backends()
