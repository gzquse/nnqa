#!/usr/bin/env python3
"""
Estimate Cloud Job Resources
============================

Analyzes circuit structure to estimate:
1. Number of qubits per iteration
2. Number of one-qubit gates per iteration
3. Number of two-qubit gates per iteration
4. Number of shots per iteration
5. Number of iterations per job
6. Total resources for all jobs
"""

import numpy as np
from research_config import POLYNOMIALS, CLOUD_CONFIG


def analyze_direct_circuit(degree):
    """
    Analyze direct approach circuit (1 qubit).
    
    Direct approach:
    - Compute polynomial classically: y = F(x)
    - Encode result into single qubit: Ry(arccos(y))
    - Measure: <Z> = y
    """
    return {
        'qubits': 1,
        'one_qubit_gates': 1,  # Just 1 RY gate
        'two_qubit_gates': 0,
        'ry_gates': 1,
        'rz_gates': 0,
        'cnot_gates': 0,
    }


def analyze_native_circuit(degree):
    """
    Analyze native quantum arithmetic circuit (multi-qubit).
    
    Circuit structure:
    - Qubit 0: Encode x (1 RY gate)
    - Qubit 1: Compute x^2 = x * x
      - 1 RY gate (copy x)
      - 1 RZ gate
      - 1 CNOT gate
    - Qubit 2: Compute x^3 = x^2 * x
      - 1 RY gate (copy x)
      - 1 RZ gate
      - 1 CNOT gate
    - Qubit k: Compute x^k = x^(k-1) * x
      - 1 RY gate (copy x)
      - 1 RZ gate
      - 1 CNOT gate
    """
    n_qubits = max(2, degree + 1)
    
    if degree == 0:
        n_one_qubit = 1
    elif degree == 1:
        n_one_qubit = 1
    else:
        n_one_qubit = 1 + 2 * (degree - 1)
    
    n_two_qubit = max(0, degree - 1)
    
    return {
        'qubits': n_qubits,
        'one_qubit_gates': n_one_qubit,
        'two_qubit_gates': n_two_qubit,
        'ry_gates': 1 + max(0, degree - 1),
        'rz_gates': max(0, degree - 1),
        'cnot_gates': max(0, degree - 1),
    }


def estimate_all_jobs():
    """Estimate resources for all cloud jobs."""
    
    config = CLOUD_CONFIG
    n_trials = config['trials']
    n_samples = config['num_samples']
    shots = config['shots']
    
    print("=" * 90)
    print("CLOUD JOB RESOURCE ESTIMATION")
    print("=" * 90)
    print(f"\nConfiguration:")
    print(f"  Jobs per degree: {n_trials}")
    print(f"  Iterations per job: {n_samples}")
    print(f"  Shots per iteration: {shots:,}")
    print(f"  Total jobs: {len(POLYNOMIALS)} degrees × {n_trials} jobs = {len(POLYNOMIALS) * n_trials}")
    print()
    
    # ========================================================================
    # TABLE 1: DIRECT APPROACH (1 QUBIT)
    # ========================================================================
    print("=" * 90)
    print("TABLE 1: DIRECT APPROACH (1 QUBIT)")
    print("=" * 90)
    print("Polynomial computed classically, result encoded into single qubit")
    print("-" * 90)
    print(f"{'Degree':<8} {'Qubits':<8} {'1Q Gates/Iter':<15} {'2Q Gates/Iter':<15} {'Iterations':<12} {'Shots/Iter':<12}")
    print("-" * 90)
    
    total_direct_iterations = 0
    total_direct_shots = 0
    
    for degree in sorted(POLYNOMIALS.keys()):
        direct = analyze_direct_circuit(degree)
        iterations = n_samples * n_trials
        total_direct_iterations += iterations
        total_direct_shots += shots * iterations
        
        print(f"{degree:<8} {direct['qubits']:<8} "
              f"{direct['one_qubit_gates']:<15} "
              f"{direct['two_qubit_gates']:<15} "
              f"{iterations:<12,} {shots:<12,}")
    
    print("-" * 90)
    print(f"{'TOTAL':<8} {'1':<8} {'1':<15} {'0':<15} "
          f"{total_direct_iterations:<12,} {shots:<12,}")
    print("=" * 90)
    
    # ========================================================================
    # TABLE 2: NATIVE APPROACH (MULTI-QUBIT)
    # ========================================================================
    print("\n" + "=" * 90)
    print("TABLE 2: NATIVE APPROACH (MULTI-QUBIT)")
    print("=" * 90)
    print("Uses quantum multiplication to compute powers natively")
    print("-" * 90)
    print(f"{'Degree':<8} {'Qubits':<8} {'1Q Gates/Iter':<15} {'2Q Gates/Iter':<15} {'Iterations':<12} {'Shots/Iter':<12}")
    print("-" * 90)
    
    total_native_iterations = 0
    total_native_shots = 0
    
    for degree in sorted(POLYNOMIALS.keys()):
        native = analyze_native_circuit(degree)
        iterations = n_samples * n_trials
        total_native_iterations += iterations
        total_native_shots += shots * iterations
        
        print(f"{degree:<8} {native['qubits']:<8} "
              f"{native['one_qubit_gates']:<15} "
              f"{native['two_qubit_gates']:<15} "
              f"{iterations:<12,} {shots:<12,}")
    
    print("-" * 90)
    print(f"{'TOTAL':<8} {'2-7':<8} {'1-11':<15} {'0-5':<15} "
          f"{total_native_iterations:<12,} {shots:<12,}")
    print("=" * 90)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Total jobs: {len(POLYNOMIALS) * n_trials}")
    print(f"Total iterations: {total_direct_iterations:,}")
    print(f"Shots per iteration: {shots:,}")
    print(f"Total shots: {total_direct_shots:,}")
    print()
    print("Direct approach (current implementation):")
    print(f"  - Qubits per iteration: 1 (constant)")
    print(f"  - 1-qubit gates per iteration: 1 (constant)")
    print(f"  - 2-qubit gates per iteration: 0 (constant)")
    print()
    print("Native approach (alternative):")
    print(f"  - Qubits per iteration: 2-7 (scales with degree)")
    print(f"  - 1-qubit gates per iteration: 1-11 (scales with degree)")
    print(f"  - 2-qubit gates per iteration: 0-5 (scales with degree)")
    print()
    
    return {
        'direct': {
            'qubits_per_iter': 1,
            'one_qubit_gates_per_iter': 1,
            'two_qubit_gates_per_iter': 0,
            'total_iterations': total_direct_iterations,
            'total_shots': total_direct_shots,
        },
        'native': {
            'qubits_per_iter_range': (2, 7),
            'one_qubit_gates_per_iter_range': (1, 11),
            'two_qubit_gates_per_iter_range': (0, 5),
            'total_iterations': total_native_iterations,
            'total_shots': total_native_shots,
        },
    }


if __name__ == '__main__':
    estimate_all_jobs()
