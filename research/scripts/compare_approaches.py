#!/usr/bin/env python3
"""
Compare Resource Usage: Native vs Direct Approach
=================================================

Compares:
1. Native approach: Uses quantum multiplication (multiple qubits)
2. Direct approach: Classical computation + single qubit encoding (1 qubit)
"""

import numpy as np
from research_config import POLYNOMIALS, CLOUD_CONFIG


def analyze_native_circuit(degree):
    """Analyze native quantum arithmetic circuit (multi-qubit)."""
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


def compare_approaches():
    """Compare native vs direct approaches."""
    
    config = CLOUD_CONFIG
    n_trials = config['trials']
    n_samples = config['num_samples']
    shots = config['shots']
    
    print("=" * 90)
    print("RESOURCE COMPARISON: NATIVE vs DIRECT APPROACH")
    print("=" * 90)
    print(f"\nConfiguration:")
    print(f"  Jobs per degree: {n_trials}")
    print(f"  Iterations per job: {n_samples}")
    print(f"  Shots per iteration: {shots:,}")
    print()
    
    print("-" * 90)
    print(f"{'Degree':<8} {'Approach':<12} {'Qubits':<8} {'1Q Gates':<12} {'2Q Gates':<12} {'Iterations':<12} {'Total Shots':<15}")
    print("-" * 90)
    
    total_native_1q = 0
    total_native_2q = 0
    total_direct_1q = 0
    total_direct_2q = 0
    
    for degree in sorted(POLYNOMIALS.keys()):
        native = analyze_native_circuit(degree)
        direct = analyze_direct_circuit(degree)
        
        native_total_1q = native['one_qubit_gates'] * n_samples * n_trials
        native_total_2q = native['two_qubit_gates'] * n_samples * n_trials
        direct_total_1q = direct['one_qubit_gates'] * n_samples * n_trials
        direct_total_2q = direct['two_qubit_gates'] * n_samples * n_trials
        
        total_native_1q += native_total_1q
        total_native_2q += native_total_2q
        total_direct_1q += direct_total_1q
        total_direct_2q += direct_total_2q
        
        iterations = n_samples * n_trials
        total_shots = shots * iterations
        
        print(f"{degree:<8} {'Native':<12} {native['qubits']:<8} "
              f"{native_total_1q:<12,} {native_total_2q:<12,} "
              f"{iterations:<12,} {total_shots:<15,}")
        print(f"{'':<8} {'Direct':<12} {direct['qubits']:<8} "
              f"{direct_total_1q:<12,} {direct_total_2q:<12,} "
              f"{iterations:<12,} {total_shots:<15,}")
        print()
    
    print("-" * 90)
    print(f"{'TOTAL':<8} {'Native':<12} {'-':<8} "
          f"{total_native_1q:<12,} {total_native_2q:<12,} "
          f"{'-':<12} {'-':<15}")
    print(f"{'':<8} {'Direct':<12} {'-':<8} "
          f"{total_direct_1q:<12,} {total_direct_2q:<12,} "
          f"{'-':<12} {'-':<15}")
    print("=" * 90)
    
    # Detailed per-iteration comparison
    print("\n" + "=" * 90)
    print("PER ITERATION COMPARISON")
    print("=" * 90)
    print(f"{'Degree':<8} {'Approach':<12} {'Qubits':<8} {'1Q Gates':<10} {'2Q Gates':<10} {'Shots/Iter':<12}")
    print("-" * 90)
    
    for degree in sorted(POLYNOMIALS.keys()):
        native = analyze_native_circuit(degree)
        direct = analyze_direct_circuit(degree)
        
        print(f"{degree:<8} {'Native':<12} {native['qubits']:<8} "
              f"{native['one_qubit_gates']:<10} {native['two_qubit_gates']:<10} {shots:<12,}")
        print(f"{'':<8} {'Direct':<12} {direct['qubits']:<8} "
              f"{direct['one_qubit_gates']:<10} {direct['two_qubit_gates']:<10} {shots:<12,}")
        print()
    
    # Savings analysis
    print("\n" + "=" * 90)
    print("RESOURCE SAVINGS: DIRECT vs NATIVE")
    print("=" * 90)
    print(f"{'Degree':<8} {'Qubit Reduction':<18} {'1Q Gate Reduction':<20} {'2Q Gate Reduction':<20}")
    print("-" * 90)
    
    for degree in sorted(POLYNOMIALS.keys()):
        native = analyze_native_circuit(degree)
        direct = analyze_direct_circuit(degree)
        
        qubit_reduction = native['qubits'] - direct['qubits']
        gate1_reduction = native['one_qubit_gates'] - direct['one_qubit_gates']
        gate2_reduction = native['two_qubit_gates'] - direct['two_qubit_gates']
        
        print(f"{degree:<8} {qubit_reduction:<18} {gate1_reduction:<20} {gate2_reduction:<20}")
    
    print("-" * 90)
    print("\nKey Insight:")
    print("  Direct approach uses 1 qubit and 1 gate per iteration (constant)")
    print("  Native approach scales with degree (linear qubit and gate scaling)")
    print("  Both use same number of shots (4,096 per iteration)")
    print()


if __name__ == '__main__':
    compare_approaches()

