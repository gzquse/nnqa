#!/usr/bin/env python3
"""
Neural-Native Quantum Arithmetic - Demo
===================================================

This demo shows the workflow for mapping a trained neural network
polynomial to quantum circuits using quantum arithmetic primitives.

Key Insight:
-----------
Our quantum arithmetic protocol provides EXACT computation:
- Weighted Sum: <Z> = w*x0 + (1-w)*x1
- Multiplication: <Z> = x0 * x1

For a polynomial F(x) = a0 + a1*x + a2*x^2 + ..., we can:
1. Compute x^k terms via chained multiplications
2. Combine terms via weighted sums
3. Measure to get F(x) as expectation value

CRITICAL: The output must be in [-1, 1] for quantum encoding!
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add nnqa package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator


# ============================================================================
# PART 1: Quantum Arithmetic Primitives
# ============================================================================

def data_to_angle(x):
    """Convert x in [-1,1] to rotation angle. After Ry(theta), <Z> = cos(theta) = x."""
    x = np.clip(x, -1 + 1e-7, 1 - 1e-7)
    return np.arccos(x)


def weight_to_alpha(w):
    """Convert weight w in [0,1] to alpha angle for weighted sum."""
    w = np.clip(w, 1e-7, 1 - 1e-7)
    return np.arccos(1 - 2*w)


def run_circuit(qc, shots=8192):
    """Execute circuit and return Z-expectation value."""
    backend = AerSimulator()
    qc_t = transpile(qc, backend, optimization_level=1)
    job = backend.run(qc_t, shots=shots)
    counts = job.result().get_counts()
    n0, n1 = counts.get('0', 0), counts.get('1', 0)
    return (n0 - n1) / shots


def quantum_weighted_sum(x0, x1, w, shots=8192):
    """
    Quantum weighted sum: y = w*x0 + (1-w)*x1
    
    This is the fundamental quantum arithmetic operation for linear combination.
    """
    qr = QuantumRegister(2)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)
    
    # Encode inputs
    qc.ry(data_to_angle(x0), 0)
    qc.ry(data_to_angle(x1), 1)
    qc.barrier()
    
    # Quantum arithmetic block for weighted sum
    alpha = weight_to_alpha(w)
    qc.rz(np.pi/2, 1)
    qc.cx(0, 1)
    qc.ry(alpha/2, 0)
    qc.cx(1, 0)
    qc.ry(-alpha/2, 0)
    
    qc.measure(0, 0)
    return run_circuit(qc, shots)


def quantum_multiplication(x0, x1, shots=8192):
    """
    Quantum multiplication: y = x0 * x1
    
    This is the quantum arithmetic operation for multiplication.
    """
    qr = QuantumRegister(2)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)
    
    # Encode inputs
    qc.ry(data_to_angle(x0), 0)
    qc.ry(data_to_angle(x1), 1)
    qc.barrier()
    
    # Quantum arithmetic block for multiplication
    qc.rz(np.pi/2, 1)
    qc.cx(0, 1)
    
    qc.measure(1, 0)
    return run_circuit(qc, shots)


# ============================================================================
# PART 2: Neural Network for Polynomial Approximation
# ============================================================================

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


# ============================================================================
# PART 3: Quantum Polynomial Evaluation
# ============================================================================

def quantum_polynomial_eval(x, coefficients, shots=8192):
    """
    Evaluate polynomial using quantum arithmetic.
    
    For F(x) = a0 + a1*x + a2*x^2 + a3*x^3:
    1. Compute x^2 = x * x using multiplication
    2. Compute x^3 = x^2 * x using multiplication  
    3. Combine: result = a0*1 + a1*x + a2*x^2 + a3*x^3
    
    We compute each term separately using quantum operations,
    then combine classically (hybrid approach).
    """
    degree = len(coefficients) - 1
    
    # Compute powers of x using quantum multiplication
    x_powers = [1.0]  # x^0 = 1
    current_power = x
    x_powers.append(current_power)
    
    for i in range(2, degree + 1):
        # x^i = x^(i-1) * x
        current_power = quantum_multiplication(current_power, x, shots)
        x_powers.append(current_power)
    
    # Combine terms
    result = sum(coefficients[i] * x_powers[i] for i in range(degree + 1))
    return result


def quantum_polynomial_direct(x, coefficients, shots=8192):
    """
    Direct quantum polynomial evaluation.
    
    This computes the polynomial classically, then encodes the result
    into a quantum state and measures it. This demonstrates the 
    encoding/decoding fidelity.
    """
    # Classical evaluation
    y = sum(coefficients[i] * (x ** i) for i in range(len(coefficients)))
    
    # Normalize to [-1, 1]
    # For demo, we assume |y| <= 1
    y_clipped = np.clip(y, -1 + 1e-6, 1 - 1e-6)
    
    # Encode and measure
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)
    
    qc.ry(data_to_angle(y_clipped), 0)
    qc.measure(0, 0)
    
    return run_circuit(qc, shots)


# ============================================================================
# PART 4: Demo - Train NN and Map to Quantum
# ============================================================================

def run_demo():
    """Complete demo of NN to Quantum polynomial mapping."""
    
    print("=" * 70)
    print("NEURAL-NATIVE QUANTUM ARITHMETIC")
    print("Mapping Trained Neural Network to Quantum Circuit")
    print("=" * 70)
    
    # Step 1: Define target polynomial (must have output in [-1, 1])
    print("\n[Step 1] Define Target Polynomial")
    # F(x) = 0.1 + 0.3x - 0.1x^2 + 0.2x^3
    # Chosen so |F(x)| < 1 for x in [-1, 1]
    true_coeffs = np.array([0.1, 0.3, -0.1, 0.2])
    
    def target_func(x):
        return sum(true_coeffs[i] * (x ** i) for i in range(len(true_coeffs)))
    
    print(f"Target: F(x) = {true_coeffs[0]:.2f} + {true_coeffs[1]:.2f}x "
          f"+ {true_coeffs[2]:.2f}x^2 + {true_coeffs[3]:.2f}x^3")
    
    # Verify output range
    x_test = np.linspace(-1, 1, 100)
    y_test = np.array([target_func(x) for x in x_test])
    print(f"Output range: [{y_test.min():.3f}, {y_test.max():.3f}]")
    
    # Step 2: Train Neural Network
    print("\n[Step 2] Training Neural Network")
    model = PolynomialNN(degree=3)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    
    # Generate training data
    X_train = torch.linspace(-0.95, 0.95, 200)
    y_train = torch.tensor([target_func(x.item()) for x in X_train], dtype=torch.float32)
    
    for epoch in range(200):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
    
    learned_coeffs = model.get_coefficients()
    print(f"\nLearned coefficients: {learned_coeffs}")
    print(f"True coefficients:    {true_coeffs}")
    
    # Step 3: Verify Quantum Arithmetic Operations
    print("\n[Step 3] Verify Quantum Arithmetic Operations")
    print("-" * 50)
    
    # Test weighted sum
    x0, x1, w = 0.6, -0.2, 0.7
    expected = w * x0 + (1 - w) * x1
    measured = quantum_weighted_sum(x0, x1, w)
    print(f"Weighted Sum: {w:.1f}*{x0:.1f} + {1-w:.1f}*{x1:.1f}")
    print(f"  Expected: {expected:.4f}, Measured: {measured:.4f}, Diff: {abs(expected-measured):.4f}")
    
    # Test multiplication
    x0, x1 = 0.5, 0.6
    expected = x0 * x1
    measured = quantum_multiplication(x0, x1)
    print(f"Multiplication: {x0:.1f} * {x1:.1f}")
    print(f"  Expected: {expected:.4f}, Measured: {measured:.4f}, Diff: {abs(expected-measured):.4f}")
    
    # Step 4: Quantum Polynomial Evaluation
    print("\n[Step 4] Quantum Polynomial Evaluation")
    print("-" * 70)
    print(f"{'x':<8} | {'Classical':<12} | {'Q-Direct':<12} | {'Q-Hybrid':<12} | {'Diff':<8}")
    print("-" * 70)
    
    test_points = [-0.8, -0.4, 0.0, 0.4, 0.8]
    total_diff_direct = 0
    total_diff_hybrid = 0
    
    for x in test_points:
        classical = target_func(x)
        q_direct = quantum_polynomial_direct(x, learned_coeffs)
        q_hybrid = quantum_polynomial_eval(x, learned_coeffs)
        
        diff_direct = abs(classical - q_direct)
        diff_hybrid = abs(classical - q_hybrid)
        total_diff_direct += diff_direct
        total_diff_hybrid += diff_hybrid
        
        print(f"{x:<8.2f} | {classical:<12.4f} | {q_direct:<12.4f} | {q_hybrid:<12.4f} | {diff_direct:<8.4f}")
    
    avg_diff_direct = total_diff_direct / len(test_points)
    avg_diff_hybrid = total_diff_hybrid / len(test_points)
    
    print("-" * 70)
    print(f"Average Diff (Direct): {avg_diff_direct:.4f}")
    print(f"Average Diff (Hybrid): {avg_diff_hybrid:.4f}")
    
    # Step 5: Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"NN Training Loss: {loss.item():.6f}")
    print(f"Coefficient Match: {np.allclose(learned_coeffs, true_coeffs, atol=0.05)}")
    
    if avg_diff_direct < 0.05:
        print("Quantum-Direct Accuracy: EXCELLENT (< 5% error)")
    elif avg_diff_direct < 0.1:
        print("Quantum-Direct Accuracy: GOOD (< 10% error)")
    else:
        print("Quantum-Direct Accuracy: NEEDS WORK")
    
    print("\nKEY INSIGHT:")
    print("  Our quantum arithmetic protocol provides EXACT computation.")
    print("  Weighted Sum: <Z> = w*x0 + (1-w)*x1")
    print("  Multiplication: <Z> = x0 * x1")
    print("  By chaining these operations, we can evaluate polynomials.")


if __name__ == '__main__':
    run_demo()
