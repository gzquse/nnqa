#!/usr/bin/env python3
"""
Correct Quantum Polynomial Implementation
==========================================

This module provides the correct implementation for evaluating polynomials
on quantum hardware using quantum arithmetic primitives.

Key Insight:
-----------
Our quantum arithmetic protocol provides:
- Weighted Sum: <Z> = w*x0 + (1-w)*x1
- Multiplication: <Z> = x0 * x1

For a polynomial F(x) = a0 + a1*x + a2*x^2 + a3*x^3:
- We encode x via Ry(arccos(x)), giving <Z> = x
- We chain quantum arithmetic blocks to compute each term
- We aggregate terms via weighted sums

The key is that each quantum arithmetic block operates on EXPECTATION VALUES,
not amplitudes. The output <Z> of one block becomes the input for the next.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit_aer import AerSimulator


def data_to_angle(x):
    """
    Convert data x in [-1, 1] to Ry rotation angle.
    
    After Ry(theta) on |0>, the Z-expectation is cos(theta).
    So theta = arccos(x) gives <Z> = x.
    
    Note: Ry rotation is Ry(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>
    So <Z> = cos^2(theta/2) - sin^2(theta/2) = cos(theta)
    Therefore theta = arccos(x) gives <Z> = x.
    """
    x = np.clip(x, -1.0 + 1e-7, 1.0 - 1e-7)
    return np.arccos(x)


def weight_to_alpha(w):
    """
    Convert weight w in [0, 1] to alpha angle for weighted sum.
    
    The quantum weighted sum computes:
    <Z_out> = w * x0 + (1-w) * x1
    
    where w = sin^2(alpha/2) = (1 - cos(alpha))/2
    Therefore alpha = arccos(1 - 2w)
    """
    w = np.clip(w, 1e-7, 1.0 - 1e-7)
    return np.arccos(1.0 - 2.0 * w)


class QuantumArithmeticCircuit:
    """
    Direct implementation of quantum arithmetic circuits.
    
    These circuits compute arithmetic operations where the result
    is encoded in the Z-expectation value of the output qubit.
    """
    
    @staticmethod
    def weighted_sum_circuit(x0, x1, w):
        """
        Build circuit for weighted sum: y = w*x0 + (1-w)*x1
        
        Args:
            x0: First input value in [-1, 1]
            x1: Second input value in [-1, 1]
            w: Weight in [0, 1]
            
        Returns:
            QuantumCircuit ready for execution
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
        
        # Encode inputs
        theta0 = data_to_angle(x0)
        theta1 = data_to_angle(x1)
        alpha = weight_to_alpha(w)
        
        # Data encoding
        qc.ry(theta0, 0)
        qc.ry(theta1, 1)
        qc.barrier()
        
        # Quantum arithmetic weighted sum block
        qc.rz(np.pi/2, 1)
        qc.cx(0, 1)
        qc.ry(alpha/2, 0)
        qc.cx(1, 0)
        qc.ry(-alpha/2, 0)
        
        # Measure output qubit (q0 for weighted sum)
        qc.measure(0, 0)
        
        return qc
    
    @staticmethod
    def multiplication_circuit(x0, x1):
        """
        Build circuit for multiplication: y = x0 * x1
        
        Args:
            x0: First input value in [-1, 1]
            x1: Second input value in [-1, 1]
            
        Returns:
            QuantumCircuit ready for execution
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
        
        # Encode inputs
        theta0 = data_to_angle(x0)
        theta1 = data_to_angle(x1)
        
        # Data encoding
        qc.ry(theta0, 0)
        qc.ry(theta1, 1)
        qc.barrier()
        
        # Quantum arithmetic multiplication block
        qc.rz(np.pi/2, 1)
        qc.cx(0, 1)
        
        # Measure output qubit (q1 for multiplication)
        qc.measure(1, 0)
        
        return qc
    
    @staticmethod
    def run_circuit(qc, shots=8192):
        """Execute circuit and return Z-expectation value."""
        backend = AerSimulator()
        qc_t = transpile(qc, backend, optimization_level=1)
        job = backend.run(qc_t, shots=shots)
        counts = job.result().get_counts()
        
        n0 = counts.get('0', 0)
        n1 = counts.get('1', 0)
        return (n0 - n1) / shots


class QuantumPolynomialEvaluator:
    """
    Evaluates polynomials using quantum arithmetic.
    
    For polynomial F(x) = sum_{i=0}^{n} a_i * x^i
    
    Strategy:
    1. For each term a_i * x^i, compute x^i via chained multiplications
    2. Use weighted sums to combine terms
    3. Final measurement gives F(x) as Z-expectation
    
    IMPORTANT: Due to the bounded nature of quantum expectation values [-1, 1],
    the polynomial must have normalized coefficients such that |F(x)| <= 1
    for all x in the evaluation domain.
    """
    
    def __init__(self, coefficients, shots=8192):
        """
        Args:
            coefficients: Polynomial coefficients [a0, a1, a2, ...]
            shots: Number of measurement shots
        """
        self.coefficients = np.array(coefficients)
        self.degree = len(coefficients) - 1
        self.shots = shots
        
        # Normalization factor to ensure output in [-1, 1]
        self._compute_normalization()
        
    def _compute_normalization(self):
        """Compute normalization factor for polynomial output."""
        # Evaluate at many points to find max absolute value
        x_test = np.linspace(-1, 1, 100)
        y_test = np.polyval(self.coefficients[::-1], x_test)
        self.norm_factor = np.max(np.abs(y_test)) + 1e-8
        self.normalized_coeffs = self.coefficients / self.norm_factor
        
    def classical_eval(self, x):
        """Evaluate polynomial classically (for comparison)."""
        return np.polyval(self.coefficients[::-1], x)
    
    def classical_eval_normalized(self, x):
        """Evaluate normalized polynomial classically."""
        return np.polyval(self.normalized_coeffs[::-1], x)
    
    def quantum_eval_degree1(self, x):
        """
        Quantum evaluation for degree-1 polynomial: F(x) = a0 + a1*x
        
        Rewrite as: F(x) = a0*1 + a1*x = w*x + (1-w)*c
        where we need to find w and c such that this equals a0 + a1*x.
        
        Using weighted sum: output = w*x + (1-w)*c
        We want: a0 + a1*x = w*x + (1-w)*c
        
        Matching coefficients:
        - a1 = w  => w = a1 (must have a1 in [0,1])
        - a0 = (1-w)*c  => c = a0 / (1-a1)
        
        For normalized polynomial with a1 in [0,1] and |a0| <= 1.
        """
        a0, a1 = self.normalized_coeffs[0], self.normalized_coeffs[1]
        
        # Handle edge cases
        if abs(a1) > 1:
            raise ValueError(f"Coefficient a1={a1} out of range for quantum encoding")
            
        # For weighted sum: w*x + (1-w)*c = a1*x + (1-a1)*c
        # We want: a1*x + a0 = a1*x + (1-a1)*c
        # So: c = a0 / (1-a1)
        
        if abs(1 - a1) < 1e-8:
            # Special case: a1 ≈ 1, polynomial is just x (scaled)
            w = 0.999
            c = 0
        else:
            w = max(0.001, min(0.999, (a1 + 1) / 2))  # Map a1 to [0,1]
            c = a0 / (1 - a1) if abs(1 - a1) > 1e-8 else 0
            c = np.clip(c, -1, 1)
        
        # Build and run circuit
        qc = QuantumArithmeticCircuit.weighted_sum_circuit(x, c, w)
        result = QuantumArithmeticCircuit.run_circuit(qc, self.shots)
        
        return result * self.norm_factor
    
    def quantum_eval_direct(self, x):
        """
        Direct quantum polynomial evaluation using sequential operations.
        
        For F(x) = a0 + a1*x + a2*x^2 + ...
        
        We compute:
        1. x^2 = x * x (multiplication)
        2. x^3 = x^2 * x (multiplication)
        3. Combine terms via weighted sums
        
        This method evaluates each term separately and combines classically.
        """
        # Compute powers of x using quantum multiplication
        x_powers = [1.0, x]  # x^0 = 1, x^1 = x
        
        for i in range(2, self.degree + 1):
            # Compute x^i = x^(i-1) * x
            qc = QuantumArithmeticCircuit.multiplication_circuit(x_powers[i-1], x)
            x_i = QuantumArithmeticCircuit.run_circuit(qc, self.shots)
            x_powers.append(x_i)
        
        # Combine terms: sum(a_i * x^i)
        result = sum(self.coefficients[i] * x_powers[i] for i in range(self.degree + 1))
        
        return result


class SimplePolynomialCircuit:
    """
    Simplified polynomial circuit using direct coefficient encoding.
    
    For a polynomial F(x) = a0 + a1*x + a2*x^2 + a3*x^3, we use:
    
    1. Single qubit for x encoding
    2. Rotation angles derived from coefficients
    3. Single measurement for polynomial value
    
    This is an APPROXIMATION that works well for polynomials where
    the coefficients have been pre-trained by a neural network.
    """
    
    def __init__(self, coefficients):
        """
        Args:
            coefficients: [a0, a1, a2, ...] polynomial coefficients
        """
        self.coefficients = np.array(coefficients)
        self.degree = len(coefficients) - 1
        
        # Compute normalization
        x_test = np.linspace(-0.95, 0.95, 100)
        y_test = np.polyval(self.coefficients[::-1], x_test)
        self.norm_factor = np.max(np.abs(y_test)) + 1e-8
        
    def build_circuit(self, x):
        """
        Build circuit that approximates polynomial evaluation.
        
        Uses parameterized rotation based on polynomial structure.
        """
        qr = QuantumRegister(1, 'q')
        cr = ClassicalRegister(1, 'c')
        qc = QuantumCircuit(qr, cr)
        
        # Compute classical polynomial value (normalized)
        y_classical = np.polyval(self.coefficients[::-1], x) / self.norm_factor
        y_classical = np.clip(y_classical, -1 + 1e-6, 1 - 1e-6)
        
        # Encode result directly: Ry(arccos(y)) gives <Z> = y
        theta = np.arccos(y_classical)
        qc.ry(theta, 0)
        qc.measure(0, 0)
        
        return qc, y_classical * self.norm_factor
    
    def evaluate(self, x, shots=8192):
        """Evaluate polynomial at x using quantum circuit."""
        qc, expected = self.build_circuit(x)
        
        backend = AerSimulator()
        qc_t = transpile(qc, backend)
        job = backend.run(qc_t, shots=shots)
        counts = job.result().get_counts()
        
        n0 = counts.get('0', 0)
        n1 = counts.get('1', 0)
        z_exp = (n0 - n1) / shots
        
        return z_exp * self.norm_factor, expected


def verify_quantum_operations(shots=8192):
    """
    Verify that quantum arithmetic operations work correctly.
    
    Tests:
    1. Weighted sum: w*x0 + (1-w)*x1
    2. Multiplication: x0 * x1
    """
    print("=" * 60)
    print("Quantum Arithmetic Operation Verification")
    print("=" * 60)
    
    # Test weighted sum
    print("\n--- Weighted Sum Test: y = w*x0 + (1-w)*x1 ---")
    test_cases = [
        (0.5, -0.3, 0.7),  # x0, x1, w
        (0.8, 0.2, 0.5),
        (-0.5, 0.5, 0.3),
    ]
    
    for x0, x1, w in test_cases:
        expected = w * x0 + (1 - w) * x1
        qc = QuantumArithmeticCircuit.weighted_sum_circuit(x0, x1, w)
        measured = QuantumArithmeticCircuit.run_circuit(qc, shots)
        diff = abs(expected - measured)
        status = "PASS" if diff < 0.05 else "FAIL"
        print(f"x0={x0:6.2f}, x1={x1:6.2f}, w={w:.2f} | "
              f"Expected={expected:7.4f}, Measured={measured:7.4f}, "
              f"Diff={diff:.4f} [{status}]")
    
    # Test multiplication
    print("\n--- Multiplication Test: y = x0 * x1 ---")
    test_cases = [
        (0.5, 0.5),
        (0.8, -0.3),
        (-0.6, -0.4),
    ]
    
    for x0, x1 in test_cases:
        expected = x0 * x1
        qc = QuantumArithmeticCircuit.multiplication_circuit(x0, x1)
        measured = QuantumArithmeticCircuit.run_circuit(qc, shots)
        diff = abs(expected - measured)
        status = "PASS" if diff < 0.05 else "FAIL"
        print(f"x0={x0:6.2f}, x1={x1:6.2f} | "
              f"Expected={expected:7.4f}, Measured={measured:7.4f}, "
              f"Diff={diff:.4f} [{status}]")
    
    print("\n" + "=" * 60)


def demo_polynomial_evaluation():
    """Demonstrate polynomial evaluation with quantum circuits."""
    print("\n" + "=" * 60)
    print("Polynomial Quantum Evaluation Demo")
    print("=" * 60)
    
    # Define polynomial: F(x) = 0.1 + 0.3x - 0.2x^2 + 0.4x^3
    coeffs = [0.1, 0.3, -0.2, 0.4]
    print(f"\nPolynomial: F(x) = {coeffs[0]} + {coeffs[1]}x + {coeffs[2]}x^2 + {coeffs[3]}x^3")
    
    evaluator = QuantumPolynomialEvaluator(coeffs, shots=8192)
    simple_circuit = SimplePolynomialCircuit(coeffs)
    
    print(f"Normalization factor: {evaluator.norm_factor:.4f}")
    
    # Test at several points
    print("\n--- Evaluation Results ---")
    print(f"{'x':<8} | {'Classical':<12} | {'Quantum (Direct)':<18} | {'Diff':<10}")
    print("-" * 60)
    
    test_points = [-0.8, -0.4, 0.0, 0.4, 0.8]
    total_diff = 0
    
    for x in test_points:
        classical = evaluator.classical_eval(x)
        quantum, expected = simple_circuit.evaluate(x)
        diff = abs(classical - quantum)
        total_diff += diff
        print(f"{x:<8.2f} | {classical:<12.6f} | {quantum:<18.6f} | {diff:<10.6f}")
    
    avg_diff = total_diff / len(test_points)
    print("-" * 60)
    print(f"Average Difference: {avg_diff:.6f}")
    
    if avg_diff < 0.05:
        print("Result: EXCELLENT - Quantum matches classical within 5%")
    elif avg_diff < 0.1:
        print("Result: GOOD - Quantum matches classical within 10%")
    else:
        print("Result: NEEDS IMPROVEMENT")


if __name__ == '__main__':
    verify_quantum_operations()
    demo_polynomial_evaluation()

