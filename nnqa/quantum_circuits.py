#!/usr/bin/env python3
"""
Quantum Circuit Primitives and Builders
========================================

This module provides quantum circuit primitives for neural-native 
quantum arithmetic operations.

Primitives:
- Weighted Sum: w*x0 + (1-w)*x1
- Multiplication: x0 * x1
- Controlled Rotation: coefficient application

Circuit Builders:
- QuantumPolynomialCircuit: Builds circuits for polynomial evaluation
- DeepQuantumCircuit: Builds layered circuits for deep NN mapping
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit_aer import AerSimulator


# =============================================================================
# Encoding Functions
# =============================================================================

def data_to_angle(x):
    """
    Convert data value x in [-1, 1] to rotation angle theta.
    
    Uses arccos encoding: theta = arccos(x)
    This encodes x into the Z-expectation value of a qubit rotated by Ry(theta).
    
    Args:
        x: Data value(s) in [-1, 1]
        
    Returns:
        Rotation angle(s) in [0, pi]
    """
    x = np.clip(x, -1.0 + 1e-6, 1.0 - 1e-6)
    return np.arccos(x)


def weight_to_alpha(w):
    """
    Convert weight w in [0, 1] to rotation angle alpha.
    
    Uses arccos encoding: alpha = arccos(1 - 2w)
    This encodes the weight for the weighted sum operation.
    
    Args:
        w: Weight value(s) in [0, 1]
        
    Returns:
        Rotation angle(s) in [0, pi]
    """
    w = np.clip(w, 1e-6, 1.0 - 1e-6)
    return np.arccos(1.0 - 2.0 * w)


def angle_to_weight(alpha):
    """
    Convert rotation angle alpha back to weight w.
    
    Inverse of weight_to_alpha: w = (1 - cos(alpha)) / 2
    
    Args:
        alpha: Rotation angle(s)
        
    Returns:
        Weight value(s) in [0, 1]
    """
    return (1.0 - np.cos(alpha)) / 2.0


# =============================================================================
# Quantum Circuit Primitives
# =============================================================================

def add_weighted_sum_block(qc, q0, q1, alpha, barrier=True):
    """
    Add quantum weighted sum block: computes w*x0 + (1-w)*x1
    
    This implements the quantum arithmetic for weighted sum where
    w = (1 - cos(alpha))/2.
    
    After this block, measuring q0 in Z-basis gives expectation value
    equal to w*x0 + (1-w)*x1.
    
    Args:
        qc: QuantumCircuit to add block to
        q0: Qubit index for x0 (becomes output qubit)
        q1: Qubit index for x1
        alpha: Rotation angle encoding the weight
        barrier: Whether to add barrier after block
        
    Circuit structure:
        q0: ──■──Ry(α/2)──X──Ry(-α/2)──
              │           │
        q1: ─Rz─X─────────■────────────
    """
    qc.rz(np.pi/2, q1)
    qc.cx(q0, q1)
    qc.ry(alpha/2, q0)
    qc.cx(q1, q0)
    qc.ry(-alpha/2, q0)
    if barrier:
        qc.barrier()


def add_multiplication_block(qc, q0, q1, barrier=True):
    """
    Add quantum multiplication block: computes x0 * x1
    
    After this block, measuring q1 in Z-basis gives expectation value
    equal to x0 * x1.
    
    Args:
        qc: QuantumCircuit to add block to
        q0: Qubit index for x0
        q1: Qubit index for x1 (becomes output qubit)
        barrier: Whether to add barrier after block
        
    Circuit structure:
        q0: ──■──────
              │
        q1: ─Rz─X────
    """
    qc.rz(np.pi/2, q1)
    qc.cx(q0, q1)
    if barrier:
        qc.barrier()


def add_controlled_rotation(qc, control, target, angle, axis='y', barrier=False):
    """
    Add controlled rotation block for coefficient encoding.
    
    Implements: Ry(angle/2) - CX - Ry(-angle/2) pattern
    
    Args:
        qc: QuantumCircuit
        control: Control qubit index
        target: Target qubit index
        angle: Rotation angle
        axis: Rotation axis ('x', 'y', 'z')
        barrier: Whether to add barrier
    """
    if axis == 'y':
        qc.ry(angle/2, target)
        qc.cx(control, target)
        qc.ry(-angle/2, target)
    elif axis == 'x':
        qc.rx(angle/2, target)
        qc.cx(control, target)
        qc.rx(-angle/2, target)
    elif axis == 'z':
        qc.rz(angle/2, target)
        qc.cx(control, target)
        qc.rz(-angle/2, target)
        
    if barrier:
        qc.barrier()


# =============================================================================
# Quantum Polynomial Circuit Builder
# =============================================================================

class QuantumPolynomialCircuit:
    """
    Builds quantum circuits for polynomial evaluation using quantum arithmetic.
    
    For polynomial F(x) = a_n*x^n + ... + a_1*x + a_0:
    
    1. Data Encoding Layer: encode x into qubit rotations
    2. Power Computation Layer: compute x^2, x^3, ... using multiplication
    3. Coefficient Weighting Layer: apply coefficients as weighted sums
    4. Aggregation Layer: combine terms into final result
    
    Attributes:
        degree: Polynomial degree
        num_data_qubits: Qubits for data encoding
        num_basis_qubits: Qubits for coefficient encoding
        num_ancilla: Ancilla qubits for intermediate computations
    """
    
    def __init__(self, degree=3):
        """
        Args:
            degree: Maximum polynomial degree
        """
        self.degree = degree
        self.num_data_qubits = degree + 1
        self.num_basis_qubits = degree + 1
        self.num_ancilla = 2
        self.total_qubits = (self.num_data_qubits + 
                            self.num_basis_qubits + 
                            self.num_ancilla)
        
    def build_circuit(self, coefficients, x_value=None):
        """
        Build quantum circuit for polynomial with given coefficients.
        
        Args:
            coefficients: Array of [a_0, a_1, ..., a_n]
            x_value: Optional input value (if None, uses parameter)
            
        Returns:
            tuple: (QuantumCircuit, x_parameter or None)
        """
        # Normalize coefficients to [0, 1] for quantum encoding
        coeff_arr = np.array(coefficients)
        coeff_max = np.max(np.abs(coeff_arr)) + 1e-8
        coeff_normalized = (coeff_arr / coeff_max + 1) / 2
        
        # Create registers
        x_qubits = QuantumRegister(self.num_data_qubits, 'x')
        b_qubits = QuantumRegister(self.num_basis_qubits, 'b')
        anc_qubits = QuantumRegister(self.num_ancilla, 'anc')
        cr = ClassicalRegister(1, 'c')
        
        qc = QuantumCircuit(x_qubits, b_qubits, anc_qubits, cr)
        
        # Input handling
        if x_value is None:
            x_param = Parameter('x')
            theta_x = x_param
            use_param = True
        else:
            x_param = None
            theta_x = data_to_angle(x_value)
            use_param = False
        
        # === Layer 1: Data Encoding ===
        qc.ry(theta_x if use_param else 2*theta_x, x_qubits[0])
        qc.barrier()
        
        # === Layer 2: Power Computation ===
        for i in range(1, self.degree + 1):
            qc.ry(theta_x if use_param else 2*theta_x, x_qubits[i])
            if i > 1:
                add_multiplication_block(qc, x_qubits[i-1], x_qubits[i])
        
        # === Layer 3: Basis Initialization ===
        for i in range(self.num_basis_qubits):
            alpha = weight_to_alpha(coeff_normalized[i])
            qc.ry(alpha, b_qubits[i])
        qc.barrier()
        
        # === Layer 4: Coefficient Application ===
        for i in range(min(self.num_data_qubits, self.num_basis_qubits)):
            qc.rz(np.pi/2, b_qubits[i])
            qc.cx(x_qubits[i], b_qubits[i])
        qc.barrier()
        
        # === Layer 5: Term Aggregation ===
        for i in range(self.num_basis_qubits - 1):
            alpha_combine = np.pi / 4
            add_weighted_sum_block(qc, b_qubits[i], b_qubits[i+1], alpha_combine)
        
        # === Layer 6: Measurement ===
        qc.measure(b_qubits[-1], cr[0])
        
        return qc, x_param
    
    def build_parameterized_circuit(self):
        """
        Build fully parameterized quantum polynomial circuit.
        
        Returns:
            tuple: (QuantumCircuit, dict of ParameterVectors)
        """
        # Create registers
        x_qubits = QuantumRegister(self.num_data_qubits, 'x')
        b_qubits = QuantumRegister(self.num_basis_qubits, 'b')
        anc_qubits = QuantumRegister(self.num_ancilla, 'anc')
        cr = ClassicalRegister(1, 'c')
        
        qc = QuantumCircuit(x_qubits, b_qubits, anc_qubits, cr)
        
        # Parameters
        theta_input = ParameterVector('theta', self.num_data_qubits)
        alpha_coeff = ParameterVector('alpha', self.num_basis_qubits)
        beta_combine = ParameterVector('beta', self.num_basis_qubits - 1)
        
        # === Data Encoding Layer ===
        for i in range(self.num_data_qubits):
            qc.ry(theta_input[i], x_qubits[i])
        qc.barrier()
        
        # === Power Computation Layer ===
        for i in range(1, self.num_data_qubits):
            qc.rz(np.pi/2, x_qubits[i])
            qc.cx(x_qubits[i-1], x_qubits[i])
        qc.barrier()
        
        # === Coefficient Layer ===
        for i in range(self.num_basis_qubits):
            qc.ry(alpha_coeff[i], b_qubits[i])
        qc.barrier()
        
        # === Entanglement Layer ===
        for i in range(min(self.num_data_qubits, self.num_basis_qubits)):
            qc.rz(np.pi/2, b_qubits[i])
            qc.cx(x_qubits[i], b_qubits[i])
        qc.barrier()
        
        # === Aggregation Layer ===
        for i in range(self.num_basis_qubits - 1):
            qc.ry(beta_combine[i]/2, b_qubits[i])
            qc.cx(b_qubits[i+1], b_qubits[i])
            qc.ry(-beta_combine[i]/2, b_qubits[i])
        qc.barrier()
        
        # === Measurement ===
        qc.measure(b_qubits[-1], cr[0])
        
        params = {
            'theta': theta_input,
            'alpha': alpha_coeff,
            'beta': beta_combine
        }
        
        return qc, params
    
    def get_circuit_info(self):
        """Get circuit structure information."""
        return {
            'degree': self.degree,
            'total_qubits': self.total_qubits,
            'data_qubits': self.num_data_qubits,
            'basis_qubits': self.num_basis_qubits,
            'ancilla_qubits': self.num_ancilla,
            'num_theta_params': self.num_data_qubits,
            'num_alpha_params': self.num_basis_qubits,
            'num_beta_params': self.num_basis_qubits - 1,
        }


class DeepQuantumCircuit:
    """
    Builds layered quantum circuits for deep neural network mapping.
    
    Each layer corresponds to a NN layer with:
    - Input encoding
    - Parameterized rotations (weights)
    - Entanglement (CNOT connections)
    - Aggregation (pooling)
    """
    
    def __init__(self, layer_dims, num_ancilla=2):
        """
        Args:
            layer_dims: List of layer dimensions matching NN architecture
            num_ancilla: Number of ancilla qubits
        """
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1
        self.max_dim = max(layer_dims)
        self.num_ancilla = num_ancilla
        self.total_qubits = self.max_dim + num_ancilla
        
    def build_circuit(self, weights_dict):
        """
        Build quantum circuit from NN weights.
        
        Args:
            weights_dict: Dictionary of weight matrices from NN
            
        Returns:
            QuantumCircuit
        """
        qr = QuantumRegister(self.total_qubits, 'q')
        cr = ClassicalRegister(1, 'c')
        qc = QuantumCircuit(qr, cr)
        
        # Process each layer
        layer_idx = 0
        for name, weights in weights_dict.items():
            if 'weight' in name and 'normalized' not in name:
                # Normalize weights to angles
                w_flat = weights.flatten()
                w_norm = (w_flat - w_flat.min()) / (w_flat.max() - w_flat.min() + 1e-8)
                
                # Apply rotations
                for i, w in enumerate(w_norm[:self.max_dim]):
                    alpha = weight_to_alpha(w)
                    qc.ry(alpha, qr[i])
                
                # Add entanglement
                for i in range(min(len(w_norm), self.max_dim) - 1):
                    qc.cx(qr[i], qr[i+1])
                
                qc.barrier()
                layer_idx += 1
        
        # Measurement
        qc.measure(qr[0], cr[0])
        
        return qc
    
    def build_variational_circuit(self, num_params_per_layer=None):
        """
        Build variational circuit with trainable parameters.
        
        Args:
            num_params_per_layer: Parameters per layer (default: max_dim)
            
        Returns:
            tuple: (QuantumCircuit, dict of parameters)
        """
        if num_params_per_layer is None:
            num_params_per_layer = self.max_dim
            
        qr = QuantumRegister(self.total_qubits, 'q')
        cr = ClassicalRegister(1, 'c')
        qc = QuantumCircuit(qr, cr)
        
        all_params = {}
        
        for layer in range(self.num_layers):
            layer_params = ParameterVector(f'L{layer}', num_params_per_layer)
            all_params[f'layer_{layer}'] = layer_params
            
            # Rotation layer
            for i in range(min(num_params_per_layer, self.max_dim)):
                qc.ry(layer_params[i], qr[i])
            
            # Entanglement layer
            for i in range(self.max_dim - 1):
                qc.cx(qr[i], qr[i+1])
            
            qc.barrier()
        
        qc.measure(qr[0], cr[0])
        
        return qc, all_params


# =============================================================================
# Execution Utilities
# =============================================================================

class CircuitExecutor:
    """
    Utility class for executing quantum circuits and extracting results.
    """
    
    def __init__(self, backend=None, shots=4096):
        """
        Args:
            backend: Qiskit backend (default: AerSimulator)
            shots: Number of measurement shots
        """
        self.backend = backend or AerSimulator()
        self.shots = shots
        
    def execute(self, circuit, parameter_values=None):
        """
        Execute circuit and return expectation value.
        
        Args:
            circuit: QuantumCircuit to execute
            parameter_values: Dict of parameter bindings
            
        Returns:
            float: Z-expectation value
        """
        # Bind parameters if provided
        if parameter_values:
            circuit = circuit.assign_parameters(parameter_values)
        
        # Transpile and run
        qc_t = transpile(circuit, self.backend, optimization_level=1)
        job = self.backend.run(qc_t, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        
        # Calculate expectation value
        n0 = counts.get('0', 0)
        n1 = counts.get('1', 0)
        exp_val = (n0 - n1) / self.shots
        
        return exp_val
    
    def execute_batch(self, circuit, parameter_list):
        """
        Execute circuit for multiple parameter sets.
        
        Args:
            circuit: QuantumCircuit
            parameter_list: List of parameter dicts
            
        Returns:
            list: Expectation values for each parameter set
        """
        results = []
        for params in parameter_list:
            exp_val = self.execute(circuit, params)
            results.append(exp_val)
        return results

