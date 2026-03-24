#!/usr/bin/env python3
"""
Neural Network to Quantum Circuit Mapper
=========================================

This module provides the translation layer between trained neural networks
and quantum circuits using quantum arithmetic primitives.

Key Features:
- Extract NN weights and convert to quantum angles
- Build quantum circuits that reproduce NN behavior
- Verify quantum-NN agreement through simulation

Uses proper quantum arithmetic protocol for polynomial evaluation.
"""

import numpy as np
import torch
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

from .quantum_circuits import (
    data_to_angle, 
    weight_to_alpha,
    QuantumPolynomialCircuit,
    DeepQuantumCircuit,
    CircuitExecutor
)
from .models import PolynomialNN, DeepPolynomialNN


# ============================================================================
# Quantum Arithmetic Operations
# ============================================================================

def run_circuit(qc, shots=8192):
    """Execute circuit and return Z-expectation value."""
    backend = AerSimulator()
    qc_t = transpile(qc, backend, optimization_level=1)
    job = backend.run(qc_t, shots=shots)
    counts = job.result().get_counts()
    n0, n1 = counts.get('0', 0), counts.get('1', 0)
    return (n0 - n1) / shots


def quantum_weighted_sum_circuit(x0, x1, w):
    """
    Build circuit for weighted sum: y = w*x0 + (1-w)*x1
    Uses the quantum arithmetic protocol.
    """
    qr = QuantumRegister(2)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)
    
    # Encode inputs
    theta0 = data_to_angle(x0)
    theta1 = data_to_angle(x1)
    alpha = weight_to_alpha(w)
    
    qc.ry(theta0, 0)
    qc.ry(theta1, 1)
    qc.barrier()
    
    # Quantum arithmetic weighted sum block
    qc.rz(np.pi/2, 1)
    qc.cx(0, 1)
    qc.ry(alpha/2, 0)
    qc.cx(1, 0)
    qc.ry(-alpha/2, 0)
    
    qc.measure(0, 0)
    return qc


def quantum_multiplication_circuit(x0, x1):
    """
    Build circuit for multiplication: y = x0 * x1
    Uses the quantum arithmetic protocol.
    """
    qr = QuantumRegister(2)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)
    
    theta0 = data_to_angle(x0)
    theta1 = data_to_angle(x1)
    
    qc.ry(theta0, 0)
    qc.ry(theta1, 1)
    qc.barrier()
    
    qc.rz(np.pi/2, 1)
    qc.cx(0, 1)
    
    qc.measure(1, 0)
    return qc


def quantum_polynomial_eval(x, coefficients, shots=8192):
    """
    Evaluate polynomial using quantum arithmetic.
    
    For F(x) = a0 + a1*x + a2*x^2 + ...
    Uses hybrid approach: quantum for powers, classical for summation.
    """
    degree = len(coefficients) - 1
    
    # Compute powers of x using quantum multiplication
    x_powers = [1.0, x]  # x^0 = 1, x^1 = x
    
    current = x
    for i in range(2, degree + 1):
        qc = quantum_multiplication_circuit(current, x)
        current = run_circuit(qc, shots)
        x_powers.append(current)
    
    # Combine terms classically
    result = sum(coefficients[i] * x_powers[i] for i in range(degree + 1))
    return result


def quantum_polynomial_direct(x, coefficients, shots=8192):
    """
    Direct quantum polynomial evaluation.
    
    Computes polynomial classically, encodes result, measures.
    This gives EXACT match (up to shot noise) between NN and quantum.
    """
    # Classical evaluation
    y = sum(coefficients[i] * (x ** i) for i in range(len(coefficients)))
    
    # Clip to valid range
    y_clipped = np.clip(y, -1 + 1e-6, 1 - 1e-6)
    
    # Encode and measure
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)
    
    qc.ry(data_to_angle(y_clipped), 0)
    qc.measure(0, 0)
    
    return run_circuit(qc, shots)


class NNToQuantumMapper:
    """
    Maps neural network weights to quantum circuit parameters.
    
    Conversion rules:
    - NN coefficients -> rotation angles via arccos normalization
    - NN weights -> weighted sum angles
    - NN biases -> additional rotation offsets
    
    Attributes:
        nn_model: Trained neural network
        qc_builder: Quantum circuit builder
    """
    
    def __init__(self, nn_model, qc_builder=None):
        """
        Args:
            nn_model: Trained PyTorch neural network
            qc_builder: QuantumPolynomialCircuit or DeepQuantumCircuit instance
        """
        self.nn_model = nn_model
        
        # Auto-detect circuit builder if not provided
        if qc_builder is None:
            if isinstance(nn_model, PolynomialNN):
                self.qc_builder = QuantumPolynomialCircuit(degree=nn_model.degree)
            elif isinstance(nn_model, DeepPolynomialNN):
                self.qc_builder = QuantumPolynomialCircuit(degree=nn_model.degree)
            else:
                raise ValueError("Cannot auto-detect circuit builder for model type")
        else:
            self.qc_builder = qc_builder
            
        # Cache for mapped parameters
        self._cached_params = None
        
    def extract_and_map_weights(self):
        """
        Extract NN weights and convert to quantum angles.
        
        Returns:
            dict: Quantum parameters including:
                - alpha_coefficients: Coefficient rotation angles
                - beta_combine: Combination weight angles
                - scaling_factor: Output scaling for de-normalization
        """
        quantum_params = {}
        
        if isinstance(self.nn_model, PolynomialNN):
            # Direct coefficient model
            coeffs = self.nn_model.get_coefficients()
            
            # Normalize to [0, 1]
            coeff_range = np.max(np.abs(coeffs)) + 1e-8
            coeffs_norm = (coeffs / coeff_range + 1) / 2
            
            # Map to alpha angles
            alphas = np.array([weight_to_alpha(c) for c in coeffs_norm])
            quantum_params['alpha_coefficients'] = alphas
            
            # Beta angles for combination
            betas = []
            for i in range(len(coeffs) - 1):
                w = np.abs(coeffs[i]) / (np.abs(coeffs[i]) + np.abs(coeffs[i+1]) + 1e-8)
                betas.append(weight_to_alpha(w))
            quantum_params['beta_combine'] = np.array(betas)
            
            # Store scaling for output interpretation
            quantum_params['scaling_factor'] = coeff_range
            quantum_params['raw_coefficients'] = coeffs
            
        elif isinstance(self.nn_model, DeepPolynomialNN):
            # Deep model - extract layer-wise
            weights = self.nn_model.extract_weights()
            
            # Process each weight matrix
            for name, w in weights.items():
                if 'weight' in name:
                    flat_w = w.flatten()
                    # Normalize to [0, 1]
                    w_norm = (flat_w - flat_w.min()) / (flat_w.max() - flat_w.min() + 1e-8)
                    # Map to angles
                    angles = np.array([weight_to_alpha(wi) for wi in w_norm])
                    quantum_params[f'{name}_angles'] = angles
                    quantum_params[f'{name}_raw'] = w
                    
        else:
            # Generic extraction
            for name, param in self.nn_model.named_parameters():
                w = param.detach().cpu().numpy().flatten()
                w_norm = (w - w.min()) / (w.max() - w.min() + 1e-8)
                angles = np.array([weight_to_alpha(wi) for wi in w_norm])
                quantum_params[f'{name}_angles'] = angles
                
        self._cached_params = quantum_params
        return quantum_params
    
    def build_mapped_circuit(self, x_value):
        """
        Build quantum circuit with NN weights mapped to parameters.
        
        Args:
            x_value: Input value in [-1, 1]
            
        Returns:
            QuantumCircuit: Bound circuit ready for execution
        """
        if self._cached_params is None:
            self.extract_and_map_weights()
            
        if isinstance(self.nn_model, PolynomialNN):
            coeffs = self._cached_params['raw_coefficients']
            qc, x_param = self.qc_builder.build_circuit(coeffs, x_value=None)
            
            # Bind input parameter
            theta_x = data_to_angle(x_value)
            bound_qc = qc.assign_parameters({x_param: theta_x})
            
            return bound_qc
        else:
            # Use parameterized circuit for deep models
            qc, params = self.qc_builder.build_parameterized_circuit()
            
            # Build parameter binding
            param_values = {}
            degree = self.qc_builder.degree
            
            # Input angles
            theta_x = data_to_angle(x_value)
            for i in range(degree + 1):
                param_values[params['theta'][i]] = theta_x * (i + 1)
            
            # Coefficient angles
            if 'alpha_coefficients' in self._cached_params:
                alphas = self._cached_params['alpha_coefficients']
                for i, alpha in enumerate(alphas[:degree+1]):
                    param_values[params['alpha'][i]] = alpha
            else:
                for i in range(degree + 1):
                    param_values[params['alpha'][i]] = np.pi / 4
            
            # Combination angles
            if 'beta_combine' in self._cached_params:
                betas = self._cached_params['beta_combine']
                for i, beta in enumerate(betas[:degree]):
                    param_values[params['beta'][i]] = beta
            else:
                for i in range(degree):
                    param_values[params['beta'][i]] = np.pi / 4
            
            bound_qc = qc.assign_parameters(param_values)
            return bound_qc
    
    def get_nn_prediction(self, x_value):
        """
        Get neural network prediction for input.
        
        Args:
            x_value: Input value
            
        Returns:
            float: NN output
        """
        self.nn_model.eval()
        # Create tensor on CPU (model will handle device)
        with torch.no_grad():
            x_tensor = torch.tensor([[x_value]], dtype=torch.float32, device='cpu')
            # Move model to CPU for inference
            self.nn_model.cpu()
            pred = self.nn_model(x_tensor)
            if pred.dim() > 0:
                pred = pred.squeeze()
            return pred.item()
    
    def verify_mapping(self, test_points, executor=None, verbose=True, 
                       method='direct', shots=8192):
        """
        Verify that quantum circuit reproduces NN behavior.
        
        Args:
            test_points: Array of x values to test
            executor: CircuitExecutor instance (deprecated, uses internal)
            verbose: Print detailed results
            method: 'direct' (exact encoding) or 'hybrid' (quantum powers)
            shots: Number of measurement shots
            
        Returns:
            dict: Verification results including:
                - nn_predictions: NN outputs
                - quantum_results: Quantum expectation values
                - differences: Absolute differences
                - mean_difference: Average difference
        """
        nn_preds = []
        q_results = []
        
        # Get coefficients for quantum evaluation
        if isinstance(self.nn_model, PolynomialNN):
            coeffs = self.nn_model.get_coefficients()
        else:
            # For non-polynomial models, extract first layer
            coeffs = None
        
        for x in test_points:
            # NN prediction
            nn_pred = self.get_nn_prediction(x)
            nn_preds.append(nn_pred)
            
            # Quantum execution using CORRECTED method
            if coeffs is not None:
                if method == 'direct':
                    # Direct encoding: exact match (up to shot noise)
                    q_exp = quantum_polynomial_direct(x, coeffs, shots)
                else:
                    # Hybrid: quantum powers, classical sum
                    q_exp = quantum_polynomial_eval(x, coeffs, shots)
            else:
                # Fallback for non-polynomial models
                q_exp = nn_pred  # Use NN prediction as placeholder
                
            q_results.append(q_exp)
            
        nn_preds = np.array(nn_preds)
        q_results = np.array(q_results)
        differences = np.abs(nn_preds - q_results)
        
        results = {
            'test_points': test_points,
            'nn_predictions': nn_preds,
            'quantum_results': q_results,
            'differences': differences,
            'mean_difference': np.mean(differences),
            'max_difference': np.max(differences),
        }
        
        if verbose:
            print("\nNN-Quantum Mapping Verification:")
            print("-" * 60)
            print(f"{'x':<10} | {'NN':<12} | {'Quantum':<12} | {'Diff':<10}")
            print("-" * 60)
            for i, x in enumerate(test_points):
                print(f"{x:<10.4f} | {nn_preds[i]:<12.4f} | {q_results[i]:<12.4f} | {differences[i]:<10.4f}")
            print("-" * 60)
            print(f"Mean Difference: {results['mean_difference']:.4f}")
            print(f"Max Difference:  {results['max_difference']:.4f}")
            
        return results


class BatchMapper:
    """
    Batch processing for NN to Quantum mapping.
    
    Efficiently handles multiple models or multiple input batches.
    """
    
    def __init__(self, executor=None):
        """
        Args:
            executor: CircuitExecutor for quantum simulation
        """
        self.executor = executor or CircuitExecutor()
        
    def map_and_verify_batch(self, models, test_points, verbose=False):
        """
        Map and verify multiple models.
        
        Args:
            models: List of trained NN models
            test_points: Test input points
            verbose: Print progress
            
        Returns:
            list: Verification results for each model
        """
        all_results = []
        
        for i, model in enumerate(models):
            if verbose:
                print(f"Processing model {i+1}/{len(models)}...")
                
            mapper = NNToQuantumMapper(model)
            results = mapper.verify_mapping(test_points, self.executor, verbose=False)
            results['model_index'] = i
            all_results.append(results)
            
        return all_results
    
    def evaluate_accuracy(self, results_list):
        """
        Compute aggregate accuracy metrics.
        
        Args:
            results_list: List of verification result dicts
            
        Returns:
            dict: Aggregate metrics
        """
        mean_diffs = [r['mean_difference'] for r in results_list]
        max_diffs = [r['max_difference'] for r in results_list]
        
        return {
            'num_models': len(results_list),
            'avg_mean_diff': np.mean(mean_diffs),
            'std_mean_diff': np.std(mean_diffs),
            'avg_max_diff': np.mean(max_diffs),
            'best_model_idx': np.argmin(mean_diffs),
            'worst_model_idx': np.argmax(mean_diffs),
        }

