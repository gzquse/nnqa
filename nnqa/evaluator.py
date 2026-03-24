#!/usr/bin/env python3
"""
Evaluation Module for Neural-Native Quantum Arithmetic
=======================================================

This module provides comprehensive evaluation of trained models
and their quantum circuit mappings.

Features:
- Model accuracy evaluation
- Quantum-NN comparison
- Visualization of results
- Benchmark reporting
"""

import os
import json
import numpy as np
import torch
from datetime import datetime

from .models import PolynomialNN, DeepPolynomialNN
from .quantum_circuits import QuantumPolynomialCircuit, CircuitExecutor, data_to_angle
from .mapper import NNToQuantumMapper, quantum_polynomial_direct, quantum_polynomial_eval


class Evaluator:
    """
    Comprehensive evaluation of NN to Quantum mapping.
    
    Performs:
    - Classical NN evaluation
    - Quantum circuit simulation
    - NN vs Quantum comparison
    - Statistical analysis
    """
    
    def __init__(self, model, output_dir='results', shots=4096):
        """
        Args:
            model: Trained PyTorch model
            output_dir: Directory for saving results
            shots: Quantum simulation shots
        """
        self.model = model
        self.output_dir = output_dir
        self.shots = shots
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.executor = CircuitExecutor(shots=shots)
        self.mapper = NNToQuantumMapper(model)
        
    def evaluate_nn(self, test_points, target_func=None):
        """
        Evaluate neural network on test points.
        
        Args:
            test_points: Array of input values
            target_func: Optional target function for comparison
            
        Returns:
            dict: NN evaluation results
        """
        self.model.eval()
        # Move model to CPU for evaluation (quantum simulation is on CPU)
        self.model.cpu()
        
        predictions = []
        with torch.no_grad():
            for x in test_points:
                x_tensor = torch.tensor([[x]], dtype=torch.float32, device='cpu')
                pred = self.model(x_tensor)
                if pred.dim() > 0:
                    pred = pred.squeeze()
                predictions.append(pred.item())
                
        predictions = np.array(predictions)
        
        results = {
            'test_points': test_points,
            'predictions': predictions,
        }
        
        if target_func is not None:
            targets = target_func(test_points)
            errors = np.abs(predictions - targets)
            results['targets'] = targets
            results['errors'] = errors
            results['mae'] = np.mean(errors)
            results['mse'] = np.mean(errors ** 2)
            results['max_error'] = np.max(errors)
            
        return results
    
    def evaluate_quantum(self, test_points, verbose=True, method='direct'):
        """
        Evaluate quantum circuit on test points.
        
        Args:
            test_points: Array of input values
            verbose: Print progress
            method: 'direct' (exact) or 'hybrid' (quantum powers)
            
        Returns:
            dict: Quantum evaluation results
        """
        quantum_results = []
        
        if verbose:
            print(f"Evaluating quantum circuit (method={method})...")
        
        # Get coefficients for polynomial models
        if isinstance(self.model, PolynomialNN):
            coeffs = self.model.get_coefficients()
        else:
            coeffs = None
            
        for i, x in enumerate(test_points):
            if coeffs is not None:
                if method == 'direct':
                    # CORRECTED: Direct encoding gives exact match
                    exp_val = quantum_polynomial_direct(x, coeffs, self.shots)
                else:
                    # Hybrid: quantum powers
                    exp_val = quantum_polynomial_eval(x, coeffs, self.shots)
            else:
                # Fallback to old method for non-polynomial models
                qc = self.mapper.build_mapped_circuit(x)
                exp_val = self.executor.execute(qc)
                
            quantum_results.append(exp_val)
            
            if verbose and (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/{len(test_points)} points")
                
        return {
            'test_points': test_points,
            'expectation_values': np.array(quantum_results),
        }
    
    def compare_nn_quantum(self, test_points, target_func=None, verbose=True):
        """
        Compare NN and quantum circuit outputs.
        
        Args:
            test_points: Array of input values
            target_func: Optional target function
            verbose: Print detailed results
            
        Returns:
            dict: Comprehensive comparison results
        """
        # Get NN predictions
        nn_results = self.evaluate_nn(test_points, target_func)
        
        # Get quantum results
        q_results = self.evaluate_quantum(test_points, verbose)
        
        # Compute differences
        nn_preds = nn_results['predictions']
        q_exp = q_results['expectation_values']
        nn_q_diff = np.abs(nn_preds - q_exp)
        
        comparison = {
            'test_points': test_points,
            'nn_predictions': nn_preds,
            'quantum_expectations': q_exp,
            'nn_quantum_diff': nn_q_diff,
            'mean_nn_q_diff': np.mean(nn_q_diff),
            'max_nn_q_diff': np.max(nn_q_diff),
            'std_nn_q_diff': np.std(nn_q_diff),
        }
        
        if target_func is not None:
            targets = nn_results['targets']
            comparison['targets'] = targets
            comparison['nn_target_mae'] = nn_results['mae']
            comparison['quantum_target_diff'] = np.abs(q_exp - targets)
            comparison['quantum_target_mae'] = np.mean(np.abs(q_exp - targets))
            
        if verbose:
            self._print_comparison(comparison)
            
        return comparison
    
    def _print_comparison(self, comparison):
        """Print formatted comparison results."""
        print("\n" + "=" * 70)
        print("NN vs Quantum Circuit Comparison")
        print("=" * 70)
        
        test_points = comparison['test_points']
        nn_preds = comparison['nn_predictions']
        q_exp = comparison['quantum_expectations']
        diffs = comparison['nn_quantum_diff']
        
        has_target = 'targets' in comparison
        
        if has_target:
            targets = comparison['targets']
            header = f"{'x':<10} | {'Target':<12} | {'NN':<12} | {'Quantum':<12} | {'NN-Q Diff':<10}"
        else:
            header = f"{'x':<10} | {'NN':<12} | {'Quantum':<12} | {'Diff':<10}"
            
        print(header)
        print("-" * 70)
        
        for i in range(len(test_points)):
            if has_target:
                print(f"{test_points[i]:<10.4f} | {targets[i]:<12.4f} | {nn_preds[i]:<12.4f} | {q_exp[i]:<12.4f} | {diffs[i]:<10.4f}")
            else:
                print(f"{test_points[i]:<10.4f} | {nn_preds[i]:<12.4f} | {q_exp[i]:<12.4f} | {diffs[i]:<10.4f}")
                
        print("-" * 70)
        print(f"Mean NN-Quantum Difference: {comparison['mean_nn_q_diff']:.4f}")
        print(f"Max NN-Quantum Difference:  {comparison['max_nn_q_diff']:.4f}")
        print(f"Std NN-Quantum Difference:  {comparison['std_nn_q_diff']:.4f}")
        
        if has_target:
            print(f"NN Target MAE:              {comparison['nn_target_mae']:.4f}")
            print(f"Quantum Target MAE:         {comparison['quantum_target_mae']:.4f}")
            
        # Quality assessment
        if comparison['mean_nn_q_diff'] < 0.05:
            print("\nQuality: EXCELLENT - Quantum circuit faithfully reproduces NN")
        elif comparison['mean_nn_q_diff'] < 0.1:
            print("\nQuality: GOOD - Minor discrepancies between NN and Quantum")
        else:
            print("\nQuality: FAIR - Noticeable differences (may need tuning)")
            
    def run_benchmark(self, num_points=20, x_range=(-0.8, 0.8), target_func=None):
        """
        Run comprehensive benchmark.
        
        Args:
            num_points: Number of test points
            x_range: Range for test points
            target_func: Optional target function
            
        Returns:
            dict: Benchmark results
        """
        test_points = np.linspace(x_range[0], x_range[1], num_points)
        
        print("\n" + "#" * 70)
        print("# NEURAL-NATIVE QUANTUM ARITHMETIC BENCHMARK")
        print("#" * 70)
        
        # Get circuit info
        circuit_info = self.mapper.qc_builder.get_circuit_info()
        
        print(f"\nModel: {type(self.model).__name__}")
        print(f"Polynomial Degree: {circuit_info['degree']}")
        print(f"Quantum Circuit Qubits: {circuit_info['total_qubits']}")
        print(f"Simulation Shots: {self.shots}")
        
        # Run comparison
        comparison = self.compare_nn_quantum(test_points, target_func)
        
        # Build benchmark report
        benchmark = {
            'timestamp': datetime.now().isoformat(),
            'model_type': type(self.model).__name__,
            'circuit_info': circuit_info,
            'shots': self.shots,
            'num_test_points': num_points,
            'x_range': x_range,
            'comparison': {
                'mean_nn_q_diff': float(comparison['mean_nn_q_diff']),
                'max_nn_q_diff': float(comparison['max_nn_q_diff']),
                'std_nn_q_diff': float(comparison['std_nn_q_diff']),
            },
        }
        
        if 'nn_target_mae' in comparison:
            benchmark['nn_accuracy'] = {
                'mae': float(comparison['nn_target_mae']),
            }
            benchmark['quantum_accuracy'] = {
                'mae': float(comparison['quantum_target_mae']),
            }
            
        # Save benchmark
        benchmark_path = os.path.join(self.output_dir, 'benchmark_results.json')
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark, f, indent=2)
        print(f"\nBenchmark saved to: {benchmark_path}")
        
        return benchmark
    
    def save_results(self, comparison, filename='evaluation_results.json'):
        """Save evaluation results to file."""
        # Convert numpy arrays to lists for JSON serialization
        results = {}
        for key, value in comparison.items():
            if isinstance(value, np.ndarray):
                results[key] = value.tolist()
            else:
                results[key] = value
                
        results['timestamp'] = datetime.now().isoformat()
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {filepath}")


def evaluate_trained_model(model_path, output_dir, degree=3, 
                          target_type='polynomial', target_kwargs=None,
                          num_test_points=20, shots=4096):
    """
    Convenience function to evaluate a trained model.
    
    Args:
        model_path: Path to saved model weights
        output_dir: Output directory for results
        degree: Polynomial degree
        target_type: Target function type
        target_kwargs: Target function parameters
        num_test_points: Number of test points
        shots: Quantum simulation shots
        
    Returns:
        dict: Benchmark results
    """
    from .trainer import TargetFunction
    
    # Load model
    model = PolynomialNN(degree=degree)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Create target function
    target_kwargs = target_kwargs or {}
    target_func = TargetFunction(func_type=target_type, **target_kwargs)
    
    # Create evaluator
    evaluator = Evaluator(model, output_dir=output_dir, shots=shots)
    
    # Run benchmark
    benchmark = evaluator.run_benchmark(
        num_points=num_test_points,
        target_func=target_func
    )
    
    return benchmark

