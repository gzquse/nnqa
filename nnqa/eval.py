#!/usr/bin/env python3
"""
Neural-Native Quantum Arithmetic - Evaluation Entry Point
===========================================================

This script evaluates a trained model and compares it with
quantum circuit execution.

Usage:
    python eval.py --model-path results/run_xxx/final_model.pt [OPTIONS]
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnqa.models import PolynomialNN, DeepPolynomialNN
from nnqa.trainer import TargetFunction
from nnqa.evaluator import Evaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Neural-Native Quantum Arithmetic Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to saved model weights')
    
    # Model parameters
    parser.add_argument('--model-type', type=str, default='polynomial',
                        choices=['polynomial', 'deep'],
                        help='Model architecture type')
    parser.add_argument('--degree', type=int, default=3,
                        help='Polynomial degree')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[16, 16],
                        help='Hidden layer dimensions for deep model')
    
    # Target function (for comparison)
    parser.add_argument('--target-type', type=str, default='polynomial',
                        choices=['polynomial', 'sin', 'cos', 'gaussian', 'none'],
                        help='Target function type')
    parser.add_argument('--target-coeffs', type=float, nargs='+', 
                        default=[0.1, 0.3, -0.2, 0.5],
                        help='Polynomial coefficients for target')
    
    # Evaluation parameters
    parser.add_argument('--num-points', type=int, default=20,
                        help='Number of test points')
    parser.add_argument('--x-min', type=float, default=-0.8,
                        help='Minimum x value')
    parser.add_argument('--x-max', type=float, default=0.8,
                        help='Maximum x value')
    parser.add_argument('--shots', type=int, default=4096,
                        help='Quantum simulation shots')
    
    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as model)')
    
    return parser.parse_args()


def load_model(args):
    """Load trained model."""
    if args.model_type == 'polynomial':
        model = PolynomialNN(degree=args.degree)
    elif args.model_type == 'deep':
        model = DeepPolynomialNN(
            degree=args.degree,
            hidden_dims=args.hidden_dims
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()
    
    return model


def create_target_function(args):
    """Create target function from arguments."""
    if args.target_type == 'none':
        return None
        
    if args.target_type == 'polynomial':
        return TargetFunction(
            func_type='polynomial',
            coefficients=args.target_coeffs
        )
    elif args.target_type == 'sin':
        return TargetFunction(func_type='sin')
    elif args.target_type == 'cos':
        return TargetFunction(func_type='cos')
    elif args.target_type == 'gaussian':
        return TargetFunction(func_type='gaussian')
    else:
        raise ValueError(f"Unknown target type: {args.target_type}")


def main():
    """Main evaluation workflow."""
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("NEURAL-NATIVE QUANTUM ARITHMETIC - EVALUATION")
    print("=" * 70)
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.dirname(args.model_path)
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nModel Path: {args.model_path}")
    print(f"Output Dir: {output_dir}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(args)
    print(f"Model Type: {type(model).__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    if hasattr(model, 'get_coefficients'):
        print(f"Coefficients: {model.get_coefficients()}")
    
    # Create target function
    target_func = create_target_function(args)
    if target_func is not None:
        print(f"Target: {target_func.get_description()}")
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        output_dir=output_dir,
        shots=args.shots
    )
    
    # Generate test points
    test_points = np.linspace(args.x_min, args.x_max, args.num_points)
    
    # Run benchmark
    print("\n" + "=" * 70)
    print("RUNNING BENCHMARK")
    print("=" * 70)
    
    benchmark = evaluator.run_benchmark(
        num_points=args.num_points,
        x_range=(args.x_min, args.x_max),
        target_func=target_func
    )
    
    # Detailed comparison
    print("\n" + "=" * 70)
    print("DETAILED COMPARISON")
    print("=" * 70)
    
    comparison = evaluator.compare_nn_quantum(
        test_points=test_points,
        target_func=target_func,
        verbose=True
    )
    
    # Save results
    evaluator.save_results(comparison, filename='evaluation_detailed.json')
    
    # Print final summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


