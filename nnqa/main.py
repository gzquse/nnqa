#!/usr/bin/env python3
"""
Neural-Native Quantum Arithmetic - Main Entry Point
=====================================================

This is the main training script for the NNQA workflow.

Usage:
    python main.py [OPTIONS]
    
    Or via shell script:
    ./main.sh

Configuration is loaded from config.sh or command line arguments.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from datetime import datetime
from time import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnqa.models import PolynomialNN, DeepPolynomialNN
from nnqa.quantum_circuits import QuantumPolynomialCircuit, CircuitExecutor
from nnqa.mapper import NNToQuantumMapper
from nnqa.trainer import Trainer, TargetFunction, create_trainer
from nnqa.evaluator import Evaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Neural-Native Quantum Arithmetic Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument('--model-type', type=str, default='polynomial',
                        choices=['polynomial', 'deep'],
                        help='Model architecture type')
    parser.add_argument('--degree', type=int, default=3,
                        help='Polynomial degree')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[16, 16],
                        help='Hidden layer dimensions for deep model')
    
    # Target function parameters
    parser.add_argument('--target-type', type=str, default='polynomial',
                        choices=['polynomial', 'sin', 'cos', 'gaussian'],
                        help='Target function type')
    parser.add_argument('--target-coeffs', type=float, nargs='+', 
                        default=[0.1, 0.3, -0.2, 0.5],
                        help='Polynomial coefficients for target')
    parser.add_argument('--target-freq', type=float, default=1.0,
                        help='Frequency for trigonometric targets')
    parser.add_argument('--target-amp', type=float, default=0.5,
                        help='Amplitude for trigonometric targets')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate')
    parser.add_argument('--samples', type=int, default=500,
                        help='Number of training samples')
    parser.add_argument('--scheduler', type=str, default='step',
                        choices=['step', 'cosine', 'none'],
                        help='Learning rate scheduler')
    
    # Quantum parameters
    parser.add_argument('--shots', type=int, default=4096,
                        help='Quantum simulation shots')
    parser.add_argument('--skip-quantum', action='store_true',
                        help='Skip quantum evaluation')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--identifier', type=str, default='default',
                        help='Run identifier for file naming')
    parser.add_argument('--log-interval', type=int, default=20,
                        help='Epochs between log messages')
    parser.add_argument('--checkpoint-interval', type=int, default=50,
                        help='Epochs between checkpoints')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Compute device (default: cpu for login nodes)')
    parser.add_argument('--cuda-device', type=int, default=0,
                        help='CUDA device number')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force CPU usage even if CUDA is available')
    
    return parser.parse_args()


def setup_device(args):
    """Setup compute device."""
    # Force CPU if requested
    if hasattr(args, 'force_cpu') and args.force_cpu:
        device = 'cpu'
    elif args.device == 'auto':
        if torch.cuda.is_available():
            try:
                # Test if we can actually use CUDA
                torch.cuda.get_device_name(args.cuda_device)
                device = f'cuda:{args.cuda_device}'
            except Exception:
                print("CUDA detected but not usable, falling back to CPU")
                device = 'cpu'
        else:
            device = 'cpu'
    elif args.device == 'cuda':
        if torch.cuda.is_available():
            device = f'cuda:{args.cuda_device}'
        else:
            print("CUDA requested but not available, falling back to CPU")
            device = 'cpu'
    else:
        device = 'cpu'
        
    print(f"Using device: {device}")
    if 'cuda' in device:
        print(f"CUDA device: {torch.cuda.get_device_name(args.cuda_device)}")
        
    return device


def create_target_function(args):
    """Create target function from arguments."""
    if args.target_type == 'polynomial':
        return TargetFunction(
            func_type='polynomial',
            coefficients=args.target_coeffs
        )
    elif args.target_type == 'sin':
        return TargetFunction(
            func_type='sin',
            frequency=args.target_freq,
            amplitude=args.target_amp
        )
    elif args.target_type == 'cos':
        return TargetFunction(
            func_type='cos',
            frequency=args.target_freq,
            amplitude=args.target_amp
        )
    elif args.target_type == 'gaussian':
        return TargetFunction(
            func_type='gaussian',
            mu=0.0,
            sigma=0.3
        )
    else:
        raise ValueError(f"Unknown target type: {args.target_type}")


def create_model(args):
    """Create model from arguments."""
    if args.model_type == 'polynomial':
        return PolynomialNN(degree=args.degree)
    elif args.model_type == 'deep':
        return DeepPolynomialNN(
            degree=args.degree,
            hidden_dims=args.hidden_dims
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")


def main():
    """Main training workflow."""
    args = parse_args()
    
    # Setup
    start_time = time()
    device = setup_device(args)
    
    # Create output directory
    run_dir = os.path.join(args.output_dir, f"run_{args.identifier}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Print configuration
    print("\n" + "=" * 70)
    print("NEURAL-NATIVE QUANTUM ARITHMETIC")
    print("=" * 70)
    print(f"\nRun ID: {args.identifier}")
    print(f"Output: {run_dir}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    print("\n--- Configuration ---")
    print(f"Model: {args.model_type} (degree={args.degree})")
    print(f"Target: {args.target_type}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Samples: {args.samples}")
    print(f"Quantum shots: {args.shots}")
    
    # Save configuration
    config = vars(args)
    config['device_used'] = device
    config['timestamp'] = datetime.now().isoformat()
    
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create components
    model = create_model(args)
    target_func = create_target_function(args)
    
    print(f"\nTarget Function: {target_func.get_description()}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        target_func=target_func,
        output_dir=run_dir,
        device=device
    )
    
    # Train
    print("\n" + "=" * 70)
    print("TRAINING PHASE")
    print("=" * 70)
    
    history = trainer.train(
        epochs=args.epochs,
        lr=args.lr,
        num_samples=args.samples,
        scheduler_type=args.scheduler if args.scheduler != 'none' else None,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Quantum evaluation
    if not args.skip_quantum:
        print("\n" + "=" * 70)
        print("QUANTUM EVALUATION PHASE")
        print("=" * 70)
        
        evaluator = Evaluator(
            model=model,
            output_dir=run_dir,
            shots=args.shots
        )
        
        # Run benchmark
        test_points = np.linspace(-0.8, 0.8, 15)
        benchmark = evaluator.run_benchmark(
            num_points=15,
            target_func=target_func
        )
        
        # Save detailed comparison
        comparison = evaluator.compare_nn_quantum(
            test_points=test_points,
            target_func=target_func,
            verbose=True
        )
        evaluator.save_results(comparison)
    
    # Final summary
    elapsed = time() - start_time
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Runtime: {elapsed:.2f} seconds")
    print(f"Final Training Loss: {history.get_final_loss():.6f}")
    
    if hasattr(model, 'get_coefficients'):
        print(f"Learned Coefficients: {model.get_coefficients()}")
    
    if not args.skip_quantum:
        print(f"NN-Quantum Mean Diff: {benchmark['comparison']['mean_nn_q_diff']:.4f}")
        
    print(f"\nResults saved to: {run_dir}")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

