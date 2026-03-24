#!/usr/bin/env python3
"""
Plot Quantum Circuits for Native Polynomial
===========================================

Generates and saves circuit diagrams for degrees 1-6 of the native polynomial
approximation circuits.

Usage:
    python plot_circuits.py --output figures
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
from PIL import Image

# Add parent paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research.submit_cloud_batch import build_polynomial_circuit, PolynomialNN, train_nn
from research.research_config import POLYNOMIALS, CLOUD_CONFIG, evaluate_polynomial

def get_circuit(degree, x_value=0.5):
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
    
    # Build circuit for a specific x
    qc, _ = build_polynomial_circuit(x_value, learned_coeffs)
    return qc

def plot_circuits(degrees=[1, 2, 3, 4, 5, 6], output_dir='figures', x_value=0.5):
    """Generate and save circuit diagrams for specified degrees."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING CIRCUIT DIAGRAMS")
    print("=" * 70)
    
    for degree in degrees:
        print(f"\nGenerating circuit for Degree {degree}...")
        
        qc = get_circuit(degree, x_value)
        
        # Get circuit info
        qc_decomp = qc.decompose()
        ops = qc_decomp.count_ops()
        n_qubits = qc.num_qubits
        n_gates = sum(ops.values())
        n_2q = ops.get('cx', 0) + ops.get('cz', 0) + ops.get('ecr', 0)
        
        print(f"  Qubits: {n_qubits}, Total gates: {n_gates}, 2-qubit gates: {n_2q}")
        
        # Save circuit diagram
        # Option 1: Text output (always works)
        txt_file = os.path.join(output_dir, f'circuit_deg{degree}_text.txt')
        with open(txt_file, 'w') as f:
            f.write(f"Degree {degree} Polynomial Circuit (x={x_value})\n")
            f.write("=" * 70 + "\n")
            f.write(f"Qubits: {n_qubits}, 2-qubit gates: {n_2q}\n")
            f.write("\nCircuit:\n")
            f.write(str(qc.draw('text', fold=120)))
            f.write("\n")
        
        # Option 2: PDF/PNG via matplotlib (if available)
        try:
            # Try different output modes
            fig = qc.draw(output='mpl', style={'fontsize': 8}, 
                         fold=150, scale=0.7, plot_barriers=True)
            
            pdf_file = os.path.join(output_dir, f'circuit_deg{degree}.pdf')
            png_file = os.path.join(output_dir, f'circuit_deg{degree}.png')
            
            fig.savefig(pdf_file, dpi=300, bbox_inches='tight', format='pdf')
            fig.savefig(png_file, dpi=300, bbox_inches='tight', format='png')
            
            print(f"  Saved: {pdf_file}, {png_file}")
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as e1:
            # Try text-based PNG conversion
            try:
                import matplotlib.pyplot as plt
                from matplotlib.patches import Rectangle, FancyBboxPatch
                
                # Create figure with text representation
                fig, ax = plt.subplots(figsize=(14, max(3, n_qubits * 0.8)))
                ax.axis('off')
                
                # Draw circuit as text
                circuit_text = qc.draw('text', fold=200)
                ax.text(0.05, 0.95, f'Degree {degree} Circuit (x={x_value})\n\n{circuit_text}',
                       fontfamily='monospace', fontsize=8, verticalalignment='top',
                       transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                
                pdf_file = os.path.join(output_dir, f'circuit_deg{degree}.pdf')
                png_file = os.path.join(output_dir, f'circuit_deg{degree}.png')
                
                fig.savefig(pdf_file, dpi=300, bbox_inches='tight', format='pdf')
                fig.savefig(png_file, dpi=300, bbox_inches='tight', format='png')
                plt.close(fig)
                print(f"  Saved: {pdf_file}, {png_file} (text-based)")
            except Exception as e2:
                print(f"  Could not generate PDF/PNG: {e2}")
                print(f"  Text version saved: {txt_file}")
        
        # Option 3: LaTeX (alternative)
        try:
            latex_file = os.path.join(output_dir, f'circuit_deg{degree}.tex')
            with open(latex_file, 'w') as f:
                latex_code = qc.draw('latex_source', fold=120)
                f.write(latex_code)
            print(f"  LaTeX version saved: {latex_file}")
        except:
            pass
    
    print("\n" + "=" * 70)
    print(f"All circuits saved to: {output_dir}")
    print("=" * 70)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate circuit diagrams for native polynomial circuits'
    )
    parser.add_argument('--degrees', default='1,2,3,4,5,6',
                       help='Comma-separated list of degrees')
    parser.add_argument('--output', default='figures',
                       help='Output directory for figures')
    parser.add_argument('--x-value', type=float, default=0.5,
                       help='x value for circuit generation')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    degree_list = [int(d.strip()) for d in args.degrees.split(',')]
    plot_circuits(degree_list, args.output, args.x_value)
