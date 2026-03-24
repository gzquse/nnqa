#!/usr/bin/env python3
"""
Extract IonQ Forte results for LaTeX table
"""

import sys
import os
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from toolbox.Util_H5io4 import read4_data_hdf5

def extract_ionq_table_data(results_dir=None):
    """Extract IonQ Forte metrics for degrees 1-6."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if results_dir is None:
        results_path = os.path.join(script_dir, 'results')
    else:
        results_path = results_dir
    
    if not os.path.exists(results_path):
        print(f"Results directory not found: {results_path}")
        return None
    
    # Look for result files
    h5_files = glob.glob(os.path.join(results_path, 'deg*.h5'))
    h5_files += glob.glob(os.path.join(results_path, '**/deg*.h5'), recursive=True)
    
    if not h5_files:
        print(f"No H5 result files found in {results_path}")
        print("\nYou may need to retrieve IonQ results first:")
        print("  python research/ionq/retrieve_ionq_jobs.py")
        return None
    
    # Load and aggregate by degree
    results_by_degree = {}
    
    for h5_file in sorted(h5_files):
        try:
            data, meta = read4_data_hdf5(h5_file, verb=0)
            
            # Only process IonQ results
            if meta.get('provider') != 'ionq' and 'ionq' not in meta.get('backend', '').lower():
                continue
            
            degree = meta['degree']
            
            if degree not in results_by_degree:
                results_by_degree[degree] = []
            
            # Compute metrics
            theoretical = data['theoretical']
            measured = data['measured']
            
            quantum_rmse = np.sqrt(np.mean((measured - theoretical) ** 2))
            quantum_corr = np.corrcoef(theoretical, measured)[0, 1]
            abs_errors = np.abs(measured - theoretical)
            pass_rate = np.mean(abs_errors < 0.03) * 100
            
            results_by_degree[degree].append({
                'rmse': quantum_rmse,
                'corr': quantum_corr,
                'pass_rate': pass_rate,
            })
        except Exception as e:
            print(f"Warning: Error processing {h5_file}: {e}")
            continue
    
    # Compute statistics
    table_data = {}
    for degree in sorted(results_by_degree.keys()):
        results = results_by_degree[degree]
        rmse_values = [r['rmse'] for r in results]
        corr_values = [r['corr'] for r in results]
        pass_values = [r['pass_rate'] for r in results]
        
        table_data[degree] = {
            'rmse_mean': np.mean(rmse_values),
            'rmse_std': np.std(rmse_values) if len(rmse_values) > 1 else 0.0,
            'corr_mean': np.mean(corr_values),
            'pass_mean': np.mean(pass_values),
        }
    
    return table_data

if __name__ == '__main__':
    data = extract_ionq_table_data()
    
    if data:
        print("\nIonQ Forte Results for LaTeX Table:")
        print("=" * 70)
        print(f"{'Degree':<8} {'RMSE':<20} {'Corr':<10} {'Pass %':<10}")
        print("-" * 70)
        
        for degree in sorted(data.keys()):
            d = data[degree]
            rmse_str = f"${d['rmse_mean']:.3f} \\pm {d['rmse_std']:.3f}$"
            corr_str = f"{d['corr_mean']:.3f}"
            pass_str = f"{d['pass_mean']:.1f}"
            print(f"{degree:<8} {rmse_str:<20} {corr_str:<10} {pass_str:<10}")
        
        print("\nLaTeX format:")
        print("-" * 70)
        for degree in sorted(data.keys()):
            d = data[degree]
            degree_name = {1: 'Linear', 2: 'Quadratic', 3: 'Cubic', 
                          4: 'Quartic', 5: 'Quintic', 6: 'Sextic'}.get(degree, f'Degree {degree}')
            rmse_latex = f"${d['rmse_mean']:.3f} \\pm {d['rmse_std']:.3f}$"
            corr_latex = f"{d['corr_mean']:.3f}"
            pass_latex = f"{d['pass_mean']:.1f}"
            print(f"{degree} ({degree_name}) & {rmse_latex} & {corr_latex} & {pass_latex} \\\\")
    else:
        print("No IonQ data found. Please retrieve results first.")
