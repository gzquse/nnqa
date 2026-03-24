#!/usr/bin/env python3
"""
Research Configuration for NN-to-Quantum Polynomial Study
==========================================================

Defines polynomial test cases for degrees 1-6, ensuring all outputs
are bounded within [-1, 1] for valid quantum encoding.
"""

import numpy as np

# ==============================================================================
# POLYNOMIAL DEFINITIONS
# ==============================================================================

# Polynomials designed to have |F(x)| <= 0.5 for x in [-1, 1]
POLYNOMIALS = {
    1: {
        'coefficients': [0.0, 0.5],  # F(x) = 0.5x
        'name': 'Linear',
        'latex': r'$F(x) = 0.5x$',
    },
    2: {
        'coefficients': [0.3, 0.4, -0.2],  # F(x) = 0.3 + 0.4x - 0.2x^2
        'name': 'Quadratic',
        'latex': r'$F(x) = 0.3 + 0.4x - 0.2x^2$',
    },
    3: {
        'coefficients': [0.1, 0.3, -0.1, 0.2],  # F(x) = 0.1 + 0.3x - 0.1x^2 + 0.2x^3
        'name': 'Cubic',
        'latex': r'$F(x) = 0.1 + 0.3x - 0.1x^2 + 0.2x^3$',
    },
    4: {
        'coefficients': [0.2, 0.2, -0.15, 0.1, -0.05],
        'name': 'Quartic',
        'latex': r'$F(x) = 0.2 + 0.2x - 0.15x^2 + 0.1x^3 - 0.05x^4$',
    },
    5: {
        'coefficients': [0.1, 0.25, -0.1, 0.15, -0.08, 0.05],
        'name': 'Quintic',
        'latex': r'$F(x) = 0.1 + 0.25x - 0.1x^2 + 0.15x^3 - 0.08x^4 + 0.05x^5$',
    },
    6: {
        'coefficients': [0.15, 0.2, -0.1, 0.1, -0.06, 0.04, -0.02],
        'name': 'Sextic',
        'latex': r'$F(x) = 0.15 + 0.2x - 0.1x^2 + 0.1x^3 - 0.06x^4 + 0.04x^5 - 0.02x^6$',
    },
}

# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

LOCAL_CONFIG = {
    'trials': 3,
    'shots': 4096,
    'num_samples': 15,
    'backend': 'aer_simulator',
    'x_range': (-0.9, 0.9),
    'train_epochs': 300,
    'train_lr': 0.1,
    'train_samples': 200,
}

CLOUD_CONFIG = {
    'trials': 10,
    'shots': 4096,
    'num_samples': 15,
    'backends': ['ibm_boston', 'ibm_pittsburgh'],
    'x_range': (-0.9, 0.9),
    'train_epochs': 300,
    'train_lr': 0.1,
    'train_samples': 200,
    'use_rc': True,  # Randomized compilation
    'use_dd': False,  # Dynamical decoupling
}

# ==============================================================================
# PLOTTING CONFIGURATION
# ==============================================================================

PLOT_CONFIG = {
    # IBM Design colorblind-friendly palette
    'colors': {
        'theoretical': '#2E8B57',   # Sea Green
        'classical': '#648FFF',     # Blue
        'quantum': '#DC267F',       # Magenta
        'quantum_alt': '#FE6100',   # Orange (for second backend)
        'error_fill': '#FFB000',    # Gold
        'pass_region': '#90EE90',   # Light Green
        'poor_region': '#FFE4B5',   # Moccasin
        'fail_region': '#FFB6C1',   # Light Pink
    },
    'figure_sizes': {
        'single_column': (3.5, 2.8),
        'double_column': (7, 5),
        'full_page': (7, 9),
    },
    'font_sizes': {
        'title': 14,
        'label': 12,
        'tick': 10,
        'legend': 9,
        'annotation': 9,
    },
    'thresholds': {
        'pass': 0.03,
        'poor': 0.10,
    },
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def evaluate_polynomial(x, coefficients):
    """Evaluate polynomial at x given coefficients [a0, a1, a2, ...]."""
    return sum(coefficients[i] * (x ** i) for i in range(len(coefficients)))


def get_polynomial_range(degree):
    """Compute the range of polynomial outputs for x in [-1, 1]."""
    coeffs = POLYNOMIALS[degree]['coefficients']
    x_test = np.linspace(-1, 1, 1000)
    y_test = np.array([evaluate_polynomial(x, coeffs) for x in x_test])
    return y_test.min(), y_test.max()


def validate_polynomials():
    """Verify all polynomials have outputs within [-1, 1]."""
    print("Validating polynomial bounds:")
    print("-" * 50)
    all_valid = True
    for degree in sorted(POLYNOMIALS.keys()):
        y_min, y_max = get_polynomial_range(degree)
        valid = (y_min >= -1) and (y_max <= 1)
        status = "OK" if valid else "FAIL"
        if not valid:
            all_valid = False
        print(f"  Degree {degree}: [{y_min:+.3f}, {y_max:+.3f}] - {status}")
    print("-" * 50)
    return all_valid


def format_polynomial_latex(coefficients):
    """Format polynomial coefficients as LaTeX string."""
    terms = []
    for i, c in enumerate(coefficients):
        if abs(c) < 1e-10:
            continue
        if i == 0:
            terms.append(f"{c:.2f}")
        elif i == 1:
            sign = '+' if c >= 0 else ''
            terms.append(f"{sign}{c:.2f}x")
        else:
            sign = '+' if c >= 0 else ''
            terms.append(f"{sign}{c:.2f}x^{i}")
    return ''.join(terms) if terms else "0"


if __name__ == '__main__':
    # Validate all polynomials when run directly
    validate_polynomials()
    
    print("\nPolynomial definitions:")
    for degree in sorted(POLYNOMIALS.keys()):
        info = POLYNOMIALS[degree]
        print(f"  Degree {degree} ({info['name']}): {info['coefficients']}")


