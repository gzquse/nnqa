#!/bin/bash
# =============================================================================
# NEURAL-NATIVE QUANTUM ARITHMETIC - Configuration File
# =============================================================================
#
# Edit this file to customize the NNQA workflow.
# Then run: ./main.sh
#
# =============================================================================

# =============================================================================
# MODEL PARAMETERS
# =============================================================================

# Model architecture: "polynomial" or "deep"
MODEL_TYPE="polynomial"

# Polynomial degree (applies to both model types)
DEGREE=3

# Hidden layer dimensions for deep model (space-separated)
HIDDEN_DIMS="16 16"

# =============================================================================
# TARGET FUNCTION
# =============================================================================

# Target function type: "polynomial", "sin", "cos", "gaussian"
TARGET_TYPE="polynomial"

# Polynomial coefficients (for polynomial target)
# Represents: a_0 + a_1*x + a_2*x^2 + a_3*x^3 + ...
TARGET_COEFFS="0.1 0.3 -0.2 0.5"

# Frequency for trigonometric targets (sin, cos)
TARGET_FREQ=1.0

# Amplitude for trigonometric targets
TARGET_AMP=0.5

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

# Number of training epochs
EPOCHS=200

# Learning rate
LR=0.05

# Number of training samples
SAMPLES=500

# Learning rate scheduler: "step", "cosine", "none"
SCHEDULER="step"

# Epochs between log messages
LOG_INTERVAL=20

# Epochs between model checkpoints
CHECKPOINT_INTERVAL=50

# =============================================================================
# QUANTUM PARAMETERS
# =============================================================================

# Number of shots for quantum simulation
SHOTS=4096

# Skip quantum evaluation (true/false)
SKIP_QUANTUM=false

# =============================================================================
# OUTPUT PARAMETERS
# =============================================================================

# Output directory for results
OUTPUT_DIR="results"

# Run identifier (used in output folder naming)
IDENTIFIER="default"

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

# Compute device: "auto", "cpu", "cuda"
DEVICE="auto"

# CUDA device number (if using GPU)
CUDA_DEVICE=0

# =============================================================================
# EXECUTION MODE
# =============================================================================

# Execution mode: "train", "eval", "both"
MODE="train"

# =============================================================================
# ENVIRONMENT
# =============================================================================

# Virtual environment path (leave empty to skip activation)
VENV_NAME=""


