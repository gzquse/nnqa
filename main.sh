#!/bin/bash
set -e  # Exit on any error

# =============================================================================
# NEURAL-NATIVE QUANTUM ARITHMETIC - Main Execution Script
# =============================================================================
# 
# This script orchestrates the full NNQA workflow:
# 1. Load configuration
# 2. Setup environment
# 3. Train neural network
# 4. Map to quantum circuit
# 5. Evaluate and benchmark
#
# Usage:
#   ./main.sh                    # Use default config
#   CONFIG_FILE=myconfig.sh ./main.sh  # Use custom config
#
# =============================================================================

# Configuration file
CONFIG_FILE="${CONFIG_FILE:-config.sh}"

# Load configuration if it exists
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
    echo "[*] Loaded configuration from $CONFIG_FILE"
else
    echo "[!] No config file found at $CONFIG_FILE, using defaults"
fi

# =============================================================================
# Default Configuration
# =============================================================================

# Model parameters
MODEL_TYPE=${MODEL_TYPE:-"polynomial"}
DEGREE=${DEGREE:-3}
HIDDEN_DIMS=${HIDDEN_DIMS:-"16 16"}

# Target function
TARGET_TYPE=${TARGET_TYPE:-"polynomial"}
TARGET_COEFFS=${TARGET_COEFFS:-"0.1 0.3 -0.2 0.5"}
TARGET_FREQ=${TARGET_FREQ:-1.0}
TARGET_AMP=${TARGET_AMP:-0.5}

# Training parameters
EPOCHS=${EPOCHS:-200}
LR=${LR:-0.05}
SAMPLES=${SAMPLES:-500}
SCHEDULER=${SCHEDULER:-"step"}
LOG_INTERVAL=${LOG_INTERVAL:-20}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-50}

# Quantum parameters
SHOTS=${SHOTS:-4096}
SKIP_QUANTUM=${SKIP_QUANTUM:-false}

# Output parameters
OUTPUT_DIR=${OUTPUT_DIR:-"results"}
IDENTIFIER=${IDENTIFIER:-"default"}

# Device
DEVICE=${DEVICE:-"auto"}
CUDA_DEVICE=${CUDA_DEVICE:-0}

# Execution mode
MODE=${MODE:-"train"}  # train, eval, both

# Virtual environment
VENV_NAME=${VENV_NAME:-""}

# =============================================================================
# Utility Functions
# =============================================================================

print_header() {
    echo ""
    echo "===== $1 ====="
    echo ""
}

print_success() {
    echo "[+] $1"
}

print_info() {
    echo "[*] $1"
}

print_error() {
    echo "[!] ERROR: $1" >&2
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "python3 not found. Please install Python 3.8+"
        exit 1
    fi
    
    local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_info "Python version: $python_version"
}

activate_venv() {
    if [[ -n "$VENV_NAME" ]]; then
        if [[ -d "$VENV_NAME" ]]; then
            print_info "Activating virtual environment: $VENV_NAME"
            source "$VENV_NAME/bin/activate"
        else
            print_info "Virtual environment not found: $VENV_NAME"
        fi
    fi
}

# =============================================================================
# Directory Setup
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Create results directory
mkdir -p "$PROJECT_ROOT/$OUTPUT_DIR"

print_header "Neural-Native Quantum Arithmetic"
print_info "Project root: $PROJECT_ROOT"
print_info "Output directory: $OUTPUT_DIR"
print_info "Mode: $MODE"

# =============================================================================
# Environment Setup
# =============================================================================

print_header "Environment Setup"
check_python
activate_venv

# Verify key packages
python3 -c "
import sys
try:
    import torch
    import numpy
    print('[+] Core packages verified')
    print(f'    PyTorch: {torch.__version__}')
    print(f'    CUDA available: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'[!] Missing package: {e}')
    sys.exit(1)
" || exit 1

# Check qiskit
python3 -c "
try:
    import qiskit
    from qiskit_aer import AerSimulator
    print('[+] Qiskit packages verified')
    print(f'    Qiskit: {qiskit.__version__}')
except ImportError as e:
    print(f'[!] Missing Qiskit package: {e}')
    print('    Install with: pip install qiskit qiskit-aer')
    exit(1)
" || exit 1

# =============================================================================
# Build Command Arguments
# =============================================================================

CMD_ARGS=(
    "--model-type" "$MODEL_TYPE"
    "--degree" "$DEGREE"
    "--hidden-dims" $HIDDEN_DIMS
    "--target-type" "$TARGET_TYPE"
    "--target-coeffs" $TARGET_COEFFS
    "--epochs" "$EPOCHS"
    "--lr" "$LR"
    "--samples" "$SAMPLES"
    "--scheduler" "$SCHEDULER"
    "--shots" "$SHOTS"
    "--output-dir" "$OUTPUT_DIR"
    "--identifier" "$IDENTIFIER"
    "--log-interval" "$LOG_INTERVAL"
    "--checkpoint-interval" "$CHECKPOINT_INTERVAL"
    "--device" "$DEVICE"
    "--cuda-device" "$CUDA_DEVICE"
)

# Add optional flags
if [[ "$SKIP_QUANTUM" == "true" ]]; then
    CMD_ARGS+=("--skip-quantum")
fi

# Add target-specific parameters
if [[ "$TARGET_TYPE" == "sin" || "$TARGET_TYPE" == "cos" ]]; then
    CMD_ARGS+=("--target-freq" "$TARGET_FREQ")
    CMD_ARGS+=("--target-amp" "$TARGET_AMP")
fi

# =============================================================================
# Execution
# =============================================================================

cd "$PROJECT_ROOT"

if [[ "$MODE" == "train" || "$MODE" == "both" ]]; then
    print_header "Training Phase"
    print_info "Running: python3 -m nnqa.main ${CMD_ARGS[*]}"
    
    if python3 -m nnqa.main "${CMD_ARGS[@]}"; then
        print_success "Training completed successfully"
    else
        print_error "Training failed"
        exit 1
    fi
fi

if [[ "$MODE" == "eval" || "$MODE" == "both" ]]; then
    print_header "Evaluation Phase"
    
    # Find latest model
    LATEST_RUN=$(ls -td "$OUTPUT_DIR"/run_* 2>/dev/null | head -1)
    
    if [[ -z "$LATEST_RUN" ]]; then
        print_error "No trained models found in $OUTPUT_DIR"
        exit 1
    fi
    
    MODEL_PATH="$LATEST_RUN/final_model.pt"
    
    if [[ ! -f "$MODEL_PATH" ]]; then
        print_error "Model not found: $MODEL_PATH"
        exit 1
    fi
    
    print_info "Evaluating model: $MODEL_PATH"
    
    EVAL_ARGS=(
        "--model-path" "$MODEL_PATH"
        "--model-type" "$MODEL_TYPE"
        "--degree" "$DEGREE"
        "--target-type" "$TARGET_TYPE"
        "--target-coeffs" $TARGET_COEFFS
        "--shots" "$SHOTS"
        "--num-points" "20"
    )
    
    if python3 -m nnqa.eval "${EVAL_ARGS[@]}"; then
        print_success "Evaluation completed successfully"
    else
        print_error "Evaluation failed"
        exit 1
    fi
fi

# =============================================================================
# Summary
# =============================================================================

print_header "Summary"
print_success "All operations completed successfully"
print_info "Results saved in: $OUTPUT_DIR"

# List output files
echo ""
echo "Output files:"
find "$OUTPUT_DIR" -name "*.json" -o -name "*.pt" 2>/dev/null | head -10


