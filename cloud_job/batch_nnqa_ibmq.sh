#!/bin/bash
# Batch submission script for NNQA IBM Quantum jobs
# Uses Heron 3 QPUs: ibm_boston, ibm_pittsburgh
#
# Usage:
#   ./batch_nnqa_ibmq.sh submit    # Submit all jobs
#   ./batch_nnqa_ibmq.sh retrieve  # Retrieve all results
#   ./batch_nnqa_ibmq.sh plot      # Plot all results
#   ./batch_nnqa_ibmq.sh all       # Do everything

set -u  # exit if uninitialized variable
set -e  # exit on error

# Configuration
basePath="cloud_job/out"
# Heron r3 backends (156 qubits) - best performance
backendList=("ibm_boston" "ibm_pittsburgh")
testTypes=("polynomial" "weighted_sum" "multiplication")

# Small batch for testing
numSample=5
numShot=4096
rndSeed=42

# Polynomial coefficients: F(x) = 0.1 + 0.3x - 0.1x^2 + 0.2x^3
polynomial="0.1 0.3 -0.1 0.2"

# Create output directories
mkdir -p "$basePath/jobs"
mkdir -p "$basePath/meas"
mkdir -p "$basePath/post"
mkdir -p "$basePath/plots"

echo "=============================================="
echo "NNQA Batch Job Script"
echo "=============================================="
echo "Base Path: $basePath"
echo "Backends: ${backendList[*]}"
echo "Test Types: ${testTypes[*]}"
echo "Samples: $numSample"
echo "Shots: $numShot"
echo "=============================================="

# Function to generate experiment name
get_exp_name() {
    local backend=$1
    local testType=$2
    local tag="${backend#ibm_}"  # Remove 'ibm_' prefix
    echo "nnqa_${tag}_${testType}"
}

# Submit jobs
submit_jobs() {
    echo ""
    echo "=== SUBMITTING JOBS ==="
    
    for backend in "${backendList[@]}"; do
        for testType in "${testTypes[@]}"; do
            expName=$(get_exp_name "$backend" "$testType")
            
            echo ""
            echo "Submitting: $expName"
            echo "  Backend: $backend"
            echo "  Test Type: $testType"
            
            python submit_nnqa_ibmq.py \
                --backend "$backend" \
                --basePath "$basePath" \
                --testType "$testType" \
                --polynomial $polynomial \
                --numSample $numSample \
                --numShot $numShot \
                --rndSeed $rndSeed \
                --expName "$expName" \
                --useRC \
                -E
            
            echo "  Done: $expName"
        done
    done
    
    echo ""
    echo "=== ALL JOBS SUBMITTED ==="
}

# Retrieve results
retrieve_jobs() {
    echo ""
    echo "=== RETRIEVING RESULTS ==="
    
    for backend in "${backendList[@]}"; do
        for testType in "${testTypes[@]}"; do
            expName=$(get_exp_name "$backend" "$testType")
            
            echo ""
            echo "Retrieving: $expName"
            
            python retrieve_nnqa_ibmq.py \
                --basePath "$basePath" \
                --expName "$expName"
            
            echo "  Done: $expName"
        done
    done
    
    echo ""
    echo "=== ALL RESULTS RETRIEVED ==="
}

# Plot results
plot_jobs() {
    echo ""
    echo "=== PLOTTING RESULTS ==="
    
    # Collect all experiment names
    expNames=""
    for backend in "${backendList[@]}"; do
        for testType in "${testTypes[@]}"; do
            expName=$(get_exp_name "$backend" "$testType")
            expNames="$expNames $expName"
        done
    done
    
    # Individual plots
    for backend in "${backendList[@]}"; do
        for testType in "${testTypes[@]}"; do
            expName=$(get_exp_name "$backend" "$testType")
            
            echo ""
            echo "Plotting: $expName"
            
            python plot_nnqa_accuracy.py \
                --basePath "$basePath" \
                --expName "$expName" \
                -p a \
                -Y
        done
    done
    
    # Comparison plot
    echo ""
    echo "Creating comparison plot..."
    python plot_nnqa_accuracy.py \
        --basePath "$basePath" \
        --expName $expNames \
        -p b \
        -Y
    
    echo ""
    echo "=== ALL PLOTS GENERATED ==="
    echo "Check: $basePath/plots/"
}

# Print summary table
print_summary() {
    echo ""
    echo "=== SUMMARY TABLE ==="
    echo ""
    printf "%-25s %-15s %-10s %-10s %-10s\n" "Experiment" "Backend" "RMSE" "Corr" "Pass%"
    echo "----------------------------------------------------------------------"
    
    for backend in "${backendList[@]}"; do
        for testType in "${testTypes[@]}"; do
            expName=$(get_exp_name "$backend" "$testType")
            
            # Extract metrics from H5 file (if exists)
            postFile="$basePath/post/${expName}.h5"
            if [ -f "$postFile" ]; then
                # Use Python to extract metrics
                python -c "
import sys
sys.path.insert(0, '..')
from toolbox.Util_H5io4 import read4_data_hdf5
expD, expMD = read4_data_hdf5('$postFile', verb=0)
pom = expMD.get('postproc', {})
rmse = pom.get('rmse', 'N/A')
corr = pom.get('correlation', 'N/A')
pass_rate = pom.get('pass_rate', 'N/A')
if isinstance(rmse, float):
    print(f'{rmse:.4f} {corr:.4f} {pass_rate*100:.1f}')
else:
    print('N/A N/A N/A')
" 2>/dev/null | while read rmse corr pass_rate; do
                    printf "%-25s %-15s %-10s %-10s %-10s\n" \
                        "$expName" "$backend" "$rmse" "$corr" "$pass_rate"
                done
            else
                printf "%-25s %-15s %-10s %-10s %-10s\n" \
                    "$expName" "$backend" "pending" "pending" "pending"
            fi
        done
    done
    
    echo ""
}

# Main
case "${1:-help}" in
    submit)
        submit_jobs
        ;;
    retrieve)
        retrieve_jobs
        ;;
    plot)
        plot_jobs
        print_summary
        ;;
    all)
        submit_jobs
        echo ""
        echo "Waiting 30 seconds before retrieval..."
        sleep 30
        retrieve_jobs
        plot_jobs
        print_summary
        ;;
    status)
        print_summary
        ;;
    *)
        echo ""
        echo "Usage: $0 {submit|retrieve|plot|all|status}"
        echo ""
        echo "Commands:"
        echo "  submit   - Submit all batch jobs to IBM Quantum"
        echo "  retrieve - Retrieve all job results"
        echo "  plot     - Generate accuracy plots"
        echo "  all      - Submit, retrieve, and plot"
        echo "  status   - Show summary table"
        echo ""
        ;;
esac

echo ""
echo "Done!"

