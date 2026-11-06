#!/usr/bin/env bash

# Exit immediately if any command fails, treat unset variables as errors,
# and ensure pipeline failures aren't hidden.
set -euo pipefail

# --- Fixed Configuration ---
PROJECT_DIR="/home/btr000/Projects/Retrievals/RACCOONN/RACCOONN"
CONDA_ENV="RACCOONN"

# --- Enforce Required Argument ---
if [ $# -eq 0 ]; then
    echo "Error: No experiment name provided."
    echo ""
    echo "Usage:   $0 <experiment_name>"
    echo "Example: $0 inverse_clearsky_patch_static_transform_0002"
    exit 1
fi

# --- Capture Experiment Name ---
EXPERIMENT="$1"

echo "=================================================="
echo " Starting RACCOONN Pipeline"
echo " Host: $(hostname)"
echo " Environment: ${CONDA_ENV}"
echo " Experiment: ${EXPERIMENT}"
echo " Timestamp: $(date)"
echo "=================================================="

# Initialize and activate Conda environment safely inside a script
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
else
    echo "Error: conda executable not found in PATH."
    exit 1
fi

# --- Fix for missing GLIBCXX version ---
# Force Python/Conda to load the environment's C++ standard library instead of the host system's
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# --- Fix for MLflow file store restriction ---
export MLFLOW_ALLOW_FILE_STORE=true

# Navigate to project root safely
cd "${PROJECT_DIR}" || { echo "Error: Could not change directory to ${PROJECT_DIR}"; exit 1; }

echo ""
echo "--- [1/3] Running configuration setup ---"
python -m config.setup "+experiment=${EXPERIMENT}"

echo ""
echo "--- [2/3] Running covariance preprocessing ---"
python -m code.preprocessing.covariance "+experiment=${EXPERIMENT}"

echo ""
echo "--- [3/3] Running model training ---"
python -m code.train "+experiment=${EXPERIMENT}"

echo ""
echo "=================================================="
echo " Pipeline Completed Successfully!"
echo " Timestamp: $(date)"
echo "=================================================="