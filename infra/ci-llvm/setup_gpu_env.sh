#!/usr/bin/env bash
#
# Sets up GPU build environment.
# Usage: source setup_gpu_env.sh

# Include common utils
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/common.sh

# Env CUDA setup
if [[ ${GPU,,} =~ "cuda" ]]; then
  echo "Setting up CUDA environment"
  echo "Hard-coding MLIR-compatible CUDA version (12.9)"
  source /swtools/cuda/12.9.0/cuda_vars.sh
  check_program nvcc
fi

# Env Intel setup
if [[ ${GPU,,} =~ "intel" ]]; then
  echo "Setting up Intel XeGPU environment"
  source /swtools/intel-gpu/latest/intel_gpu_vars.sh
  check_program iga64
fi
