#!/usr/bin/env bash
#
# Usage: source setup_llvm_env.sh

# Include common utils
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/common.sh

# Deduce LLVM installation directory
LLVMROOT=${HOME}/installs/llvm
LLVM_VERSION=$(llvm_version)
LLVM_INSTALL_DIR=${LLVMROOT}/${LLVM_VERSION}
LLVM_INSTALL_DIR=$(add_device_extensions ${LLVM_INSTALL_DIR} ${GPU})

if [ ! -d "${LLVM_INSTALL_DIR}" ]; then
  echo "LLVM install directory not found: ${LLVM_INSTALL_DIR}"
  exit 1
fi

# Set PYTHONPATH to override LLVM Python bindings
export PYTHONPATH="${LLVM_INSTALL_DIR}/python_packages/mlir_core/":$PYTHONPATH
export PATH="${LLVM_INSTALL_DIR}/bin/":$PATH
