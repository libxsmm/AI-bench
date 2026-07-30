#!/usr/bin/env bash
#
# Script for automation only.
# Environment variables must have been declared already.
#
# Check LLVM installation.

# Include common utils
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/common.sh

LLVMROOT=${HOME}/installs/llvm
mkdir -p ${LLVMROOT}

# Find LLVM_VERSION
LLVM_VERSION=$(llvm_version)
LLVM_INSTALL_DIR=${LLVMROOT}/${LLVM_VERSION}
LLVM_INSTALL_DIR=$(add_device_extensions ${LLVM_INSTALL_DIR} ${GPU})

if [ -f "${LLVM_INSTALL_DIR}/bin/mlir-opt" ]; then
  echo "Found $LLVM_VERSION"
  exit 0
else
  echo "Not Found 'mlir-opt' in ${LLVM_INSTALL_DIR}"
fi

exit 1
