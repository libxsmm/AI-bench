#!/bin/bash
# Source this file to set up the SYCL/CUTLASS build environment.
#
# Usage:
#   export SYCL_TLA_DIR=/path/to/sycl-tla
#   source env_sycl.sh
#
# Requires:
#   - Intel oneAPI DPC++ compiler (icpx)
#   - Intel GPU drivers (Level Zero)
#   - sycl-tla headers (git clone https://github.com/intel/sycl-tla)

set +u

# oneAPI compiler (source setvars.sh if available)
if [ -n "$ONEAPI_ROOT" ] && [ -f "$ONEAPI_ROOT/setvars.sh" ]; then
    source "$ONEAPI_ROOT/setvars.sh" --force 2>/dev/null
fi

# Verify icpx is available
if ! command -v icpx &>/dev/null; then
    echo "ERROR: icpx not found. Source the oneAPI compiler environment first." >&2
    return 1 2>/dev/null || exit 1
fi

# SYCL-TLA / CUTLASS headers
: "${SYCL_TLA_DIR:=""}"
if [ -z "$SYCL_TLA_DIR" ] || [ ! -d "$SYCL_TLA_DIR" ]; then
    echo "WARNING: SYCL_TLA_DIR not set or not found. Set it to your sycl-tla checkout." >&2
else
    _compiler_inc="$(dirname "$(command -v icpx)")/../include"
    export AIBENCH_SYCL_INCLUDE="${SYCL_TLA_DIR}/include:${SYCL_TLA_DIR}/tools/util/include:${SYCL_TLA_DIR}/examples/common:${_compiler_inc}/sycl:${_compiler_inc}"
fi

export AIBENCH_SYCL_COMPILER=icpx

echo "SYCL env ready  (icpx: $(icpx --version 2>&1 | head -1))"
