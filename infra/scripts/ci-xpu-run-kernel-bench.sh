#!/usr/bin/env bash
#
# Script for CI - XPU job.
#
# Run KernelBench on Intel GPU.

SCRIPTS_DIR=$(realpath $(dirname $0))

# Backends
BENCH_BACKEND_TORCH="torch"
BENCH_BACKEND_TORCH_COMPILE="torch-compile"
BENCH_BACKEND_TRITON="triton"
BENCH_BACKEND_HELION="helion"
BENCH_BACKEND_MLIR="mlir"

# Run modes
RUN_MODE_BENCH="bench"
RUN_MODE_CI="ci"

die_syntax() {
  echo "Syntax: $0 [-b (${BENCH_BACKEND_TORCH}|${BENCH_BACKEND_TORCH_COMPILE}|${BENCH_BACKEND_TRITON}|${BENCH_BACKEND_HELION}|${BENCH_BACKEND_MLIR})] [-m (${RUN_MODE_BENCH}|${RUN_MODE_CI})]"
  echo ""
  echo "  -b: Optional, backend to use (default: torch)"
  echo "  -m: Optional, run mode to use (default: bench)"
  exit 1
}

# Options
BENCH_BACKEND=${BENCH_BACKEND_TORCH}
RUN_MODE=${RUN_MODE_BENCH}

while getopts "b:m:" arg; do
  case ${arg} in
    b)
      if [ "${OPTARG}" == "${BENCH_BACKEND_TORCH}" ] || \
         [ "${OPTARG}" == "${BENCH_BACKEND_TORCH_COMPILE}" ] || \
         [ "${OPTARG}" == "${BENCH_BACKEND_TRITON}" ] || \
         [ "${OPTARG}" == "${BENCH_BACKEND_HELION}" ] || \
         [ "${OPTARG}" == "${BENCH_BACKEND_MLIR}" ]; then
        BENCH_BACKEND="${OPTARG}"
      else
        echo "Invalid backend: ${OPTARG}"
        die_syntax
      fi
      ;;
    m)
      if [ "${OPTARG}" == "${RUN_MODE_BENCH}" ] || \
         [ "${OPTARG}" == "${RUN_MODE_CI}" ]; then
        RUN_MODE="${OPTARG}"
      else
        echo "Invalid run mode: ${OPTARG}"
        die_syntax
      fi
      ;;
    ?)
      echo "Invalid option: ${OPTARG}"
      die_syntax
      ;;
  esac
done

# Setup
echo "--- Setup environment"
source /swtools/intel-gpu/latest/intel_gpu_vars.sh

export TRITON_ALWAYS_COMPILE=1
echo "TRITON_ALWAYS_COMPILE=${TRITON_ALWAYS_COMPILE}"
echo ""

echo "--- Setup project"
git submodule update --init

pip install --upgrade --user uv
AI_BENCH_UV=${HOME}/.local/bin/uv

PROJECT_DEPS="--extra xpu"
if [[ "${BENCH_BACKEND}" == "${BENCH_BACKEND_MLIR}" ]]; then
  PROJECT_DEPS="${PROJECT_DEPS} --extra mlir"
fi

${AI_BENCH_UV} sync ${PROJECT_DEPS} --link-mode copy
echo ""

# Run benchmark
echo "--- Run KernelBench (${BENCH_BACKEND})"

BENCH_FLAGS="--xpu"

if [[ "${RUN_MODE}" == "${RUN_MODE_BENCH}" ]]; then
  BENCH_FLAGS="${BENCH_FLAGS} --bench"
fi
if [[ "${RUN_MODE}" == "${RUN_MODE_CI}" ]]; then
  BENCH_FLAGS="${BENCH_FLAGS} --ci"
fi

if [[ "${BENCH_BACKEND}" == "${BENCH_BACKEND_TORCH_COMPILE}" ]]; then
  BENCH_FLAGS="${BENCH_FLAGS} --torch-compile"
fi
if [[ "${BENCH_BACKEND}" == "${BENCH_BACKEND_TRITON}" ]]; then
  BENCH_FLAGS="${BENCH_FLAGS} --triton"
fi
if [[ "${BENCH_BACKEND}" == "${BENCH_BACKEND_HELION}" ]]; then
  BENCH_FLAGS="${BENCH_FLAGS} --helion"
  # Suppress logging to minimize noise in the benchmark output.
  export HELION_AUTOTUNE_PROGRESS_BAR=0
  export HELION_AUTOTUNE_LOG_LEVEL=0
fi
if [[ "${BENCH_BACKEND}" == "${BENCH_BACKEND_MLIR}" ]]; then
  GPU=intel source ${SCRIPTS_DIR}/../ci-llvm/setup_llvm_env.sh
  echo "LLVM: ${LLVM_INSTALL_DIR}"
  BENCH_FLAGS="${BENCH_FLAGS} --mlir --variant mlir-gpu-bench"
fi

${AI_BENCH_UV} run ai-bench ${BENCH_FLAGS}
EXIT_CODE=$?

echo ""

exit ${EXIT_CODE}
