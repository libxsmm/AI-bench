#!/usr/bin/env bash
#
# Script for CI - CUDA job.
#
# Run KernelBench on Nvidia GPU.

SCRIPTS_DIR=$(realpath $(dirname $0))

# Backends
BENCH_BACKEND_TORCH="torch"
BENCH_BACKEND_TORCH_COMPILE="torch-compile"
BENCH_BACKEND_TRITON="triton"
BENCH_BACKEND_HELION="helion"

die_syntax() {
  echo "Syntax: $0 [-b (${BENCH_BACKEND_TORCH}|${BENCH_BACKEND_TORCH_COMPILE}|${BENCH_BACKEND_TRITON}|${BENCH_BACKEND_HELION})]"
  echo ""
  echo "  -b: Optional, backend to use (default: torch)"
  exit 1
}

# Options
BENCH_BACKEND=${BENCH_BACKEND_TORCH}
while getopts "b:" arg; do
  case ${arg} in
    b)
      if [ "${OPTARG}" == "${BENCH_BACKEND_TORCH}" ] || \
         [ "${OPTARG}" == "${BENCH_BACKEND_TORCH_COMPILE}" ] || \
         [ "${OPTARG}" == "${BENCH_BACKEND_TRITON}" ] || \
         [ "${OPTARG}" == "${BENCH_BACKEND_HELION}" ]; then
        BENCH_BACKEND="${OPTARG}"
      else
        echo "Invalid backend: ${OPTARG}"
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
source /swtools/cuda/latest/cuda_vars.sh
echo ""

echo "--- Setup project"
git submodule update --init

pip install --upgrade --user uv
AI_BENCH_UV=${HOME}/.local/bin/uv

${AI_BENCH_UV} sync --extra cuda --link-mode copy
echo ""

# Run benchmark
echo "--- Run KernelBench (${BENCH_BACKEND})"

BENCH_FLAGS="--cuda --bench"

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

${AI_BENCH_UV} run ai-bench ${BENCH_FLAGS}
EXIT_CODE=$?

echo ""

exit ${EXIT_CODE}
