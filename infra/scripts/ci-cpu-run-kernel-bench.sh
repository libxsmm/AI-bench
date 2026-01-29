#!/usr/bin/env bash
#
# Script for CI - CPU job.
#
# Run KernelBench on Intel CPU.

SCRIPTS_DIR=$(realpath $(dirname $0))

# Backends
BENCH_BACKEND_TORCH="torch"
BENCH_BACKEND_TORCH_COMPILE="torch-compile"
BENCH_BACKEND_MLIR="mlir"

die_syntax() {
  echo "Syntax: $0 [-b (${BENCH_BACKEND_TORCH}|${BENCH_BACKEND_TORCH_COMPILE}|${BENCH_BACKEND_MLIR})]"
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
         [ "${OPTARG}" == "${BENCH_BACKEND_MLIR}" ]; then
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
export OMP_NUM_THREADS=1
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

THREADS_PER_CORE=$(lscpu | grep --color=never "Thread.*core" | tee - | grep -o "[0-9]\+")
SKIP=$((THREADS_PER_CORE-1)) # 0 for no HT, 1 for 2, 3 for 4, etc.
export KMP_AFFINITY=granularity=fine,compact,$SKIP,0
echo "KMP_AFFINITY=${KMP_AFFINITY}"
echo ""

echo "--- Setup project"
git submodule update --init

pip install --upgrade --user uv
AI_BENCH_UV=${HOME}/.local/bin/uv

PROJECT_DEPS="--extra cpu"
if [[ "${BENCH_BACKEND}" == "${BENCH_BACKEND_MLIR}" ]]; then
  PROJECT_DEPS="${PROJECT_DEPS} --extra mlir"
fi

${AI_BENCH_UV} sync ${PROJECT_DEPS} --link-mode copy
echo ""

# Run benchmark
echo "--- Run KernelBench (${BENCH_BACKEND})"

BENCH_FLAGS="--bench --gflops"

if [[ "${BENCH_BACKEND}" == "${BENCH_BACKEND_TORCH_COMPILE}" ]]; then
  BENCH_FLAGS="${BENCH_FLAGS} --torch-compile"
fi
if [[ "${BENCH_BACKEND}" == "${BENCH_BACKEND_MLIR}" ]]; then
  BENCH_FLAGS="${BENCH_FLAGS} --mlir"
fi

${AI_BENCH_UV} run ai-bench ${BENCH_FLAGS}
EXIT_CODE=$?

echo ""

exit ${EXIT_CODE}
