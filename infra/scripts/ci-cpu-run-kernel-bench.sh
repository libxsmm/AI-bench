#!/usr/bin/env bash
#
# Script for CI - CPU job.
#
# Run KernelBench on Intel CPU.

SCRIPTS_DIR=$(realpath $(dirname $0))

# Backends
BENCH_BACKEND_TORCH="torch"
BENCH_BACKEND_TORCH_COMPILE="torch-compile"
BENCH_BACKEND_TRITON="triton"
BENCH_BACKEND_MLIR="mlir"

# Run modes
RUN_MODE_BENCH="bench"
RUN_MODE_CI="ci"

die_syntax() {
  echo "Syntax: $0 [-b (${BENCH_BACKEND_TORCH}|${BENCH_BACKEND_TORCH_COMPILE}|${BENCH_BACKEND_TRITON}|${BENCH_BACKEND_MLIR})] [-m (${RUN_MODE_BENCH}|${RUN_MODE_CI})]"
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
if [[ "${BENCH_BACKEND}" == "${BENCH_BACKEND_TRITON}" ]]; then
  PROJECT_DEPS="${PROJECT_DEPS} --extra triton-cpu"
fi
if [[ "${BENCH_BACKEND}" == "${BENCH_BACKEND_MLIR}" ]]; then
  PROJECT_DEPS="${PROJECT_DEPS} --extra mlir"
fi

${AI_BENCH_UV} sync ${PROJECT_DEPS} --link-mode copy
echo ""

# Run benchmark
echo "--- Run KernelBench (${BENCH_BACKEND})"

BENCH_FLAGS="--gflops"

if [[ "${RUN_MODE}" == "${RUN_MODE_BENCH}" ]]; then
  BENCH_FLAGS="${BENCH_FLAGS} --bench"
fi

if [[ "${BENCH_BACKEND}" == "${BENCH_BACKEND_TORCH_COMPILE}" ]]; then
  BENCH_FLAGS="${BENCH_FLAGS} --torch-compile"
fi
if [[ "${BENCH_BACKEND}" == "${BENCH_BACKEND_TRITON}" ]]; then
  BENCH_FLAGS="${BENCH_FLAGS} --triton"
fi
if [[ "${BENCH_BACKEND}" == "${BENCH_BACKEND_MLIR}" ]]; then
  BENCH_FLAGS="${BENCH_FLAGS} --mlir"
  MLIR_PACKAGE_PATH=$(${AI_BENCH_UV} run python -c "import mlir; print(mlir.__path__[0])")
  export AIBENCH_MLIR_LIB_PATH=${MLIR_PACKAGE_PATH}/_mlir_libs/libmlir_c_runner_utils.so
fi

${AI_BENCH_UV} run ai-bench ${BENCH_FLAGS}
EXIT_CODE=$?

echo ""

exit ${EXIT_CODE}
