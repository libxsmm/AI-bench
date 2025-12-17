#!/usr/bin/env bash
#
# Script for CI - XPU job.
#
# Run KernelBench on Intel GPU.

SCRIPTS_DIR=$(realpath $(dirname $0))

die_syntax() {
  echo "Syntax: $0 [-b (torch|triton)]"
  echo ""
  echo "  -b: Optional, backend to use (default: torch)"
  exit 1
}

# Backends
BENCH_BACKEND_TORCH="torch"
BENCH_BACKEND_TRITON="triton"

# Options
BENCH_BACKEND="torch"
while getopts "b:" arg; do
  case ${arg} in
    b)
      if [ "${OPTARG}" == "${BENCH_BACKEND_TORCH}" ]; then
        BENCH_BACKEND="${BENCH_BACKEND_TORCH}"
      elif [ "${OPTARG}" == "${BENCH_BACKEND_TRITON}" ]; then
        BENCH_BACKEND="${BENCH_BACKEND_TRITON}"
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
source /swtools/intel/2025.2.0/setvars.sh --force
source /swtools/intel-gpu/latest/intel_gpu_vars.sh
echo ""

echo "--- Setup project"
git submodule update --init

pip install --upgrade --user uv
AI_BENCH_UV=${HOME}/.local/bin/uv

${AI_BENCH_UV} sync --extra xpu --link-mode copy
echo ""

# Run benchmark
echo "--- Run KernelBench (${BENCH_BACKEND})"

BENCH_FLAGS="--xpu --bench"
if [[ "${BENCH_BACKEND}" == "${BENCH_BACKEND_TRITON}" ]]; then
  BENCH_FLAGS="${BENCH_FLAGS} --triton"
fi

${AI_BENCH_UV} run python ${SCRIPTS_DIR}/run_kernel_bench.py ${BENCH_FLAGS}
echo ""
