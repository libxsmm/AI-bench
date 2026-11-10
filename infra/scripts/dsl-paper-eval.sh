#!/bin/bash

#SBATCH --job-name=dsl-paper-eval
#SBATCH --partition=emr
#SBATCH --constraint="ddr5600&notrb"
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=8:00:00

set -e

# OpenMP configuration for EMR
export LD_PRELOAD=/lib64/libomp.so:$LD_PRELOAD
export OMP_NUM_THREADS=64
export KMP_AFFINITY=granularity=fine,compact,1,0
export NUMACTL_CMD="numactl -m 0 -C 0-63"

# Allocate cache directories for TorchInductor and Triton on scratch partition
export TORCHINDUCTOR_CACHE_DIR=/local_scratch/ti_cache
export TRITON_HOME=/local_scratch/tt_home
export TRITON_CACHE_DIR=/local_scratch/tt_cache

# AI-bench settings
export AIBENCH_WARMUP=50
export AIBENCH_REP=500
export AIBENCH_CPU_MIN_CACHE_NUKE_MIB=5000

# Build a fresh clone of AI-bench
cd /local_scratch
git clone --recurse-submodules --branch dsl-paper-eval https://github.com/libxsmm/AI-bench.git
cd AI-bench
uv sync --python 3.12 --extra cpu --extra triton-cpu --extra helion-mlir

# Filter problems of interest
AIBENCH_DIR=$PWD
export AIBENCH_SPECS_DIR=$AIBENCH_DIR/dsl-paper-eval-specs
mkdir -p $AIBENCH_SPECS_DIR/KernelBench/level2
cd problems/specs/KernelBench/level2
cp   9_Matmul_Subtract_Multiply_ReLU.yaml         \
    12_Gemm_Multiply_LeakyReLU.yaml               \
    14_Gemm_Divide_Sum_Scaling.yaml               \
    29_Matmul_Mish_Mish.yaml                      \
    30_Gemm_GroupNorm_Hardtanh.yaml               \
    33_Gemm_Scale_BatchNorm.yaml                  \
    37_Matmul_Swish_Sum_GroupNorm.yaml            \
    39_Gemm_Scale_BatchNorm.yaml                  \
    40_Matmul_Scaling_ResidualAdd.yaml            \
    41_Gemm_BatchNorm_GELU_ReLU.yaml              \
    53_Gemm_Scaling_Hardtanh_GELU.yaml            \
    56_Matmul_Sigmoid_Sum.yaml                    \
    59_Matmul_Swish_Scaling.yaml                  \
    62_Matmul_GroupNorm_LeakyReLU_Sum.yaml        \
    63_Gemm_ReLU_Divide.yaml                      \
    68_Matmul_Min_Subtract.yaml                   \
    70_Gemm_Sigmoid_Scaling_ResidualAdd.yaml      \
    76_Gemm_Add_ReLU.yaml                         \
    81_Gemm_Swish_Divide_Clamp_Tanh_Clamp.yaml    \
    86_Matmul_Divide_GELU.yaml                    \
    88_Gemm_GroupNorm_Swish_Multiply_Swish.yaml   \
    94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm.yaml  \
    95_Matmul_Add_Swish_Tanh_GELU_Hardtanh.yaml   \
    97_Matmul_BatchNorm_BiasAdd_Divide_Swish.yaml \
    $AIBENCH_SPECS_DIR/KernelBench/level2/
cd $AIBENCH_DIR

# Create output directory
OUTPUT_DIR="$SLURM_SUBMIT_DIR/dsl-paper-eval-$SLURM_JOB_PARTITION.$SLURM_JOB_ID"
mkdir -p $OUTPUT_DIR

cat << EOF
Running DSL paper evaluation with the following settings:
  SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION
  HOST: $(hostname -s)
  NO_TURBO: $(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
  OMP_NUM_THREADS: $OMP_NUM_THREADS
  KMP_AFFINITY: $KMP_AFFINITY
  NUMACTL_CMD: $NUMACTL_CMD

  TORCHINDUCTOR_CACHE_DIR: $TORCHINDUCTOR_CACHE_DIR
  TRITON_HOME: $TRITON_HOME
  TRITON_CACHE_DIR: $TRITON_CACHE_DIR

  AIBENCH_REV: $(git rev-parse HEAD)
  AIBENCH_WARMUP: $AIBENCH_WARMUP
  AIBENCH_REP: $AIBENCH_REP
  AIBENCH_CPU_MIN_CACHE_NUKE_MIB: $AIBENCH_CPU_MIN_CACHE_NUKE_MIB
  AIBENCH_SPECS_DIR: $AIBENCH_SPECS_DIR

  OUTPUT_DIR: $OUTPUT_DIR
EOF

# Enter virtual environment
source .venv/bin/activate

AIBENCH_COMMON="--bench --dtype bfloat16 --gflops"

# PyTorch reference (eager mode)
$NUMACTL_CMD ai-bench $AIBENCH_COMMON --csv "$OUTPUT_DIR/torch-eager.csv"

# TorchInductor
TORCHINDUCTOR_MAX_AUTOTUNE=0 \
$NUMACTL_CMD ai-bench $AIBENCH_COMMON --torch-compile --csv "$OUTPUT_DIR/torch-compile.csv"
TORCHINDUCTOR_MAX_AUTOTUNE=1 \
$NUMACTL_CMD ai-bench $AIBENCH_COMMON --torch-compile --csv "$OUTPUT_DIR/torch-compile-max-autotune.csv"

# Triton CPU
$NUMACTL_CMD ai-bench $AIBENCH_COMMON --triton        --csv "$OUTPUT_DIR/triton.csv"

# Helion via MLIR
$NUMACTL_CMD ai-bench $AIBENCH_COMMON --helion-mlir   --csv "$OUTPUT_DIR/helion-mlir.csv"
