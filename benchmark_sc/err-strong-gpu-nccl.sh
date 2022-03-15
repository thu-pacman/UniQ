#!/bin/bash

# salloc

root_name=../build/logs/gpu-err-scaling-nccl-`date +%Y%m%d-%H%M%S`

mkdir -p $root_name

export std_dir="../tests/output-err"
export tests="bv_13 efficient_su2_11 iqp_14 hidden_shift_12 qaoa_14 qft_13 supremacy_12"

name=$root_name/1gpu-nccl
mkdir -p $name
MPIRUN_CONFIG="`which mpirun` -n 1 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG ../scripts/check_wrapper.sh $name -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DMAT=7 -DGPU_BACKEND=group -DLOCAL_QUBIT_SIZE=5 -DCOALESCE=1 -DMODE=densityerr -DTHREAD_DEP=6

name=$root_name/4gpu-nccl
mkdir -p $name
MPIRUN_CONFIG="`which mpirun` -n 4 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG ../scripts/check_wrapper.sh $name -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DMAT=7 -DGPU_BACKEND=group -DLOCAL_QUBIT_SIZE=5 -DCOALESCE=1 -DMODE=densityerr -DTHREAD_DEP=6

name=$root_name/16gpu-nccl
mkdir -p $name
MPIRUN_CONFIG="`which mpirun` -n 16 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG ../scripts/check_wrapper.sh $name -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DMAT=7 -DGPU_BACKEND=group -DLOCAL_QUBIT_SIZE=5 -DCOALESCE=1 -DMODE=densityerr -DTHREAD_DEP=6
