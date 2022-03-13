#!/bin/bash

# salloc bash ./*.sh

root_name=../build/logs/gpu-pure-scaling-nccl-`date +%Y%m%d-%H%M%S`

mkdir -p $root_name

export std_dir="../tests/output-pure"
export tests="bv_13 efficient_su2_11 iqp_14 hidden_shift_12 qaoa_14 qft_13 supremacy_12"

name=$root_name/1gpu-nccl
mkdir -p $name
MPIRUN_CONFIG="`which mpirun` -n 1 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG ../scripts/check_wrapper.sh $name -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DEVALUATOR_PREPROCESS=off -DENABLE_OVERLAP=off -DUSE_MPI=on -DDISABLE_ASSERT=on -DMAT=7 -DGPU_BACKEND=group -DMODE=densitypure 2>&1 | tee $name/std.out

name=$root_name/2gpu-nccl
mkdir -p $name
MPIRUN_CONFIG="`which mpirun` -n 2 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG ../scripts/check_wrapper.sh $name -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DEVALUATOR_PREPROCESS=off -DENABLE_OVERLAP=off -DUSE_MPI=on -DDISABLE_ASSERT=on -DMAT=7 -DGPU_BACKEND=group -DMODE=densitypure 2>&1 | tee $name/std.out

name=$root_name/4gpu-nccl
mkdir -p $name
MPIRUN_CONFIG="`which mpirun` -n 4 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG ../scripts/check_wrapper.sh $name -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DEVALUATOR_PREPROCESS=off -DENABLE_OVERLAP=off -DUSE_MPI=on -DDISABLE_ASSERT=on -DMAT=7 -DGPU_BACKEND=group -DMODE=densitypure 2>&1 | tee $name/std.out

name=$root_name/8gpu-nccl
mkdir -p $name
MPIRUN_CONFIG="`which mpirun` -n 8 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG ../scripts/check_wrapper.sh $name -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DEVALUATOR_PREPROCESS=off -DENABLE_OVERLAP=off -DUSE_MPI=on -DDISABLE_ASSERT=on -DMAT=7 -DGPU_BACKEND=group -DMODE=densitypure 2>&1 | tee $name/std.out

name=$root_name/16gpu-nccl
mkdir -p $name
MPIRUN_CONFIG="`which mpirun` -n 16 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG ../scripts/check_wrapper.sh $name -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DEVALUATOR_PREPROCESS=off -DENABLE_OVERLAP=off -DUSE_MPI=on -DDISABLE_ASSERT=on -DMAT=7 -DGPU_BACKEND=group -DMODE=densitypure 2>&1 | tee $name/std.out
