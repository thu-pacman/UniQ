#!/bin/bash

# use srun

root_name=../build/logs/gpu-sv-scaling-`date +%Y%m%d-%H%M%S`

mkdir -p $root_name

name=$root_name/1gpu-cpy
mkdir -p $name
export CUDA_VISIBLE_DEVICES=0
MPIRUN_CONFIG="" ../scripts/check_wrapper.sh $name -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DEVALUATOR_PREPROCESS=off -DENABLE_OVERLAP=off -DDISABLE_ASSERT=on -DMAT=7 -DUSE_MPI=off -DGPU_BACKEND=group 2>&1 | tee $name/std.out

name=$root_name/2gpu-cpy
mkdir -p $name
export CUDA_VISIBLE_DEVICES=0,1
MPIRUN_CONFIG="" ../scripts/./check_wrapper.sh $name -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DEVALUATOR_PREPROCESS=off -DENABLE_OVERLAP=off -DDISABLE_ASSERT=on -DMAT=7 -DUSE_MPI=off -DGPU_BACKEND=group 2>&1 | tee $name/std.out

name=$root_name/4gpu-cpy
mkdir -p $name
export CUDA_VISIBLE_DEVICES=0,1,2,3
MPIRUN_CONFIG="" ../scripts/./check_wrapper.sh $name -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DEVALUATOR_PREPROCESS=off -DENABLE_OVERLAP=off -DDISABLE_ASSERT=on -DMAT=7 -DUSE_MPI=off -DGPU_BACKEND=group 2>&1 | tee $name/std.out

name=$root_name/8gpu-cpy
mkdir -p $name
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MPIRUN_CONFIG="" ../scripts/check_wrapper.sh $name -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DEVALUATOR_PREPROCESS=off -DENABLE_OVERLAP=off -DDISABLE_ASSERT=on -DMAT=7 -DUSE_MPI=off -DGPU_BACKEND=group 2>&1 | tee $name/std.out
