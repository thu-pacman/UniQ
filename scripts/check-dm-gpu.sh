#!/bin/bash
name=../build/logs/pure-`date +%Y%m%d-%H%M%S`
mkdir -p $name

# CUDA_VISIBLE_DEVICES=0 srun -p Big --exclusive ./check-dm-gpu.sh
# MPIRUN_CONFIG="" ./check_wrapper.sh $name -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=off -DMAT=7 -DGPU_BACKEND=group -DLOCAL_QUBIT_SIZE=5 -DCOALESCE=1 -DMODE=densityerr -DTHREAD_DEP=6

# CUDA_VISIBLE_DEVICES=0,1,2,3 srun -p Big --exclusive ./check-dm-gpu.sh
# MPIRUN_CONFIG="" ./check_wrapper.sh $name -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=off -DMAT=7 -DGPU_BACKEND=group -DLOCAL_QUBIT_SIZE=5 -DCOALESCE=1 -DMODE=densityerr -DTHREAD_DEP=6


# salloc -w nico1,nico2 -p Big --exclusive bash ./check-dm-gpu.sh
# MPIRUN_CONFIG="`which mpirun` -host nico1:1 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh"
# MPIRUN_CONFIG=$MPIRUN_CONFIG ./check_wrapper.sh $name -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DMAT=7 -DGPU_BACKEND=group -DLOCAL_QUBIT_SIZE=5 -DCOALESCE=1 -DMODE=densityerr 2>&1 | tee $name/std.out

# salloc -w nico1,nico2 -p Big --exclusive bash ./check-dm-gpu.sh
# MPIRUN_CONFIG="`which mpirun` -host nico1:4 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh"
# MPIRUN_CONFIG=$MPIRUN_CONFIG ./check_wrapper.sh $name -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DMAT=7 -DGPU_BACKEND=group -DLOCAL_QUBIT_SIZE=5 -DCOALESCE=1 -DMODE=densityerr 2>&1 | tee $name/std.out

# salloc -w nico1,nico2 -p Big --exclusive bash ./check-dm-gpu.sh
MPIRUN_CONFIG="`which mpirun` -host nico1:8,nico2:8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG ./check_wrapper.sh $name -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DMAT=7 -DGPU_BACKEND=group -DLOCAL_QUBIT_SIZE=5 -DCOALESCE=1 -DMODE=densityerr 2>&1 | tee $name/std.out