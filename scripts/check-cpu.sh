#!/bin/bash
name=../build/logs/cpu-`date +%Y%m%d-%H%M%S`
mkdir -p $name

# MPIRUN_CONFIG="`which mpirun` -n 32 -host nico3:32 -x OMP_NUM_THREADS=1 ../scripts/cpu-bind.sh"
# MPIRUN_CONFIG=$MPIRUN_CONFIG ./check_wrapper.sh $name -DHARDWARE=cpu -DGPU_BACKEND=group-serial -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=off -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on 2>&1 | tee $name/std.out

# MPIRUN_CONFIG="`which mpirun` -n 64 -genv OMP_NUM_THREADS=1 ../scripts/cpu-bind.sh"
# MPIRUN_CONFIG=$MPIRUN_CONFIG CC=`which mpicc` CXX=`which mpiicpc` ./check_wrapper.sh $name -DHARDWARE=cpu -DGPU_BACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on 2>&1 | tee $name/std.out

MPIRUN_CONFIG="`which mpirun` -n 1 -genv OMP_NUM_THREADS=64 ../scripts/cpu-bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG CC=`which mpicc` CXX=`which mpiicpc` ./check_wrapper.sh $name -DHARDWARE=cpu -DGPU_BACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on 2>&1 | tee $name/std.out
 