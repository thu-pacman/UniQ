#!/bin/bash

name=../build/logs/cpu-`date +%Y%m%d-%H%M%S`
mkdir -p $name

export std_dir="../tests/output-pure"
export tests="bv_13 efficient_su2_11 iqp_14 hidden_shift_12 qaoa_14 qft_13 supremacy_12"
source /opt/intel/oneapi/setvars.sh

MPIRUN_CONFIG="`which mpirun` -bootstrap slurm -n 64 -ppn 64 -genv OMP_NUM_THREADS=1 ../scripts/cpu-bind.sh"

MPIRUN_CONFIG=$MPIRUN_CONFIG CC=`which mpicc` CXX=`which mpiicpc` ./check_wrapper.sh $name -DHARDWARE=cpu -DGPU_BACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DUSE_AVX512=on -DGPU_BACKEND=group -DHARDWARE=cpu -DLOCAL_QUBIT_SIZE=5 -DCOALESCE=1 -DMODE=densityerr 2>&1 | tee $name/std.out