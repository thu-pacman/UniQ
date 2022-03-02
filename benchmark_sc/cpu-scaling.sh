#!/bin/bash
name=../build/logs/cpu-scaling-`date +%Y%m%d-%H%M%S`

mkdir -p $name

mkdir -p $name/1core
MPIRUN_CONFIG="`which mpirun` -bootstrap slurm -n 1 -ppn 1 -genv OMP_NUM_THREADS=1 ../scripts/cpu-bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG CC=`which mpicc` CXX=`which mpiicpc` ./check_wrapper.sh $name/1core -DHARDWARE=cpu -DGPU_BACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DUSE_AVX512=on 2>&1 | tee $name/1core/std.out

mkdir -p $name/2core
MPIRUN_CONFIG="`which mpirun` -bootstrap slurm -n 1 -ppn 1 -genv OMP_NUM_THREADS=2 ../scripts/cpu-bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG CC=`which mpicc` CXX=`which mpiicpc` ./check_wrapper.sh $name/2core -DHARDWARE=cpu -DGPU_BACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DUSE_AVX512=on 2>&1 | tee $name/2core/std.out

mkdir -p $name/4core
MPIRUN_CONFIG="`which mpirun` -bootstrap slurm -n 1 -ppn 1 -genv OMP_NUM_THREADS=4 ../scripts/cpu-bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG CC=`which mpicc` CXX=`which mpiicpc` ./check_wrapper.sh $name/4core -DHARDWARE=cpu -DGPU_BACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DUSE_AVX512=on 2>&1 | tee $name/4core/std.out
 
mkdir -p $name/8core
MPIRUN_CONFIG="`which mpirun` -bootstrap slurm -n 1 -ppn 1 -genv OMP_NUM_THREADS=8 ../scripts/cpu-bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG CC=`which mpicc` CXX=`which mpiicpc` ./check_wrapper.sh $name/8core -DHARDWARE=cpu -DGPU_BACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DUSE_AVX512=on 2>&1 | tee $name/8core/std.out

mkdir -p $name/16core
MPIRUN_CONFIG="`which mpirun` -bootstrap slurm -n 1 -ppn 1 -genv OMP_NUM_THREADS=16 ../scripts/cpu-bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG CC=`which mpicc` CXX=`which mpiicpc` ./check_wrapper.sh $name/16core -DHARDWARE=cpu -DGPU_BACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DUSE_AVX512=on 2>&1 | tee $name/16core/std.out

mkdir -p $name/32core
MPIRUN_CONFIG="`which mpirun` -bootstrap slurm -n 1 -ppn 1 -genv OMP_NUM_THREADS=32 ../scripts/cpu-bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG CC=`which mpicc` CXX=`which mpiicpc` ./check_wrapper.sh $name/32core -DHARDWARE=cpu -DGPU_BACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DUSE_AVX512=on 2>&1 | tee $name/32core/std.out

mkdir -p $name/64core
MPIRUN_CONFIG="`which mpirun` -bootstrap slurm -n 1 -ppn 1 -genv OMP_NUM_THREADS=64 ../scripts/cpu-bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG CC=`which mpicc` CXX=`which mpiicpc` ./check_wrapper.sh $name/64core -DHARDWARE=cpu -DGPU_BACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DUSE_AVX512=on 2>&1 | tee $name/64core/std.out

mkdir -p $name/128core
MPIRUN_CONFIG="`which mpirun` -bootstrap slurm -n 2 -ppn 1 -genv OMP_NUM_THREADS=64 ../scripts/cpu-bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG CC=`which mpicc` CXX=`which mpiicpc` ./check_wrapper.sh $name/128core -DHARDWARE=cpu -DGPU_BACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DUSE_AVX512=on 2>&1 | tee $name/128core/std.out

# TODO: script for 4 node
