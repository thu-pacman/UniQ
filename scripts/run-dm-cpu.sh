#!/bin/bash
source /opt/intel/oneapi/setvars.sh
which mpicc
which mpiicpc
CC=`which mpicc` CXX=`which mpiicpc` source ../scripts/init.sh -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=on -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=off -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DMAT=7 -DGPU_BACKEND=group -DHARDWARE=cpu -DLOCAL_QUBIT_SIZE=5 -DCOALESCE=1 -DMODE=densityerr
# OMP_NUM_THREADS=64 ./main ../tests/input-extend/bv_13.qasm
mpirun -bootstrap slurm -n 1 -ppn 64 -genv OMP_NUM_THREADS=64 ../scripts/cpu-bind.sh ./main ../tests/input-extend/supremacy_12.qasm