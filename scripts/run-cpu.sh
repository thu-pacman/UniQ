#!/bin/bash
# baseline
# source ../scripts/init.sh -DHARDWARE=cpu -DGPU_BACKEND=group-serial -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=on -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=off -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on
# srun -n 32 --ntasks-per-node=32 bash -c 'OMP_NUM_THREADS=1 ../scripts/env.sh ./main ../tests/input/basis_change_25.qasm'

# cache-opt
# source ../scripts/init.sh -DHARDWARE=cpu -DGPU_BACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=on -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=off -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on
# srun -n 32 --ntasks-per-node=32 bash -c 'OMP_NUM_THREADS=1 ../scripts/env.sh ./main ../tests/input/basis_change_25.qasm'

# salloc -N 1 ./run-cpu.sh

source /opt/intel/oneapi/setvars.sh
CC=`which mpicc` CXX=`which mpiicpc` source ../scripts/init.sh -DHARDWARE=cpu -DGPU_BACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=on -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DLOCAL_QUBIT_SIZE=15
# srun -n 32 --ntasks-per-node=32 bash -c 'OMP_NUM_THREADS=1 ../scripts/env.sh ./main ../tests/input/basis_change_25.qasm'
mpirun -bootstrap slurm -n 64 -genv OMP_NUM_THREADS=1 ../scripts/cpu-bind.sh ./main ../tests/input/qaoa_25.qasm
