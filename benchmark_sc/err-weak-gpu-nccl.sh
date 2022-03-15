#!/bin/bash

# salloc

set -x

root_name=../build/logs/gpu-err-weak-nccl-`date +%Y%m%d-%H%M%S`

mkdir -p $root_name

source ../scripts/init.sh -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DMAT=7 -DGPU_BACKEND=group -DLOCAL_QUBIT_SIZE=5 -DCOALESCE=1 -DMODE=densityerr

cd ../build

`which mpirun` -n 1 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_11.qasm 2>&1 | tee $root_name/qft_11_01gpu.log
`which mpirun` -n 4 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_11.qasm 2>&1 | tee $root_name/qft_11_04gpu.log
`which mpirun` -n 16 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_11.qasm 2>&1 | tee $root_name/qft_11_16gpu.log

`which mpirun` -n 1 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_12.qasm 2>&1 | tee $root_name/qft_12_01gpu.log
`which mpirun` -n 4 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_12.qasm 2>&1 | tee $root_name/qft_12_04gpu.log
`which mpirun` -n 16 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_12.qasm 2>&1 | tee $root_name/qft_12_16gpu.log

`which mpirun` -n 1 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_13.qasm 2>&1 | tee $root_name/qft_13_01gpu.log
`which mpirun` -n 4 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_13.qasm 2>&1 | tee $root_name/qft_13_04gpu.log
`which mpirun` -n 16 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_13.qasm 2>&1 | tee $root_name/qft_13_16gpu.log

`which mpirun` -n 1 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_14.qasm 2>&1 | tee $root_name/qft_14_01gpu.log
`which mpirun` -n 4 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_14.qasm 2>&1 | tee $root_name/qft_14_04gpu.log
`which mpirun` -n 16 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_14.qasm 2>&1 | tee $root_name/qft_14_16gpu.log

`which mpirun` -n 4 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_15.qasm 2>&1 | tee $root_name/qft_15_04gpu.log
`which mpirun` -n 16 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_15.qasm 2>&1 | tee $root_name/qft_15_16gpu.log

`which mpirun` -n 16 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_16.qasm 2>&1 | tee $root_name/qft_16_16gpu.log


source ../scripts/init.sh -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DMAT=7 -DGPU_BACKEND=group -DLOCAL_QUBIT_SIZE=5 -DCOALESCE=1 -DMODE=densityerr -DINPLACE=8 -DMAX_SLICE=12
cd ../build
`which mpirun` -n 1 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_15.qasm 2>&1 | tee $root_name/qft_15_01gpu.log
`which mpirun` -n 4 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_16.qasm 2>&1 | tee $root_name/qft_16_04gpu.log
`which mpirun` -n 16 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_17.qasm 2>&1 | tee $root_name/qft_17_16gpu.log