#!/bin/bash
set -x

root_name=../build/logs/gpu-sv-scaling-weak-`date +%Y%m%d-%H%M%S`

mkdir -p $root_name

source ../scripts/init.sh -DGPU_BACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DMAT=7

cd ../build

`which mpirun` -n 1 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_26.qasm 2>&1 | tee $root_name/qft_26_01gpu.log
`which mpirun` -n 2 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_26.qasm 2>&1 | tee $root_name/qft_26_02gpu.log
`which mpirun` -n 4 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_26.qasm 2>&1 | tee $root_name/qft_26_04gpu.log
`which mpirun` -n 8 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_26.qasm 2>&1 | tee $root_name/qft_26_08gpu.log
`which mpirun` -n 16 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_26.qasm 2>&1 | tee $root_name/qft_26_16gpu.log

`which mpirun` -n 1 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_27.qasm 2>&1 | tee $root_name/qft_27_01gpu.log
`which mpirun` -n 2 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_27.qasm 2>&1 | tee $root_name/qft_27_02gpu.log
`which mpirun` -n 4 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_27.qasm 2>&1 | tee $root_name/qft_27_04gpu.log
`which mpirun` -n 8 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_27.qasm 2>&1 | tee $root_name/qft_27_08gpu.log
`which mpirun` -n 16 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_27.qasm 2>&1 | tee $root_name/qft_27_16gpu.log

`which mpirun` -n 1 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_28.qasm 2>&1 | tee $root_name/qft_28_01gpu.log
`which mpirun` -n 2 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_28.qasm 2>&1 | tee $root_name/qft_28_02gpu.log
`which mpirun` -n 4 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_28.qasm 2>&1 | tee $root_name/qft_28_04gpu.log
`which mpirun` -n 8 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_28.qasm 2>&1 | tee $root_name/qft_28_08gpu.log
`which mpirun` -n 16 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_28.qasm 2>&1 | tee $root_name/qft_28_16gpu.log

`which mpirun` -n 1 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_29.qasm 2>&1 | tee $root_name/qft_29_01gpu.log
`which mpirun` -n 2 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_29.qasm 2>&1 | tee $root_name/qft_29_02gpu.log
`which mpirun` -n 4 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_29.qasm 2>&1 | tee $root_name/qft_29_04gpu.log
`which mpirun` -n 8 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_29.qasm 2>&1 | tee $root_name/qft_29_08gpu.log
`which mpirun` -n 16 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_29.qasm 2>&1 | tee $root_name/qft_29_16gpu.log

`which mpirun` -n 1 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_29.qasm 2>&1 | tee $root_name/qft_29_01gpu.log
`which mpirun` -n 2 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_29.qasm 2>&1 | tee $root_name/qft_29_02gpu.log
`which mpirun` -n 4 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_29.qasm 2>&1 | tee $root_name/qft_29_04gpu.log
`which mpirun` -n 8 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_29.qasm 2>&1 | tee $root_name/qft_29_08gpu.log
`which mpirun` -n 16 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_29.qasm 2>&1 | tee $root_name/qft_29_16gpu.log

`which mpirun` -n 1 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_30.qasm 2>&1 | tee $root_name/qft_30_01gpu.log
`which mpirun` -n 2 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_30.qasm 2>&1 | tee $root_name/qft_30_02gpu.log
`which mpirun` -n 4 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_30.qasm 2>&1 | tee $root_name/qft_30_04gpu.log
`which mpirun` -n 8 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_30.qasm 2>&1 | tee $root_name/qft_30_08gpu.log
`which mpirun` -n 16 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_30.qasm 2>&1 | tee $root_name/qft_30_16gpu.log

`which mpirun` -n 4 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_31.qasm 2>&1 | tee $root_name/qft_31_04gpu.log
`which mpirun` -n 8 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_31.qasm 2>&1 | tee $root_name/qft_31_08gpu.log
`which mpirun` -n 16 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_31.qasm 2>&1 | tee $root_name/qft_31_16gpu.log

`which mpirun` -n 8 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_32.qasm 2>&1 | tee $root_name/qft_32_08gpu.log
`which mpirun` -n 16 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_32.qasm 2>&1 | tee $root_name/qft_32_16gpu.log

`which mpirun` -n 16 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_33.qasm 2>&1 | tee $root_name/qft_33_16gpu.log


source ../scripts/init.sh -DGPU_BACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DMAT=7 -DINPLACE=15

cd ../build
`which mpirun` -n 2 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_31.qasm 2>&1 | tee $root_name/qft_31_02gpu.log
`which mpirun` -n 4 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_32.qasm 2>&1 | tee $root_name/qft_32_04gpu.log
`which mpirun` -n 8 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_33.qasm 2>&1 | tee $root_name/qft_33_08gpu.log
`which mpirun` -n 16 -npernode 8 -x GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-qft/qft_34.qasm 2>&1 | tee $root_name/qft_34_16gpu.log
