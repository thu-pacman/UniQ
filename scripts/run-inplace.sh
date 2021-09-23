#!/bin/bash
source ../scripts/init.sh -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=on -DMICRO_BENCH=on -DUSE_DOUBLE=on -DDISABLE_ASSERT=off -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=on -DUSE_MPI=on -DMAT=7 -DINPLACE=15
`which mpirun` -host nico1:2,nico2:2 -x GPUPerRank=4 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input/hidden_shift_28.qasm
