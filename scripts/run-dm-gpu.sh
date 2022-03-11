#!/bin/bash
# source ../scripts/init.sh -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=on -DMICRO_BENCH=on -DUSE_DOUBLE=on -DDISABLE_ASSERT=off -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=off -DMAT=7 -DGPU_BACKEND=group -DLOCAL_QUBIT_SIZE=5 -DCOALESCE=1 -DMODE=densityerr -DTHREAD_DEP=6
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./main ../tests/input-extend/supremacy_12.qasm

source ../scripts/init.sh -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=on -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=off -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DMAT=7 -DGPU_BACKEND=group -DLOCAL_QUBIT_SIZE=5 -DCOALESCE=1 -DMODE=densityerr -DTHREAD_DEP=6
srun -n 4 --ntasks-per-node=4 bash -c 'GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/input-extend/bv_13.qasm'
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./main ../tests/input-extend/supremacy_12.qasm
