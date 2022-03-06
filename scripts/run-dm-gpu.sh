#!/bin/bash
source ../scripts/init.sh -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=on -DMICRO_BENCH=on -DUSE_DOUBLE=on -DDISABLE_ASSERT=off -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=off -DMAT=7 -DGPU_BACKEND=group -DMODE=densityerr
CUDA_VISIBLE_DEVICES=0 ./main xxx # ../tests/input-extend/qaoa_26.qasm
