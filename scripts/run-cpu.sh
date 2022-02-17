#!/bin/bash
source ../scripts/init.sh -DHARDWARE=cpu -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=on -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=off -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=off
CUDA_VISIBLE_DEVICES=0 ./main ../tests/input/supremacy_28.qasm
