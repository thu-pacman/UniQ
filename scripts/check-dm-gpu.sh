#!/bin/bash
name=../build/logs/pure-`date +%Y%m%d-%H%M%S`
mkdir -p $name

MPIRUN_CONFIG="" ./check_wrapper.sh $name -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=off -DMAT=7 -DGPU_BACKEND=group -DLOCAL_QUBIT_SIZE=5 -DCOALESCE=1 -DMODE=densityerr -DTHREAD_DEP=6

