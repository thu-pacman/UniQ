#!/bin/bash
source ../scripts/init.sh -DBACKEND=group -DMODE=densitypure -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=on -DMICRO_BENCH=on -DUSE_DOUBLE=on -DDISABLE_ASSERT=off -DENABLE_OVERLAP=on -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=on -DUSE_MPI=on -DMAT=7
srun -n 8 --ntasks-per-node=8 bash -c 'GPUPerRank=1 UCX_MEMTYPE_CACHE=n ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/final/qaoa_14.qasm'
