#!/bin/bash
source ../scripts/init.sh -DBACKEND=group -DMODE=densitypure -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=on -DMICRO_BENCH=on -DUSE_DOUBLE=on -DDISABLE_ASSERT=off -DENABLE_OVERLAP=on -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=on -DUSE_MPI=on -DMAT=7
set -x
set -e
tests="bv_13 hidden_shift_12 ising_9 qaoa_10 quantum_volume_11 supremacy_12 basis_change_9 basis_change_10 basis_change_11 basis_change_12 basis_change_13 basis_change_14"
for ngpu in 1 2 4 8 16; do
    for test in ${tests[*]}; do
        echo $ngpu $test
        srun -n $ngpu -p Big --ntasks-per-node=8 bash -c 'GPUPerRank=1 ../scripts/env.sh ../scripts/gpu-bind.sh ./main ../tests/dmpure/'"${test}"'.qasm' 2>&1 | tee logs/density_pure_v100/${test}_${ngpu}.log
    done
done
