#!/bin/bash
root_name=../build/logs/cpu-density-`date +%Y%m%d-%H%M%S`
mkdir -p $root_name

export input_dir="../../tests/input"
export std_dir="../../tests/output-pure"
export tests="bv_13 efficient_su2_11 hidden_shift_12 iqp_14 qaoa_14 qft_13 supremacy_12"


name=$root_name/gate
mkdir -p $name
./compile.sh -DHARDWARE=cpu -DGPU_BACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DUSE_AVX512=on -DLOCAL_QUBIT_SIZE=10 -DMODE=densitypure
MPIRUN_CONFIG="`which mpirun` -n 1 -npernode 1 --bind-to none -x OMP_NUM_THREADS=64 `pwd`/bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG ./check_wrapper.sh $name 2>&1 | tee $name/std.out

name=$root_name/kraus
mkdir -p $name
./compile.sh -DHARDWARE=cpu -DGPU_BACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DUSE_AVX512=on -DLOCAL_QUBIT_SIZE=5 -DCOALESCE=1 -DMODE=densityerr -DUSE_ALL_TO_ALL=on
MPIRUN_CONFIG="`which mpirun` -n 1 -npernode 1 --bind-to none -x OMP_NUM_THREADS=64 `pwd`/bind.sh"
export std_dir="../../tests/output-err"
MPIRUN_CONFIG=$MPIRUN_CONFIG ./check_wrapper.sh $name 2>&1 | tee $name/std.out

echo "Summary:"
grep -r "Logger\[0\]: Time Cost" $root_name/*/*.log | tee fig16_uniq.log
