#!/bin/bash
root_name=../build/logs/cpu-sv-strong-`date +%Y%m%d-%H%M%S`
mkdir -p $root_name

export input_dir="../../tests/input"
export std_dir="../../tests/output"
export tests="bv_27 efficient_su2_28 hidden_shift_27 iqp_25 qaoa_26 qft_29 supremacy_28"

./compile.sh -DHARDWARE=cpu -DGPU_BACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DUSE_AVX512=on

name=$root_name/1node
mkdir -p $name
MPIRUN_CONFIG="`which mpirun` -n 1 -npernode 1 --bind-to none -x OMP_NUM_THREADS=64 `pwd`/bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG ./check_wrapper.sh $name 2>&1 | tee $name/std.out

name=$root_name/2node
mkdir -p $name
MPIRUN_CONFIG="`which mpirun` -n 2 -npernode 1 --bind-to none -x OMP_NUM_THREADS=64 `pwd`/bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG ./check_wrapper.sh $name 2>&1 | tee $name/std.out

echo "Summary:"
grep -r "Logger\[0\]: Time Cost" $root_name/*/*.log | tee fig14_uniq.log
