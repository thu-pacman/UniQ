#!/bin/bash

root_name=../build/logs/gpu-err-scaling-`date +%Y%m%d-%H%M%S`

mkdir -p $root_name

export input_dir="../../tests/input"
export std_dir="../../tests/output-err"
export tests="bv_13 efficient_su2_11 iqp_14 hidden_shift_12 qaoa_14 qft_13 supremacy_12"

./compile.sh -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=off -DMAT=7 -DGPU_BACKEND=group -DLOCAL_QUBIT_SIZE=5 -DCOALESCE=1 -DMODE=densityerr -DTHREAD_DEP=6

name=$root_name/1gpu-cpy
mkdir -p $name
export CUDA_VISIBLE_DEVICES=0
MPIRUN_CONFIG="" ./check_wrapper.sh $name  2>&1 | tee $name/std.out

name=$root_name/4gpu-cpy
mkdir -p $name
export CUDA_VISIBLE_DEVICES=0,1,2,3
MPIRUN_CONFIG="" ./check_wrapper.sh $name  2>&1 | tee $name/std.out

echo "Summary:"
grep -r "Compile Time" $root_name/1gpu-cpy/*.log | tee fig12_uniq.log
grep -r "Time Cost" $root_name/*/*.log | tee -a fig12_uniq.log
