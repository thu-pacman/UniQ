#!/bin/bash

root_name=../build/logs/gpu-pure-scaling-`date +%Y%m%d-%H%M%S`

mkdir -p $root_name

export input_dir="../../tests/input"
export std_dir="../../tests/output-pure"
export tests="bv_13 efficient_su2_11 iqp_14 hidden_shift_12 qaoa_14 qft_13 supremacy_12"

./compile.sh -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DEVALUATOR_PREPROCESS=off -DENABLE_OVERLAP=off -DUSE_MPI=off -DDISABLE_ASSERT=on -DMAT=7 -DGPU_BACKEND=group -DMODE=densitypure

name=$root_name/1gpu-cpy
mkdir -p $name
export CUDA_VISIBLE_DEVICES=0
MPIRUN_CONFIG="" ./check_wrapper.sh $name 2>&1 | tee $name/std.out

name=$root_name/2gpu-cpy
mkdir -p $name
export CUDA_VISIBLE_DEVICES=0,2
MPIRUN_CONFIG="" ./check_wrapper.sh $name 2>&1 | tee $name/std.out

name=$root_name/4gpu-cpy
mkdir -p $name
export CUDA_VISIBLE_DEVICES=0,1,2,3
MPIRUN_CONFIG="" ./check_wrapper.sh $name 2>&1 | tee $name/std.out

echo "Summary:"
grep -r "Compile Time" $root_name/1gpu-cpy/*.log | tee fig9_uniq.log
grep -r "Time Cost" $root_name/*/*.log | tee -a fig9_uniq.log
