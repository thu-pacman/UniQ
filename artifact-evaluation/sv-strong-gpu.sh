#!/bin/bash

root_name=../build/logs/gpu-sv-scaling-`date +%Y%m%d-%H%M%S`

mkdir -p $root_name

export input_dir="../../tests/input"
export std_dir="../../tests/output"
export tests="bv_27 efficient_su2_28 hidden_shift_27 iqp_25 qaoa_26 qft_29 supremacy_28"

./compile.sh -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DEVALUATOR_PREPROCESS=off -DENABLE_OVERLAP=off -DDISABLE_ASSERT=on -DMAT=7 -DUSE_MPI=off -DGPU_BACKEND=group

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
grep -r "Compile Time" $root_name/1gpu-cpy/*.log | tee fig8_uniq.log
grep -r "Time Cost" $root_name/*/*.log | tee -a fig8_uniq.log
