#!/bin/bash

root_name=../build/logs/gpu-pure-scaling-nvprof-`date +%Y%m%d-%H%M%S`

mkdir -p $root_name

export input_dir="../../tests/input"
export std_dir="../../tests/output-pure"
export tests="bv_13 efficient_su2_11 iqp_14 hidden_shift_12 qaoa_14 qft_13 supremacy_12"

./compile.sh -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DEVALUATOR_PREPROCESS=off -DENABLE_OVERLAP=off -DUSE_MPI=off -DDISABLE_ASSERT=on -DMAT=7 -DGPU_BACKEND=group -DMODE=densitypure

mkdir -p $root_name/nvprof
cd ../build

for test in ${tests[*]}; do
    CUDA_VISIBLE_DEVICES=0,2 nvprof -o $root_name/nvprof/${test}_2gpu.sql ./main ${input_dir}/$test.qasm 2>&1 | tee ${root_name}/${test}_2gpu.log
done

for test in ${tests[*]}; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 nvprof -o $root_name/nvprof/${test}_4gpu.sql ./main ${input_dir}/$test.qasm 2>&1 | tee ${root_name}/${test}_4gpu.log
done

cd ../artifact-evaluation
python3 parse_profile.py ${root_name}/nvprof 2>&1 | tee fig10_uniq.log
