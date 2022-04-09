#!/bin/bash
root_name=../build/logs/cpu-sv-breakdown-`date +%Y%m%d-%H%M%S`

mkdir -p $root_name

export input_dir="../../tests/input"
export std_dir="../../tests/output"
export tests="bv_27 efficient_su2_28 hidden_shift_27 iqp_25 qaoa_26 qft_29 supremacy_28"

name=$root_name/1-no-opt
mkdir -p $name
./compile.sh -DHARDWARE=cpu -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on  -DGPU_BACKEND=group-serial -DENABLE_TRANSFORM=off -DUSE_AVX512=off _DUSE_AVX2=off
MPIRUN_CONFIG="`which mpirun` -n 1 -npernode 1 --bind-to none -x OMP_NUM_THREADS=64 `pwd`/bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG ./check_wrapper.sh $name 2>&1 | tee $name/std.out

name=$root_name/2-general
mkdir -p $name
./compile.sh -DHARDWARE=cpu -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on  -DGPU_BACKEND=group-serial -DENABLE_TRANSFORM=on -DUSE_AVX512=off _DUSE_AVX2=off
MPIRUN_CONFIG="`which mpirun` -n 1 -npernode 1 --bind-to none -x OMP_NUM_THREADS=64 `pwd`/bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG ./check_wrapper.sh $name 2>&1 | tee $name/std.out

name=$root_name/3-cache
mkdir -p $name
./compile.sh -DHARDWARE=cpu -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on  -DGPU_BACKEND=group -DENABLE_TRANSFORM=on -DUSE_AVX512=off _DUSE_AVX2=off
MPIRUN_CONFIG="`which mpirun` -n 1 -npernode 1 --bind-to none -x OMP_NUM_THREADS=64 `pwd`/bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG ./check_wrapper.sh $name 2>&1 | tee $name/std.out

name=$root_name/4-avx2
mkdir -p $name
./compile.sh -DHARDWARE=cpu -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on  -DGPU_BACKEND=group -DENABLE_TRANSFORM=on -DUSE_AVX512=off _DUSE_AVX2=on
MPIRUN_CONFIG="`which mpirun` -n 1 -npernode 1 --bind-to none -x OMP_NUM_THREADS=64 `pwd`/bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG ./check_wrapper.sh $name 2>&1 | tee $name/std.out

name=$root_name/5-avx512
mkdir -p $name
./compile.sh -DHARDWARE=cpu -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on  -DGPU_BACKEND=group -DENABLE_TRANSFORM=on -DUSE_AVX512=on _DUSE_AVX2=off
MPIRUN_CONFIG="`which mpirun` -n 1 -npernode 1 --bind-to none -x OMP_NUM_THREADS=64 `pwd`/bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG ./check_wrapper.sh $name 2>&1 | tee $name/std.out

echo "Summary:"
grep -r "Time Cost" $root_name/*/*.log | tee -a fig12_fig13_uniq.log

