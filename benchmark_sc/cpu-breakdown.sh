name=../build/logs/cpu-breakdown-`date +%Y%m%d-%H%M%S`
mkdir -p $name

#no-opt
mkdir -p $name/no-opt
MPIRUN_CONFIG="`which mpirun` -bootstrap slurm -n 1 -ppn 1 -genv OMP_NUM_THREADS=64 ../scripts/cpu-bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG CC=`which mpicc` CXX=`which mpiicpc` ./check_wrapper.sh $name/no-opt -DHARDWARE=cpu -DGPU_BACKEND=group-serial -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on 2>&1 | tee $name/no-opt/std.out

#no-avx
mkdir -p $name/no-avx
MPIRUN_CONFIG="`which mpirun` -bootstrap slurm -n 1 -ppn 1 -genv OMP_NUM_THREADS=64 ../scripts/cpu-bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG CC=`which mpicc` CXX=`which mpiicpc` ./check_wrapper.sh $name/no-avx -DHARDWARE=cpu -DGPU_BACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=off -DUSE_DOUBLE=on -DDISABLE_ASSERT=on -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=off -DUSE_MPI=on -DUSE_AVX512=off 2>&1 | tee $name/no-avx/std.out

#avx: use data from scaling test
