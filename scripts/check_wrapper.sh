#!/bin/bash
set -x
# if [ "$SLURM_LOCALID" == "0" ]; then
source ../scripts/init.sh ${@: 2}
input_dir=../tests/input-extend
if [ -z $std_dir ]; then
    std_dir=../tests/output-extend
fi
for test in ${tests[*]}; do
    $MPIRUN_CONFIG ./main $input_dir/$test.qasm > $1/$test.log
    grep "Logger" $1/$test.log
done

set +x
set +e

for test in ${tests[*]}; do
    line=`cat $std_dir/$test.log | wc -l`
    echo $test
    grep -Ev "Logger|CLUSTER" $1/$test.log > tmp.log
    diff -q -B $std_dir/$test.log tmp.log || true
done

grep -Er "Logger.*Time" $1/*.log 
# fi