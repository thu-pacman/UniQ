#!/bin/bash
set -x

# run all tests
cd ../build
for test in ${tests[*]}; do
    $MPIRUN_CONFIG ./main $input_dir/$test.qasm > $1/$test.log
    grep "Logger" $1/$test.log
done
set +x
set +e

# check results
for test in ${tests[*]}; do
    line=`cat $std_dir/$test.log | wc -l`
    echo $test
    grep -Ev "Logger|CLUSTER|UCX" $1/$test.log > tmp.log
    diff -q -B $std_dir/$test.log tmp.log || true
done
