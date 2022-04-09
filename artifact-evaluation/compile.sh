#!/bin/bash
set -u
set -e
set -x

mkdir -p ../build
cd ../build
rm CMakeCache.txt || true
cmake $* ..
make clean
make -j
