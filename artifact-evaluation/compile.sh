#!/bin/bash
set -u
set -e

mkdir -p $HYQUAS_ROOT/build
cd $HYQUAS_ROOT/build
rm CMakeCache.txt || true
cmake $* ..
make clean
make -j
