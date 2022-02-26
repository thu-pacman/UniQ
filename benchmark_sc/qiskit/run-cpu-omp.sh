#!/bin/bash
set -x
OMP_NUM_THREADS=1 python sv-cpu-omp.py
OMP_NUM_THREADS=2 python sv-cpu-omp.py
OMP_NUM_THREADS=4 python sv-cpu-omp.py
OMP_NUM_THREADS=8 python sv-cpu-omp.py
OMP_NUM_THREADS=16 python sv-cpu-omp.py
OMP_NUM_THREADS=32 python sv-cpu-omp.py
OMP_NUM_THREADS=64 python sv-cpu-omp.py
