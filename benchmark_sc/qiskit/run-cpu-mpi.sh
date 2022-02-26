#!/bin/bash
source /opt/spack/share/spack/setup-env.sh
spack load openblas@0.3.18
spack load openmpi@4.1.1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qiskit-cpu-mpi

set -x

`which mpirun` -n 1 -host nico1:1 -x PATH=$PATH -x OMP_NUM_THREADS=1 python sv-cpu-mpi.py
`which mpirun` -n 2 -host nico1:2 -x PATH=$PATH -x OMP_NUM_THREADS=1 python sv-cpu-mpi.py
`which mpirun` -n 4 -host nico1:4 -x PATH=$PATH -x OMP_NUM_THREADS=1 python sv-cpu-mpi.py
`which mpirun` -n 8 -host nico1:8 -x PATH=$PATH -x OMP_NUM_THREADS=1 python sv-cpu-mpi.py
`which mpirun` -n 16 -host nico1:16 -x PATH=$PATH -x OMP_NUM_THREADS=1 python sv-cpu-mpi.py
`which mpirun` -n 32 -host nico1:32 -x PATH=$PATH -x OMP_NUM_THREADS=1 python sv-cpu-mpi.py
`which mpirun` -n 64 -host nico1:64 -x PATH=$PATH -x OMP_NUM_THREADS=1 python sv-cpu-mpi.py
`which mpirun` -n 64 -host nico1:32,nico2:32 -x PATH=$PATH -x OMP_NUM_THREADS=1 python sv-cpu-mpi.py
`which mpirun` -n 128 -host nico1:64,nico2:64 -x PATH=$PATH -x OMP_NUM_THREADS=1 python sv-cpu-mpi.py
