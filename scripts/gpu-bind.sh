#!/bin/bash
if [ $OMPI_COMM_WORLD_LOCAL_RANK ]; then
    rank=$OMPI_COMM_WORLD_LOCAL_RANK
elif [ $SLURM_LOCALID ]; then
    rank=$SLURM_LOCALID
else
    echo "ERROR: Unknown rank"
    exit 1
fi
GPU_start=$(( $rank * $GPUPerRank ))
GPU_end=$(( ($rank + 1) * $GPUPerRank - 1 ))
GPU=`echo $(for i in $(seq $GPU_start $GPU_end); do printf "$i,"; done)`
CUDA_VISIBLE_DEVICES=$GPU $@
