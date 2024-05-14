#!/bin/bash
#PBS -N stutter-detection-keras
#PBS -o out/out.txt
#PBS -e out/err.txt
#PBS -l select=1:ncpus=4:ngpus=1
cd $PBS_O_WORKDIR
apptainer run --nv container_tf_2_11.sif