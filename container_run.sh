#!/bin/bash
#PBS -N stutter-detection-keras
#PBS -o out/out.txt
#PBS -e out/err.txt
#PBS -l select=1:ncpus=1:ngpus=1
cd $PBS_O_WORKDIR
apptainer run --nv container.sif