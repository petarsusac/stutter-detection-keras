#!/bin/bash
#PBS -N apptainer-test
#PBS -o out/out.txt
#PBS -e out/err.txt
cd $PBS_O_WORKDIR
apptainer run --nv container.sif