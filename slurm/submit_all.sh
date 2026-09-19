#!/bin/bash
# Fisher matrix: 2 perturbations (K_INDEX 0/1) x 2000 realizations.
# Split in two arrays because of the usual 1001-task limit per array.
# Run from the repository root:  bash slurm/submit_all.sh
# The number of realizations must match config.dict -> oqe.nsamples_fisher.
mkdir -p source/logs
for K in 0 1
do
  sbatch --export=ALL,K_INDEX=$K,OFFSET=0    --array=0-1000 slurm/fisher_array.sh
  sbatch --export=ALL,K_INDEX=$K,OFFSET=1001 --array=0-998  slurm/fisher_array.sh
done
