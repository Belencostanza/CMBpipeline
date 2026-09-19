#!/bin/bash
# Noise bias: 2000 realizations, split in two arrays.
# Run from the repository root:  bash slurm/submit_all_bias.sh
# The number of realizations must match config.dict -> oqe.nsamples_bias.
mkdir -p source/logs
sbatch --export=ALL,OFFSET=0    --array=0-1000 slurm/bias_array.sh
sbatch --export=ALL,OFFSET=1001 --array=0-998  slurm/bias_array.sh
