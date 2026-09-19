#!/bin/bash
#SBATCH --job-name=true_maps
#SBATCH --output=source/logs/true_maps_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=112G
#SBATCH --time=04:00:00
##SBATCH --partition=...           # set according to your cluster
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=you@example.org

# ---- environment: edit to match your installation ---------------------------
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch_cmb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Submit from the repository root (sbatch slurm/<job>.sh); the scripts read
# source/config.dict (or the file given in $WF_CONFIG).
cd "${SLURM_SUBMIT_DIR}/source"

srun python make_true_maps.py
