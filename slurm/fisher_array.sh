#!/bin/bash
#SBATCH --job-name=fisher
#SBATCH --output=source/logs/fisher_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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

# One Fisher realization per array task. Needs (see submit_all.sh):
#   SLURM_ARRAY_TASK_ID  local index, K_INDEX  0/1 (perturbation -0.5/+0.5),
#   OFFSET               added to the array index -> realization j
python compute_fisher.py
