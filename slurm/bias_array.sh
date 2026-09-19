#!/bin/bash
#SBATCH --job-name=bias
#SBATCH --output=source/logs/bias_%A_%a.out
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

# One noise-bias realization per array task. Needs OFFSET (see submit_all_bias.sh).
python compute_bias.py
