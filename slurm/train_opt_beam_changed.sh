#!/bin/bash
#SBATCH --job-name=optuna
#SBATCH --output=source/logs/optuna_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=112G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
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

srun python training_opt_beam_changed.py
