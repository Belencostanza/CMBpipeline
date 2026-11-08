#!/bin/bash
#SBATCH --job-name=bb_mcmc
#SBATCH --output=source/logs/bb_mcmc_%j.out
#SBATCH --ntasks=4              # 4 MCMC chains (MPI)
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
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

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
# run_cobaya_bb.py fits ONE realization per call (--estimator, --realization);
# output prefix and convergence are set in config.dict -> "cobaya".
# Choose estimator and realizations at submission, e.g.
#   sbatch --export=ALL,ESTIMATOR=master,FIRST=0,LAST=199 slurm/job_cobaya_bb.sh
# If the job hits the time limit, resubmit: with "resume": True the chains continue.
ESTIMATOR=${ESTIMATOR:-oqe}
FIRST=${FIRST:-0}
LAST=${LAST:-0}
for i in $(seq ${FIRST} ${LAST}); do
    echo "${ESTIMATOR} realization ${i}"
    srun python run_cobaya_bb.py --estimator ${ESTIMATOR} --realization ${i} || exit 1
done
