#!/bin/bash
#SBATCH --job-name=bb_mcmc
#SBATCH --output=logs/bb_mcmc_%j.out
#SBATCH --error=logs/bb_mcmc_%j.err
#SBATCH --ntasks=4              # 4 cadenas MCMC (procesos MPI)
#SBATCH --cpus-per-task=4       # threads OpenMP para CAMB dentro de cada cadena
#SBATCH --time=12:00:00
##SBATCH --partition=...        # completar segun el cluster
##SBATCH --mem-per-cpu=2G

# Entorno: activar el env con cobaya, camb, h5py y mpi4py
# module load openmpi            # o el modulo MPI del cluster
# source activate mi_env

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# el script se corre desde soft/source (ahi estan config.dict y utilities)
#cd soft/source

mkdir -p logs chains

# todo el input (estimador, output, convergencia) se define en config.dict,
# seccion "cobaya"
srun python run_cobaya_bb.py

# Si el job se corta por tiempo, relanzar tal cual: con "resume": True en
# config.dict las cadenas continuan desde donde quedaron.
