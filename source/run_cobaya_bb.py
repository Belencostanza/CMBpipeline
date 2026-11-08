#!/usr/bin/env python

"""
Run Cobaya inference for one individual realization.

Examples
--------
OQE:
    python run_cobaya_bb.py --estimator oqe --realization 0

NaMaster:
    python run_cobaya_bb.py --estimator master --realization 0

MPI:
    mpirun -n 4 python run_cobaya_bb.py --estimator oqe --realization 0

Each realization gets an independent Cobaya output.
"""

"""
MCMC con Cobaya para (r, Alens) a partir del espectro BB binneado.

Misma estructura que constraint_r_lensing.ipynb: el espectro teorico es una
combinacion lineal de dos templates fiduciales fijos,

    Cl_th = r * clBB_prim + Alens * clBB_lens

donde clBB_prim (tensores con r=1, nt = -r_ref/8) y clBB_lens (escalares
lensados, r=0, Alens=1) se calculan con CAMB UNA sola vez al inicio.
Durante el MCMC no se llama mas a CAMB: cada paso es solo la combinacion
lineal + binneado, asi que la evaluacion es practicamente instantanea.
Los unicos parametros libres son r y Alens.

El dato para el chi2 (Cl_dato) es la estimacion del espectro BB sobre
simulaciones generadas con (r, Alens) "true" distintos de los fiduciales
(ej: r=0.028, Alens=1.025):

  - "oqe":    El_true de make_true_maps.py + fisher + bias, combinados con
              utilities.correction_modified (igual que el notebook).
  - "master": estimacion de compute_namaster.py (arr_0) para esa realizacion.

Cada corrida ajusta UNA realizacion (--realization i) con el estimador elegido
(--estimator oqe|master); las cadenas van a <dir de cobaya.output>/bb_{est}_sim{i:02d}.

La covarianza (master u oqe) se lee del .h5 generado por make_covariance.py.

Agregado para correr en cluster: output a disco con resume — si el job se
corta, se relanza el mismo script y las cadenas continuan desde donde
quedaron.

El resto del input se define en config.dict, seccion "cobaya". Los archivos se
resuelven en config_loader: datafile -> aux_folder, estimaciones/fisher/bias
-> spectrum_folder, output -> chains_folder.

Correr local:             python run_cobaya_bb.py --estimator oqe --realization 0
Correr en cluster (MPI):  mpirun -n 4 python run_cobaya_bb.py --estimator oqe --realization 0
                          (cada proceso corre una cadena; requiere mpi4py)
"""

import os
import argparse

import numpy as np
import h5py

import utilities
from config_loader import load_config, bin_edges
from spectra import fiducial_spectrum

from cobaya.run import run


# ============================================================================
# COMMAND LINE
# ============================================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--estimator",
    type=str,
    choices=["oqe", "master"],
    required=True,
    help="Estimator used for the inference."
)

parser.add_argument(
    "--realization",
    type=int,
    required=True,
    help="Realization index: 0, ..., nsims-1."
)

args = parser.parse_args()

estimator = args.estimator
ireal = args.realization

if ireal < 0:
    raise ValueError("realization must be >= 0")

# ================= CONFIG =================================================

cfg = load_config()
cob = cfg["cobaya"]
paths = cfg["cobaya_paths"]

datafile    = paths["datafile"]
output      = paths["output"]
lmax_theory = cob["lmax_theory"]

r_fid = cfg["r"]                    # r fiducial (r_ref del template primordial)
fid   = cfg["cosmo_fid"]

# ================= BINNEADO ================================================

bin_frac = bin_edges(cfg)
nbins = len(bin_frac) - 1   # 18, igual que en el notebook
nbins_use = nbins - 1
ell = np.arange(lmax_theory + 1)

# ================= COVARIANZA (del .h5 de make_covariance.py) =============

with h5py.File(datafile, "r") as f:
    Cov_oqe = f["covariance/oqe"][:]
    Cov_pcl = f["covariance/master"][:]

inv_Cov = {"master": np.linalg.inv(Cov_pcl[:nbins_use, :nbins_use]),
           "oqe":    np.linalg.inv(Cov_oqe[:nbins_use, :nbins_use])}[estimator]


# ================= TEMPLATES FIDUCIALES (una sola llamada a CAMB) ==========

_, _, clBB_prim, clBB_lens = utilities.get_templates_fiducial(
    H0=fid["H0"], ombh2=fid["ombh2"], omch2=fid["omch2"], tau=fid["tau"],
    ns=fid["ns"], As=fid["As"], mnu=fid["mnu"], r_ref=r_fid, lmax=lmax_theory)


def Clbb_theory_binned(r, Alens):
    clbb_th = utilities.clbb_theory(r, Alens, clBB_prim, clBB_lens)
    _, clbb_bin_th, _ = utilities.bineado(ell, clbb_th, bin_frac)
    return clbb_bin_th


# ================= DATO (estimacion OQE / NaMaster) ========================
# Cl_dato: estimacion sobre las sims generadas con (r_new, Alens_new) por
# make_true_maps.py (OQE) o compute_namaster.py (NaMaster), igual que en el
# notebook.

# ============================================================================
# DATA
# ============================================================================

print()
print("=" * 70)
print(f"Estimator   : {estimator}")
print(f"Realization : {ireal}")
print("=" * 70)
print()

if estimator == "master":

    est_master = np.load(paths["master_estimation"])

    # arr_0 contains the individual realizations
    master_all = est_master["arr_0"]

    print("NaMaster estimations shape:", master_all.shape)

    if ireal >= master_all.shape[0]:
        raise ValueError(
            f"Requested realization {ireal}, "
            f"but only {master_all.shape[0]} realizations exist."
        )

    # Select ONE realization
    Cl_dato = master_all[ireal, :nbins]

elif estimator == "oqe":

    # npz de make_true_maps.py: arr_0 = EE, arr_1 = BB
    El_true_bb_all = np.load(paths["oqe_El"])["arr_1"]

    print("OQE El shape:", El_true_bb_all.shape)

    if ireal >= El_true_bb_all.shape[0]:
        raise ValueError(
            f"Requested realization {ireal}, "
            f"but only {El_true_bb_all.shape[0]} realizations exist."
        )

    # fisher: promedio de las perturbaciones +0.5 / -0.5 (k1 y k0)
    fisher = np.mean([np.load(f) for f in paths["oqe_fisher"]], axis=0)

    bias = np.load(paths["oqe_bias"])

    if cfg["fid_iter"] is None:
        # expansion alrededor del fiducial (r_fid, Alens=1), como el notebook
        clbb_bin_fid = Clbb_theory_binned(r_fid, 1.0)
    else:
        # expansion alrededor del fiducial iterado (CAMB con r, Alens de fid_iter),
        # el mismo usado para el fisher, el bias y El
        _, _, clfid_bb = fiducial_spectrum(cfg)
        _, clbb_bin_fid, _ = utilities.bineado(np.arange(len(clfid_bb)), clfid_bb, bin_frac)
        print("fiducial iterado r, Alens:", cfg["fid_iter"]["r"], cfg["fid_iter"]["Alens"])

    # ------------------------------------------------------------------------
    # Select ONE realization
    #
    # Keep first dimension:
    #
    # (nsims, nbins) -> (1, nbins)
    # ------------------------------------------------------------------------

    El_true_bb = El_true_bb_all[ireal:ireal + 1]

    Cl_dato, _, _ = utilities.correction_modified(El_true_bb, bias, fisher,
                                                  clbb_bin_fid,
                                                  nsamples=1,
                                                  nbins=nbins)
    print("Cl_dato shape:", Cl_dato.shape)

else:
    raise ValueError(f"cobaya.estimator must be 'master' or 'oqe', got '{estimator}'")

# ============================================================================
# OUTPUT
# ============================================================================

# Use the directory defined in config, but make an independent chain
# for every estimator and realization.

output_base = paths["output"]

output_dir = os.path.dirname(output_base)

if not output_dir:
    output_dir = "."


output = os.path.join(
    output_dir,
    f"bb_{estimator}_sim{ireal:02d}"
)

print("Cobaya output:", output)
print()

# ================= LIKELIHOOD ==============================================
# Cobaya lee (r, Alens) de la firma de la funcion; no hay theory block.


def MyLikelihood(r, Alens):

    cl_theory_binned = Clbb_theory_binned(r, Alens)

    delta = Cl_dato[:nbins_use] - cl_theory_binned[:nbins_use]

    chi2 = delta @ inv_Cov @ delta

    return -0.5 * chi2


# ================= INFO ====================================================

info = {

    "params": {
        "r": {
            "prior": {"min": 0.0, "max": 0.1},
            "ref": 0.03,
            "proposal": 0.005,
            "latex": "r",
        },
        "Alens": {
            "prior": {"min": 0.5, "max": 1.5},
            "ref": 1.0,
            "proposal": 0.01,
            "latex": "A_\\mathrm{lens}",
        },
    },

    "likelihood": {
        "MyLikelihood": {
            "external": MyLikelihood,
        },
    },

    "sampler": {
        "mcmc": {
            "Rminus1_stop": cob["rminus1"],   # criterio de convergencia entre cadenas
            "learn_proposal": True,
            "max_tries": 10000,
        },
    },

    # cadenas a disco + resume si ya existen
    "output": output,
    "resume": cob["resume"],
}

if cob.get("max_samples") is not None:
    info["sampler"]["mcmc"]["max_samples"] = cob["max_samples"]

# ================= RUN =====================================================

os.makedirs(output_dir, exist_ok=True)


print("=" * 70)
print("Starting Cobaya")
print("=" * 70)
print()

updated_info, sampler = run(info)
