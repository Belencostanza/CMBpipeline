#author: Belén Costanza

import numpy as np
import sys
import os
import scipy
from scipy import interpolate
import time
import h5py
import torch

import healpy as hp
import utilities

import PowerSpectrum as PP
from geometry import build_geometry

####################### read-config #####################

from config_loader import load_config, bin_edges
cfg = load_config()

tag             = cfg["spectrum_tag"]
spectrum_folder = cfg["spectrum_folder"]
mask_type       = cfg["mask_type"]
noise_type      = cfg["noise_type"]
nx              = cfg["npixels_x"]
ny              = cfg["npixels_y"]
nside           = cfg["nside"]
fwhm            = cfg["fwhm_arcmin"]
DEFAULT_NOISE_LEVEL = cfg["DEFAULT_NOISE_LEVEL"]
factor          = cfg["map_rescale_factor"]
model_folder    = cfg["model_path"]
study_name      = cfg["study_name"]

# precomputed noise realizations + psi on the sphere (config: fisher_modes_file -> data_folder)
fisher_modes    = cfg["fisher_modes_path"]

_, clee_ang, clbb_ang = utilities.signal_spectrum(r=cfg["r"])
ele = np.arange(len(clee_ang))
bin_edges = bin_edges(cfg)
print(bin_edges)
nbins = len(bin_edges) - 1


######################### sky region / projection (geometry.py) ################################

geo = build_geometry(cfg, load_inho2d=(noise_type == "inho"), variance=False)
mask, valid_index, proj, z_eq = geo.mask, geo.valid_index, geo.proj, geo.z_eq
dx, dy, mask2d, inho2d = geo.dx, geo.dy, geo.mask2d, geo.inho2d

ps = PP.PowerSpectrum(nside, nx, ny, dx, dy, fwhm, nbins, bin_edges, factor, DEFAULT_NOISE_LEVEL, mask_type, noise_type)

ell_flat, clee_flat = ps.flat_spectrum_xy(clee_ang) #fiducial flat power spectrum
_, clbb_flat = ps.flat_spectrum_xy(clbb_ang)

el_bin_ee, clee_bin, _ = utilities.bineado(ele[1:], clee_ang[1:], bin_edges)  #binned flat power spectrum
el_bin_bb, clbb_bin, _ = utilities.bineado(ele[1:], clbb_ang[1:], bin_edges)

print(el_bin_ee)
print(clee_bin)
#load models once
print('loading model')
model = ps.get_models(model_folder, study_name, storage=cfg["study_db"])


################################# calculate fisher matrix per worker###################################


print('start fisher calculation')

k = np.array([-0.5, 0.5])

j_local = int(os.environ["SLURM_ARRAY_TASK_ID"])
k_index = int(os.environ["K_INDEX"])
offset = int(os.environ["OFFSET"])
j_index = j_local + offset

#seed = 100000 * k_index + j_index
seed = j_index
np.random.seed(seed)

# load only noiseQ and psi
with h5py.File(fisher_modes, "r") as f:
    noiseQ = f["noiseQ"][:]
    noiseU = f["noiseU"][:]
    #inho2d = f["inho2d"][:]
    psi = f["psi"][:]


almE_fid = hp.synalm(clee_ang, lmax=1600)
almB_fid = hp.synalm(clbb_ang, lmax=1600)

t0 = time.time()
if noise_type == "ho":
    data_fisher = ps.generate_one_pln(z_eq, almE_fid, almB_fid, mask, noiseQ, noiseU, psi, valid_index, proj, mask2d)
elif noise_type == "inho":
    data_fisher = ps.generate_one_inho_pln(z_eq, almE_fid, almB_fid, mask, noiseQ, noiseU, psi, valid_index, proj, inho2d, mask2d)

t1 = time.time()
print('Time to generate one realization:', t1-t0)
Elee_fiducial, Elbb_fiducial = ps.fiducial_one(data_fisher, model, ell_flat, clee_flat, clbb_flat, clee_bin, clbb_bin)

cte = k[k_index]

print(f"Running k index {k_index}, realization {j_index}")

t4 = time.time()

# perturbation only depends on k
cl_pert_prop_e, c = ps.perturbation_angular(ele, clee_ang, k=cte)
cl_pert_prop_b, _ = ps.perturbation_angular(ele, clbb_ang, k=cte)


fisher_ee, fisher_bb = ps.fisher_parallel(ele,
    z_eq, mask, model,
    almE_fid,
    almB_fid,
    Elee_fiducial,
    Elbb_fiducial,
    ell_flat, clee_ang, clbb_ang,
    clee_flat, clbb_flat,
    clee_bin, clbb_bin,
    cl_pert_prop_e, cl_pert_prop_b,
    cte, proj, noiseQ, noiseU,
    valid_index, psi, mask2d, inho2d = inho2d
)

t5 = time.time()
print("Time for one fisher realization:", t5 - t4)

# Guardar resultado individual
np.save(spectrum_folder + f"fisher_2k_ee_k{k_index}_j{j_index}_{tag}.npy", fisher_ee)
np.save(spectrum_folder + f"fisher_2k_bb_k{k_index}_j{j_index}_{tag}.npy", fisher_bb)
