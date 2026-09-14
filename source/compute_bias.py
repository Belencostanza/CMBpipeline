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

# precomputed psi on the sphere (config: fisher_modes_file -> data_folder)
fisher_modes    = cfg["fisher_modes_path"]

_, clee_ang, clbb_ang = utilities.signal_spectrum(r=cfg["r"])
ele = np.arange(len(clee_ang))
bin_edges = bin_edges(cfg)
print(bin_edges)
nbins = len(bin_edges) - 1


######################### sky region / projection (geometry.py) ################################

geo = build_geometry(cfg, load_inho2d=(noise_type == "inho"), variance=True)
make_data = geo.make_data
mask, valid_index, proj, z_eq = geo.mask, geo.valid_index, geo.proj, geo.z_eq
dx, dy, mask2d, inho2d, variance_map = geo.dx, geo.dy, geo.mask2d, geo.inho2d, geo.variance_map

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


################################# calculate noise bias term ###################################


j_local = int(os.environ["SLURM_ARRAY_TASK_ID"])
offset = int(os.environ["OFFSET"])
j_index = j_local + offset

seed = 123456 + j_index
np.random.seed(seed)

print('seed:', seed)
# load psi
with h5py.File(fisher_modes, "r") as f:
    psi = f["psi"][:]

almE_fid = hp.synalm(clee_ang, lmax=1600)
almB_fid = hp.synalm(clbb_ang, lmax=1600)

# create differents noise realizations
#noiseQ, noiseU = make_data.get_QUnoise(nl)

noiseQ = make_data.get_inho_noise(variance_map, mask)
noiseU = make_data.get_inho_noise(variance_map, mask)

t0 = time.time()
if noise_type == "ho":
    data_bias = ps.generate_one_pln(z_eq, almE_fid, almB_fid, mask, noiseQ, noiseU, psi, valid_index, proj, mask2d)
elif noise_type == "inho":
    data_bias = ps.generate_one_inho_pln(z_eq, almE_fid, almB_fid, mask, noiseQ, noiseU, psi, valid_index, proj, inho2d, mask2d)

t1 = time.time()
print('Time to generate one realization:', t1-t0)
Elee_fiducial, Elbb_fiducial = ps.fiducial_one(data_bias, model, ell_flat, clee_flat, clbb_flat, clee_bin, clbb_bin)

np.save(spectrum_folder + f"bias2k_ee_j{j_index}_{tag}.npy", Elee_fiducial)
np.save(spectrum_folder + f"bias2k_bb_j{j_index}_{tag}.npy", Elbb_fiducial)
