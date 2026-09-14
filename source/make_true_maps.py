"""
Evaluate the OQE (network + quadratic estimator) on simulations generated with
a "true" power spectrum different from the fiducial one.

All input comes from config.dict, section "true_maps":
  spectrum_model / r_new / Alens_new / As_new / cosmo_pert -> spectra.true_spectrum
  file_sph  (data_folder)     Q/U observations on the sphere
  file_pt   (data_folder)     projected plane data cube (nsims, 3|4, ny, nx)
  result_name (spectrum_folder)  npz with (El_ee_true, El_bb_true)
  make_data_sph / make_data_pln / nsims
"""
import sys, platform, os
import numpy as np

import healpy as hp
import torch
import time

import utilities
import PowerSpectrum as PP
from geometry import build_geometry
from spectra import fiducial_spectrum, true_spectrum

from config_loader import load_config, bin_edges
cfg = load_config()

spectrum_folder = cfg["spectrum_folder"]
model_folder    = cfg["model_path"]
study_name      = cfg["study_name"]
factor          = cfg["map_rescale_factor"]

tm            = cfg["true_maps"]
paths         = cfg["true_maps_paths"]
file_sph      = paths["file_sph"]
file_pt       = paths["file_pt"]
result_path   = paths["result"]
make_data_sph = tm["make_data_sph"]
make_data_pln = tm["make_data_pln"]
nsims         = tm["nsims"]
cmb_seed      = tm["cmb_seed"]
noise_seed    = tm["noise_seed"]

mask_type  = cfg["mask_type"]
noise_type = cfg["noise_type"]
nside      = cfg["nside"]
nx         = cfg["npixels_x"]
ny         = cfg["npixels_y"]
fwhm       = cfg["fwhm_arcmin"]

################## SPECTRA ####################################################

clfid_tt, clfid_ee, clfid_bb = fiducial_spectrum(cfg)
ele = np.arange(len(clfid_ee))

cltrue_tt, cltrue_ee, cltrue_bb = true_spectrum(cfg, tm)

nl = np.zeros((len(clfid_ee)))
nl[:] = cfg["DEFAULT_NOISE_LEVEL"]

################## SKY REGION / PROJECTION (geometry.py) ######################

geo = build_geometry(cfg, load_inho2d=(noise_type == "inho"), variance=True)
make_data = geo.make_data
cls = make_data._create_spectrum_array(cltrue_tt, cltrue_ee, cltrue_bb)

mask, valid_index, proj, z_eq, psi = geo.mask, geo.valid_index, geo.proj, geo.z_eq, geo.psi
dx, dy, mask2d, inho2d, variance_map = geo.dx, geo.dy, geo.mask2d, geo.inho2d, geo.variance_map

################## BINNING / ESTIMATOR ########################################

bin_lin = bin_edges(cfg)
nbins_lin = len(bin_lin)-1

ps_lin = PP.PowerSpectrum(nside, nx, ny, dx, dy, fwhm, nbins_lin, bin_lin, factor, DEFAULT_NOISE_LEVEL=nl[0], mask_type=mask_type, noise_type=noise_type, nsamples_fisher=nsims)

################## DATA #######################################################

if make_data_sph:
    print("Generating simulations...")
    # make dataset with true power spectrum and equidistant projection

    if noise_type == "ho":
        data = torch.zeros((nsims, 3, ny, nx), dtype=torch.float32)
    elif noise_type == "inho":
        data = torch.zeros((nsims, 4, ny, nx), dtype=torch.float32)
    data_sph = np.zeros((nsims, 2, len(mask)))

    t0 = time.time()
    for i in range(nsims):
        if noise_type == "ho":
            gridQ_data, gridU_data, mask2d, dataQ_sph, dataU_sph = make_data.make_maps_pln_cls(cls, nl, proj, valid_index, mask, z_eq, psi, mask_type, mask2d)
            data[i, 0, :, :] = torch.from_numpy(gridQ_data)
            data[i, 1, :, :] = torch.from_numpy(gridU_data)
            data[i, 2, :, :] = torch.from_numpy(mask2d)
            data_sph[i,0,:] = dataQ_sph
            data_sph[i,1,:] = dataU_sph

        elif noise_type == "inho":
            if cmb_seed is not None:
                gridQ_data, gridU_data, mask2d, dataQ_sph, dataU_sph = make_data.make_maps_inho_pln_cls(cls, variance_map, proj, valid_index, mask, z_eq, psi, mask_type, mask2d, cmb_seed, noise_seed)
            else:
                gridQ_data, gridU_data, mask2d, dataQ_sph, dataU_sph = make_data.make_maps_inho_pln_cls(cls, variance_map, proj, valid_index, mask, z_eq, psi, mask_type, mask2d)
            data[i, 0, :, :] = torch.from_numpy(gridQ_data)
            data[i, 1, :, :] = torch.from_numpy(gridU_data)
            data[i, 2, :, :] = torch.from_numpy(mask2d)
            data[i, 3, :, :] = torch.from_numpy(inho2d)
            data_sph[i,0,:] = dataQ_sph
            data_sph[i,1,:] = dataU_sph

    t1 = time.time()
    elapsed = t1 - t0
    print(f"Generated {nsims} simulations, sph and pln")
    print(f"Total time: {elapsed:.2f} s ({elapsed/60:.2f} min)")
    print(f"Time per simulation: {elapsed/nsims:.2f} s")

    np.save(file_sph, data_sph)
    torch.save(data, file_pt)

elif make_data_pln:
    print("Loading existing sph simulations...")

    data_sph = np.load(file_sph)
    mapQ = data_sph[:,0,:]
    mapU = data_sph[:,1,:]

    print("Generating pln simulations...")
    # make dataset with true power spectrum and equidistant projection
    data = torch.zeros((nsims, 4, ny, nx), dtype=torch.float32)

    for i in range(nsims):
        gridQ_data, gridU_data, _  = make_data.make_maps_inho_pln_from_sph(mapQ[i], mapU[i], proj, valid_index, mask, z_eq, psi, mask_type, mask2d)
        data[i, 0, :, :] = torch.from_numpy(gridQ_data)
        data[i, 1, :, :] = torch.from_numpy(gridU_data)
        data[i, 2, :, :] = torch.from_numpy(mask2d)
        data[i, 3, :, :] = torch.from_numpy(inho2d)

    torch.save(data, file_pt)

else:
    print("Loading existing pln simulations...")
    data = torch.load(file_pt)

################## ESTIMATION #################################################

model = ps_lin.get_models(model_folder, study_name, storage=cfg["study_db"])

# flat fiducal power spectrum
ell_flat, clee_flat = ps_lin.flat_spectrum_xy(clfid_ee)
_, clbb_flat = ps_lin.flat_spectrum_xy(clfid_bb)

# bin power spectrum
el, clee_bin, _ = utilities.bineado(ele[1:], clfid_ee[1:], bin_lin)
_, clbb_bin, _ = utilities.bineado(ele[1:], clfid_bb[1:], bin_lin)

# calculate El_true with nsims maps
t2 = time.time()
El_ee_true, El_bb_true = ps_lin.fiducial(data, model, ell_flat, clee_flat, clbb_flat, clee_bin, clbb_bin, all=True)
t3 = time.time()
elapsed = t3 - t2
print(f"Total time El: {elapsed:.2f} s ({elapsed/60:.2f} min)")
print(f"Time per El: {elapsed/nsims:.2f} s")

np.savez(result_path, El_ee_true, El_bb_true)
print("saved", result_path + ".npz")
