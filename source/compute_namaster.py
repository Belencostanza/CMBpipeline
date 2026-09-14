"""
Pseudo-Cl (NaMaster) BB estimation on the sphere for the "true" simulations.

All input comes from config.dict, section "namaster":
  spectrum_model / r_new / Alens_new / As_new / cosmo_pert -> spectra.true_spectrum
  name_data       (data_folder)      Q/U observations on the sphere (nsims, 2, npix)
  name_coupling   (aux_folder)       NaMaster workspace (read if read_matrix, written otherwise)
  name_noise_bias (aux_folder)       noise bias <N_l> computed by make_noise_bias.py
  name_result     (spectrum_folder)  npz with (cl_decoupled_sims, cl_mean_sims, cl_std_sims)
  nsims / generate_sims / read_matrix / purify_b / scale (apodization, deg)
"""
import pymaster as nmt
import numpy as np

import healpy as hp

import utilities
from geometry import sky_region_from_cfg, make_dataset_from_cfg
from spectra import fiducial_spectrum, true_spectrum

from config_loader import load_config, bin_edges
cfg = load_config()

DEFAULT_NOISE_LEVEL   = cfg["DEFAULT_NOISE_LEVEL"]
fwhm                  = cfg["fwhm_arcmin"]
mask_type             = cfg["mask_type"]
noise_type            = cfg["noise_type"]
inho_type             = cfg["inho_type"]
nside                 = cfg["nside"]

nm              = cfg["namaster"]
paths           = cfg["namaster_paths"]
name_result     = paths["result"]
name_data       = paths["data"]
name_coupling   = paths["coupling"]
name_noise_bias = paths["noise_bias"]
make_data       = nm["generate_sims"]
nsims           = nm["nsims"]
read_matrix     = nm["read_matrix"]
purify_b        = nm.get("purify_b", True)
scale           = cfg["apo_scale"]          # namaster["scale"] resolved for the active mask_type

lmax_sph = 3*nside - 1                       # NaMaster lmax (1535 for nside 512)

def compute_master(f_a, f_b, wsp):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled, cl_noise=cl_noise)
    return cl_decoupled


def compute_master_bias(f_a, f_b, wsp, cl_noise_sim):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled, cl_noise=cl_noise_sim)
    return cl_decoupled#, cl_coupled

################## SPECTRUM ###############################################

cltt, clee, clbb = fiducial_spectrum(cfg)
_, cltrue_ee, cltrue_bb = true_spectrum(cfg, nm)

ele = np.arange(len(cltrue_bb))
fwhm_rad = np.radians(fwhm/60)
beam = utilities.bl(fwhm_rad, len(ele)-1)

################### MASK ######################################################

sky_region = sky_region_from_cfg(cfg, move=True)

if mask_type == "circ":
    mask, valid_index, vecs, vec_center, theta_idx, phi_idx, lonc, latc = sky_region.make_mask()
    nhits = None
elif mask_type == "rect":
    mask, nhits, valid_index, vecs, vec_center, theta_idx, phi_idx, lonc, latc = sky_region.make_SO_mask()

mask_apod = nmt.mask_apodization(mask, aposize=scale, apotype='Smooth')


################## DATA #########################################################


if make_data:
    print('making simulations')
    obs_sph = np.zeros((nsims, 2, len(mask)))

    make_data = make_dataset_from_cfg(cfg)
    cls = make_data._create_spectrum_array(cltt, cltrue_ee, cltrue_bb)

    # variance map on the sphere: hits model (circ, or elliptical if inho_type == "elliptical")
    # or SO hits (rect); same call as make_dataset.make_train_dataset
    theta0, phi0 = hp.vec2ang(vec_center)
    Nhits_masked, variance_map = make_data.get_inhomogenous_noise(
        mask_type, mask, inho_type=inho_type, nhits=nhits,
        vec_center=vec_center, theta0=theta0, lonc=lonc, latc=latc)

    for i in range(nsims):

        mapT, mapQ, mapU = make_data.get_QUmaps_cl(cls, beam=True)
        noiseQ = make_data.get_inho_noise(variance_map, mask)
        noiseU = make_data.get_inho_noise(variance_map, mask)
        dataQ = mask*(mapQ + noiseQ)
        dataU = mask*(mapU + noiseU)
        obs_sph[i,0,:] = dataQ
        obs_sph[i,1,:] = dataU

    np.save(name_data, obs_sph)

else:

    print('loading simulations')
    obs_sph = np.load(name_data)

############### BINNING ##########################################################

bin_lin = bin_edges(cfg)

bin_extend = np.zeros(len(bin_lin)+1, dtype=int)
bin_extend[:-1] = np.copy(bin_lin)
bin_extend[-1] = lmax_sph + 1
nbins = len(bin_extend) - 1

ell_in = bin_extend[:-1]
ell_fin = bin_extend[1:]

b = nmt.NmtBin.from_edges(ell_in, ell_fin)

# COUPLING MATRIX
if read_matrix == True:
    print('reading coupling matrix', name_coupling)
    workspace = nmt.NmtWorkspace()
    workspace.read_from(name_coupling)
else:
    print('computing coupling matrix')
    qobs = obs_sph[0,0,:]
    uobs = obs_sph[0,1,:]
    f2 = nmt.NmtField(mask_apod, [mask_apod*qobs, mask_apod*uobs], beam=beam[:lmax_sph+1], purify_b=purify_b, masked_on_input=True)
    workspace = nmt.NmtWorkspace()
    workspace.compute_coupling_matrix(f2, f2, b)
    workspace.write_to(name_coupling)
    print('saved coupling matrix', name_coupling)

# BIAS (make_noise_bias.py)
cl_noise_sim = np.load(name_noise_bias)

print('computing..')

Qobs_sims = obs_sph[:,0,:]
Uobs_sims = obs_sph[:,1,:]

cl_decoupled_sims = np.zeros((nsims,nbins))
for i in range(nsims):
    Qobs_apod = mask_apod*Qobs_sims[i]
    Uobs_apod = mask_apod*Uobs_sims[i]
    f2_sim = nmt.NmtField(mask_apod, [Qobs_apod, Uobs_apod], beam=beam[:lmax_sph+1], purify_b=purify_b, masked_on_input=True)
    cl_decoupled_sims[i] = compute_master_bias(f2_sim, f2_sim, workspace, cl_noise_sim)[3]

cl_mean_sims = np.mean(cl_decoupled_sims, axis=0)
cl_std_sims = np.std(cl_decoupled_sims, axis=0)

np.savez(name_result, cl_decoupled_sims, cl_mean_sims, cl_std_sims)
print("saved", name_result + ".npz")
