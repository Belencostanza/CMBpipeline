"""
NaMaster noise bias  (was: Namaster_sphere.ipynb, cells "cl_noise_sim")

For the active mask_type and noise configuration it
  1. builds the variance map on the sphere (hits model for circ, SO hits for rect,
     elliptical hits if inho_type == "elliptical"),
  2. apodizes the mask with namaster["scale"][mask_type] ('Smooth'),
  3. draws nsims_noise_bias pairs of Q/U noise maps, masks them and computes the
     coupled pseudo-Cl with NaMaster (no purification, no beam),
  4. saves the mean over realizations, shape (4, 3*nside), to

        cfg["namaster_paths"]["noise_bias"] = aux_folder / namaster["name_noise_bias"]

compute_namaster.py passes this array as `cl_noise` to workspace.decouple_cell.

Run:  python make_noise_bias.py
"""

import numpy as np
import healpy as hp
import pymaster as nmt

from geometry import sky_region_from_cfg, make_dataset_from_cfg
from config_loader import load_config

cfg = load_config()

mask_type = cfg["mask_type"]
inho_type = cfg["inho_type"]
nside     = cfg["nside"]
nm        = cfg["namaster"]
nsims     = nm.get("nsims_noise_bias", 100)
scale     = cfg["apo_scale"]
out_file  = cfg["namaster_paths"]["noise_bias"]

# ----------------------------------------------------------------------------- mask
sky_region = sky_region_from_cfg(cfg, move=True)

if mask_type == "circ":
    mask, valid_index, vecs, vec_center, theta_idx, phi_idx, lonc, latc = sky_region.make_mask()
    nhits = None
elif mask_type == "rect":
    mask, nhits, valid_index, vecs, vec_center, theta_idx, phi_idx, lonc, latc = sky_region.make_SO_mask()

mask_apod = nmt.mask_apodization(mask, aposize=scale, apotype='Smooth')

# ----------------------------------------------------------------------------- noise on the sphere
make_data = make_dataset_from_cfg(cfg)
theta0, phi0 = hp.vec2ang(vec_center)
_, variance_map = make_data.get_inhomogenous_noise(
    mask_type, mask, inho_type=inho_type, nhits=nhits,
    vec_center=vec_center, theta0=theta0, lonc=lonc, latc=latc)

# ----------------------------------------------------------------------------- pseudo-Cl of the noise
lmax_sph = 3*nside
cl_noise_sim = np.zeros((nsims, 4, lmax_sph))

print(f"mask_type={mask_type}, aposize={scale} deg, {nsims} noise realizations")
for i in range(nsims):
    noiseQ = make_data.get_inho_noise(variance_map, mask)
    noiseU = make_data.get_inho_noise(variance_map, mask)
    Qnoise_apod = mask_apod*noiseQ
    Unoise_apod = mask_apod*noiseU
    f2_noise = nmt.NmtField(mask_apod, [Qnoise_apod, Unoise_apod], masked_on_input=True)
    cl_noise_sim[i] = nmt.compute_coupled_cell(f2_noise, f2_noise)

cl_mean_sims = np.mean(cl_noise_sim, axis=0)

np.save(out_file, cl_mean_sims)
print("saved", out_file, cl_mean_sims.shape)
