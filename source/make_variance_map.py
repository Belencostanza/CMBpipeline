"""
2D variance map of the projected inhomogeneous noise  (was: correlated_noise_inhomogenous.ipynb)

For the active mask_type it
  1. builds the variance map on the sphere from the hits model (circ; elliptical if
     inho_type == "elliptical") or from the SO hits map (rect),
  2. draws nsims_inho2d noise realizations, masks them and projects them to the
     plane with the same projection used for the training data,
  3. saves the pixel variance over the realizations, shape (ny, nx), to

        cfg["inho2d_path"]  =  aux_folder / inho2d_file[mask_type]

This is the 4th channel of the network input (sigma2_pln in make_dataset.make_train_dataset)
and the `inho2d` array used by compute_fisher / compute_bias / make_true_maps.

Run:  python make_variance_map.py
"""

import numpy as np

import utilities
from geometry import build_geometry
from config_loader import load_config

cfg = load_config()

mask_type = cfg["mask_type"]
nsims     = cfg["nsims_inho2d"]
out_file  = cfg["inho2d_path"]

geo = build_geometry(cfg, load_inho2d=False, variance=True)
mask, valid_index, proj, z_eq, mask2d = geo.mask, geo.valid_index, geo.proj, geo.z_eq, geo.mask2d
variance_map = geo.variance_map

ny, nx = mask2d.shape
print(f"mask_type={mask_type}, plane grid (ny, nx)=({ny}, {nx}), {nsims} noise realizations")

noise_pln = np.zeros((nsims, ny, nx))

for i in range(nsims):

    noise_map = utilities.sigma2_to_map(variance_map, mask)
    noise_piece = (mask * noise_map)[valid_index]

    if mask_type == "rect":
        noise_pln[i] = proj.grid_bins_mask_from_res_margin(z_eq, noise_piece, mask2d)
    elif mask_type == "circ":
        noise_pln[i] = proj.grid_bins_mask_margin(z_eq, noise_piece, mask2d)

var_pln = np.var(noise_pln, axis=0)

np.save(out_file, var_pln)
print("saved", out_file, var_pln.shape)
