"""
Shared sky-region / projection set-up.

Every pipeline script (compute_fisher, compute_bias, make_true_maps,
make_variance_map, make_noise_bias, training, ...) used to repeat the same
block: build the mask, project it, compute the plane resolution, the 2D mask,
the rotation angle psi and load the 2D variance map of the noise.  That block
lives here now, driven only by config.dict.

    from config_loader import load_config
    from geometry import build_geometry

    cfg = load_config()
    geo = build_geometry(cfg)
    geo.mask, geo.proj, geo.z_eq, geo.dx, geo.dy, geo.mask2d, geo.psi, geo.inho2d, ...

Nothing is hard-coded: the input files come from cfg["qubic_path"],
cfg["so_hits_path"], cfg["apo_mask_path"], cfg["inho2d_path"].
"""

from types import SimpleNamespace

import numpy as np
import healpy as hp

import projections as pj
import utilities
import make_dataset as dd


def sky_region_from_cfg(cfg, move=True):
    """projections.sph_mask with the mask input files taken from config.dict."""
    return pj.sph_mask(nside=cfg["nside"], radius=cfg["radius_deg"], move=move,
                       qubic_file=cfg["qubic_path"], so_hits_file=cfg["so_hits_path"])


def make_dataset_from_cfg(cfg, nsims_train=1, nsims_valid=1, nsims_test=1):
    """make_dataset.make_dataset built from config.dict (used for noise / map generation)."""
    return dd.make_dataset(nsims_train, nsims_valid, nsims_test, cfg["smooth"], cfg["apo"],
                           nbins=cfg["npixels_x"], nside=cfg["nside"], fwhm=cfg["fwhm_arcmin"],
                           DEFAULT_NOISE_LEVEL=cfg["DEFAULT_NOISE_LEVEL"])


def build_geometry(cfg, load_inho2d=True, variance=True, move=True):
    """
    Build the sky region and its plane projection for the active mask_type.

    Returns a SimpleNamespace with:
      mask, nhits (None for circ), valid_index, vecs, vec_center,
      theta_idx, phi_idx, lonc, latc, theta0, phi0,
      proj, z_eq, z_tan, z_str, dx, dy, mask2d, psi,
      noise_pix_sph, variance_map (sphere, if variance=True),
      inho2d (plane, if load_inho2d=True), make_data
    """
    mask_type = cfg["mask_type"]
    nside     = cfg["nside"]
    nx        = cfg["npixels_x"]
    ny        = cfg["npixels_y"]
    fact      = cfg["fact"]

    make_data = make_dataset_from_cfg(cfg)
    sky_region = sky_region_from_cfg(cfg, move=move)

    g = SimpleNamespace(mask_type=mask_type, make_data=make_data, nhits=None,
                        variance_map=None, inho2d=None)

    if mask_type == "circ":
        (g.mask, g.valid_index, g.vecs, g.vec_center,
         g.theta_idx, g.phi_idx, g.lonc, g.latc) = sky_region.make_mask()
        if cfg["apo"]:
            g.mask = np.load(cfg["apo_mask_path"])

        g.proj = pj.proj_2d(g.lonc, g.latc, g.theta_idx, g.phi_idx, g.vec_center, nbins=nx, fact=fact)
        g.z_eq, g.z_tan, g.z_str = g.proj.proj_conventions(g.vecs, g.vec_center)

        dx1, dy1, _ = utilities.calculate_res(g.z_eq, nx, nx)
        _, _, g.dx  = utilities.calculate_res_margin(g.z_eq, dx1, dy1, nx, nx, fact=fact)
        g.dy = g.dx
        g.mask2d = g.proj.define_region_mask_nbins_margins(g.z_eq, g.mask, nside)

    elif mask_type == "rect":
        (g.mask, g.nhits, g.valid_index, g.vecs, g.vec_center,
         g.theta_idx, g.phi_idx, g.lonc, g.latc) = sky_region.make_SO_mask()

        g.proj = pj.proj_2d(g.lonc, g.latc, g.theta_idx, g.phi_idx, g.vec_center, nbins=nx, fact=fact)
        g.z_eq, g.z_tan, g.z_str = g.proj.proj_conventions(g.vecs, g.vec_center)
        g.mask2d, _, _ = g.proj.define_region_mask_from_res_margins(g.z_eq, g.mask, nside)  # (ny, nx)

        dx1, dy1, _   = utilities.calculate_res(g.z_eq, nx, ny)
        g.dx, g.dy, _ = utilities.calculate_res_margin(g.z_eq, dx1, dy1, nx, ny, fact=fact)

    else:
        raise ValueError(f"mask_type must be 'circ' or 'rect', got '{mask_type}'")

    # rotation of Q/U from the sphere to the plane
    g.theta0, g.phi0 = hp.vec2ang(g.vec_center)
    theta, phi = hp.vec2ang(g.vecs)
    phi_wrapped  = (phi + np.pi) % (2 * np.pi) - np.pi
    phi0_wrapped = (g.phi0 + np.pi) % (2 * np.pi) - np.pi
    g.psi = pj.rotate_geo(theta, phi_wrapped, g.theta0, phi0_wrapped).psi_Q_to_P()

    # inhomogeneous noise on the sphere (variance map from the hits model / SO hits)
    dx_sph = hp.nside2resol(nside)
    g.noise_pix_sph = cfg["DEFAULT_NOISE_LEVEL"] / dx_sph**2
    if variance:
        _, g.variance_map = make_data.get_inhomogenous_noise(
            mask_type, g.mask, inho_type=cfg["inho_type"], nhits=g.nhits,
            vec_center=g.vec_center, theta0=g.theta0, lonc=g.lonc, latc=g.latc)

    # 2D variance map of the projected noise (make_variance_map.py)
    if load_inho2d and cfg["noise_type"] == "inho":
        g.inho2d = np.load(cfg["inho2d_path"])

    return g
