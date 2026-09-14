"""
Fiducial and "true" angular power spectra driven by config.dict.

make_true_maps.py and compute_namaster.py used to repeat the same block with the
fiducial cosmology and the Planck 1-sigma shifts typed in by hand; both now call
`true_spectrum(cfg, cfg["true_maps"])` / `true_spectrum(cfg, cfg["namaster"])`.
"""

import numpy as np

import utilities


def fiducial_spectrum(cfg):
    """(cltt, clee, clbb) of the fiducial model (utilities.components_spectrum_cosmo defaults)."""
    return utilities.components_spectrum_cosmo()


def true_spectrum(cfg, section):
    """
    "True" spectra used to generate the evaluation simulations.

    section: cfg["true_maps"] or cfg["namaster"]; keys used:
      spectrum_model : "cosmo" -> full CAMB run with r_new, As_new and the fiducial
                                  cosmology shifted by cfg["cosmo_sigma"] (Planck 1-sigma)
                                  or by a 1% fractional perturbation if cosmo_pert is True
                       "alens" -> Cl_true = r_new*clBB_prim + Alens_new*clBB_lens
                                  (TT and EE stay fiducial)
      r_new, As_new, Alens_new, cosmo_pert

    Returns (cltrue_tt, cltrue_ee, cltrue_bb).
    """
    fid = cfg["cosmo_fid"]
    r_fid = cfg["r"]

    clfid_tt, clfid_ee, clfid_bb = fiducial_spectrum(cfg)

    spectrum_model = section["spectrum_model"]
    r_new = section["r_new"]

    if spectrum_model == "cosmo":
        As_new = section["As_new"]

        if section["cosmo_pert"]:
            frac = 0.01  # 1%
            delta = {k: frac * fid[k] for k in ("H0", "ombh2", "omch2", "tau", "ns", "mnu")}
        else:
            delta = cfg["cosmo_sigma"]

        cltrue_tt, cltrue_ee, cltrue_bb = utilities.components_spectrum_cosmo(
            r_new,
            H0=fid["H0"] + delta["H0"], ombh2=fid["ombh2"] + delta["ombh2"],
            omch2=fid["omch2"] + delta["omch2"], tau=fid["tau"] + delta["tau"],
            ns=fid["ns"] + delta["ns"], As=As_new, mnu=fid["mnu"] + delta["mnu"])

    elif spectrum_model == "alens":
        Alens_new = section["Alens_new"]

        _, _, clBB_prim, clBB_lens = utilities.get_templates_fiducial(
            H0=fid["H0"], ombh2=fid["ombh2"], omch2=fid["omch2"], tau=fid["tau"],
            ns=fid["ns"], As=fid["As"], mnu=fid["mnu"], r_ref=r_fid)
        cltrue_tt = clfid_tt
        cltrue_ee = clfid_ee
        cltrue_bb = utilities.clbb_theory(r_new, Alens_new, clBB_prim, clBB_lens)

    else:
        raise ValueError(f"spectrum_model desconocido: {spectrum_model}")

    return cltrue_tt, cltrue_ee, cltrue_bb
