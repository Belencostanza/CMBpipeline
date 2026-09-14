"""
Covariance matrices for the cobaya likelihood  (was: constraint_r_lensing.ipynb, cells "covarianza")

Sample covariance over the nsims simulations of a dedicated run (r = 0.025 + sigma
in the paper) of

  - the OQE estimates:  clbb_bin_fid + F^-1 (El_true_i - bias)   (utilities.correction_modified)
  - the NaMaster estimates: cl_decoupled_sims of compute_namaster.py

Inputs (config.dict, section "covariance", files in spectrum_folder):
  oqe_El             npz of make_true_maps.py   (arr_1 = El BB, shape (nsims, nbins))
  oqe_fisher         [k1, k0] npy of reduce_fisher.py (averaged)
  oqe_bias           npy of reduce_bias.py
  master_estimation  npz of compute_namaster.py (arr_0 = per-sim estimates)
  nsims

Output: cfg["cobaya_paths"]["datafile"] (aux_folder / cobaya["datafile"]) with the
datasets "covariance/oqe" and "covariance/master", as read by run_cobaya_bb.py.

Run:  python make_covariance.py
"""

import numpy as np
import h5py

import utilities
from spectra import fiducial_spectrum
from config_loader import load_config, bin_edges

cfg = load_config()

cv    = cfg["covariance"]
paths = cfg["covariance_paths"]
nsims = cv["nsims"]
out_file = cfg["cobaya_paths"]["datafile"]

# ----------------------------------------------------------------------------- binned fiducial BB
bin_frac = bin_edges(cfg)
nbins = len(bin_frac) - 1

_, _, clfid_bb = fiducial_spectrum(cfg)
ele = np.arange(len(clfid_bb))
_, clbb_bin_fid, _ = utilities.bineado(ele, clfid_bb, bin_frac)

# ----------------------------------------------------------------------------- OQE estimates per sim
El_true_bb = np.load(paths["oqe_El"])["arr_1"]
fisher = np.mean([np.load(f) for f in paths["oqe_fisher"]], axis=0)
bias = np.load(paths["oqe_bias"])

_, _, correction = utilities.correction_modified(El_true_bb, bias, fisher, clbb_bin_fid,
                                                 nsamples=El_true_bb.shape[0], nbins=nbins)
est_oqe = clbb_bin_fid + correction                    # (nsims_available, nbins)

# ----------------------------------------------------------------------------- NaMaster estimates per sim
est_master = np.load(paths["master_estimation"])["arr_0"]   # (nsims_available, nbins + 1 extended bin)

# ----------------------------------------------------------------------------- covariances
Cov_oqe = np.cov(est_oqe[:nsims, :], rowvar=False)
Cov_pcl = np.cov(est_master[:nsims, :nbins], rowvar=False)

print("OQE estimates    :", est_oqe.shape, "-> Cov", Cov_oqe.shape)
print("NaMaster estimates:", est_master.shape, "-> Cov", Cov_pcl.shape)

with h5py.File(out_file, "w") as f:
    cov_grp = f.create_group("covariance")
    cov_grp.create_dataset("oqe", data=Cov_oqe)
    cov_grp.create_dataset("master", data=Cov_pcl)
    cov_grp.attrs["nsims"] = nsims
    cov_grp.attrs["oqe_El"] = paths["oqe_El"]
    cov_grp.attrs["master_estimation"] = paths["master_estimation"]

print("saved", out_file)
