# reduce_bias.py
# Average the per-realization noise-bias terms written by compute_bias.py
# (bias2k_{ee,bb}_j{j}_{tag}.npy) into bias_mean{N}_{ee,bb}_{label}_{tag}.npy

import numpy as np

from config_loader import load_config
cfg = load_config()

tag             = cfg["spectrum_tag"]
spectrum_folder = cfg["spectrum_folder"]
Ns              = cfg["oqe"]["nsamples_bias"]     # 2000 -> bias_mean2000_...
label           = cfg["binning"]["label"]         # "newbin5"

# acumuladores
sum_ee = None
sum_bb = None

sum2_ee = None
sum2_bb = None

for j in range(Ns):

    b_ee = np.load(spectrum_folder + f"bias2k_ee_j{j}_{tag}.npy")
    b_bb = np.load(spectrum_folder + f"bias2k_bb_j{j}_{tag}.npy")

    if sum_ee is None:
        sum_ee = np.zeros_like(b_ee)
        sum_bb = np.zeros_like(b_bb)

        sum2_ee = np.zeros_like(b_ee)
        sum2_bb = np.zeros_like(b_bb)

    sum_ee += b_ee
    sum_bb += b_bb

    sum2_ee += b_ee**2
    sum2_bb += b_bb**2

# media
mean_ee = sum_ee / Ns
mean_bb = sum_bb / Ns

# varianza: E[x^2] - (E[x])^2
var_ee = (sum2_ee / Ns) - mean_ee**2
var_bb = (sum2_bb / Ns) - mean_bb**2

# std
std_ee = np.sqrt(var_ee)
std_bb = np.sqrt(var_bb)

# guardar
np.save(spectrum_folder + f"bias_mean{Ns}_ee_{label}_{tag}.npy", mean_ee)
np.save(spectrum_folder + f"bias_mean{Ns}_bb_{label}_{tag}.npy", mean_bb)

#np.save(spectrum_folder + f"bias_std{Ns}_ee_{label}_{tag}.npy", std_ee)
#np.save(spectrum_folder + f"bias_std{Ns}_bb_{label}_{tag}.npy", std_bb)

print("Mean and std computed.")
