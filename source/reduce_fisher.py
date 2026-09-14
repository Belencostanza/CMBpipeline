# reduce_fisher.py
# Average the per-realization fisher matrices written by compute_fisher.py
# (fisher_2k_{ee,bb}_k{k}_j{j}_{tag}.npy) into fisher_mean{N}_{ee,bb}_{label}_k{k}_{tag}.npy

import numpy as np

from config_loader import load_config
cfg = load_config()

tag             = cfg["spectrum_tag"]
spectrum_folder = cfg["spectrum_folder"]
nsamples_fisher = cfg["oqe"]["nsamples_fisher"]   # 2000 -> fisher_mean2000_...
label           = cfg["binning"]["label"]         # "newbin5"

k = [-0.5, 0.5]
Nk = len(k)
Ns = nsamples_fisher

for k_index in range(Nk):

    fisher_sum_ee = None
    fisher_sum_bb = None

    for j in range(Ns):

        F_ee = np.load(spectrum_folder + f"fisher_2k_ee_k{k_index}_j{j}_{tag}.npy")
        F_bb = np.load(spectrum_folder + f"fisher_2k_bb_k{k_index}_j{j}_{tag}.npy")

        if fisher_sum_ee is None:
            fisher_sum_ee = np.zeros_like(F_ee)
            fisher_sum_bb = np.zeros_like(F_bb)

        fisher_sum_ee += F_ee
        fisher_sum_bb += F_bb

    fisher_mean_ee = fisher_sum_ee / Ns
    fisher_mean_bb = fisher_sum_bb / Ns

    np.save(spectrum_folder + f"fisher_mean{Ns}_ee_{label}_k{k_index}_{tag}.npy", fisher_mean_ee)
    np.save(spectrum_folder + f"fisher_mean{Ns}_bb_{label}_k{k_index}_{tag}.npy", fisher_mean_bb)


print("Means computed.")
