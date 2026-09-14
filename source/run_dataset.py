import sys, platform, os
import numpy as np

import time
import make_dataset as dd
from config_loader import load_config

cfg = load_config()

nsims_train         = cfg["nsims_train"]
nsims_valid         = cfg["nsims_valid"]
nsims_test          = cfg["nsims_test"]
smooth              = cfg["smooth"]
apo                 = cfg["apo"]
nside               = cfg["nside"]
nbins               = cfg["npixels_x"]
fwhm                = cfg["fwhm_arcmin"]
noise_type          = cfg["noise_type"]
mask_type           = cfg["mask_type"]
default_noise_level = cfg["DEFAULT_NOISE_LEVEL"]
inho_type           = cfg["inho_type"]
radius_deg          = cfg["radius_deg"]
filename_train      = cfg["name_train"]
filename_valid      = cfg["name_valid"]

make_data = dd.make_dataset(nsims_train, nsims_valid, nsims_test, smooth, apo,
                            nbins=nbins, nside=nside, fwhm=fwhm,
                            DEFAULT_NOISE_LEVEL=default_noise_level)

t0 = time.time()
make_data.make_train_dataset(radius_deg, filename_train, filename_valid,
                             mask_type=mask_type, noise_type=noise_type, inho_type=inho_type,
                             model=False, method="cubic",
                             inho2d_file=cfg["inho2d_path"],      # 2D variance map (make_variance_map.py)
                             apo_mask_file=cfg["apo_mask_path"],  # apodized mask, only used if apo = True
                             fact=cfg["fact"],
                             qubic_file=cfg["qubic_path"], so_hits_file=cfg["so_hits_path"])
#make_data.make_spectra_dataset(radius_deg, filename_bias, filename_fisher)
t1 = time.time()

print('Time to create dataset:', t1-t0)
