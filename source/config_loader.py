"""
Single entry point for reading config.dict.

    from config_loader import load_config
    cfg = load_config()            # config.dict next to the running script (or $WF_CONFIG)

Besides the raw keys of the file, the returned dict carries *resolved* paths so
that no script has to concatenate folders or hard-code file names:

  folders (absolute, trailing "/", created on demand)
      root_folder, data_folder, aux_folder, study_folder, chains_folder,
      model_folder, loss_folder, result_folder, spectrum_folder
      (+ aliases model_path, loss_path, spectrum_path)

  mask-type dependent keys resolved to a scalar for the active mask_type
      map_rescale_factor, inho2d_file, apo_mask_file, fisher_modes_file,
      namaster["scale"] -> cfg["apo_scale"]

  files
      name_train, name_valid, result_path, study_db,
      inho2d_path, apo_mask_path, fisher_modes_path, qubic_path, so_hits_path,
      true_maps_paths, namaster_paths, covariance_paths, cobaya_paths

  misc
      fwhm_rad, bin_kwargs (arguments of utilities.compute_bins_fractional)
"""

import ast
import math
import os
import re

REQUIRED_KEYS = [
    "mask_type", "nside", "npixels_x", "npixels_y", "radius_deg", "fact",
    "r", "lmin", "lmax",
    "fwhm_arcmin",
    "noise_type", "inho_type", "DEFAULT_NOISE_LEVEL", "noise_pix",
    "map_rescale_factor",
    "epochs", "loss_j3", "batch_size", "smooth", "apo",
    "nsims_train", "nsims_valid", "nsims_test",
    "data_folder", "model_folder", "loss_folder", "result_folder", "spectrum_folder",
    "study_name", "dataset_tag",
    "mask_files",
    "true_maps", "namaster",
]

# Defaults for keys introduced by the path refactor, so older config files still load.
DEFAULTS = {
    "aux_folder":    "./",
    "study_folder":  "./",
    "chains_folder": "./",
    "spectrum_tag":  "",
    "nsims_inho2d":  100,
    "inho2d_file": {
        "circ": "var_map_sims100.npy",
        "rect": "var_map_sims100_SO_inho2_res_arcmin.npy",
    },
    "apo_mask_file": {
        "circ": "guassian_apodization0.8_mask.npy",
        "rect": "guassian_apodization0.8_maskSO.npy",
    },
    "fisher_modes_file": {
        "circ": "nCMB_fisher1_beam_proj2d_eq_n512_mask_rad40_inho.h5",
        "rect": "nCMB_fisher1_beam_proj2d_eq_n512_maskSO_rad40_inho_hitsSO.h5",
    },
    "binning": {"frac": 0.2, "min_width": 20, "extra_edges": None, "label": "newbin5"},
    "cosmo_fid": {"H0": 67.5, "ombh2": 0.022, "omch2": 0.122, "tau": 0.06,
                  "ns": 0.965, "As": 2e-9, "mnu": 0.06},
    "cosmo_sigma": {"H0": 0.42, "ombh2": 0.00015, "omch2": 0.0012, "tau": 0.0,
                    "ns": 0.0042, "mnu": 0.0},
    "oqe": {"nsamples_fisher": 2000, "nsamples_bias": 2000},
}

# Output folders that load_config() creates when create_dirs=True.
OUTPUT_FOLDERS = ["data_folder", "aux_folder", "study_folder", "chains_folder",
                  "model_folder", "loss_folder", "result_folder", "spectrum_folder"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _strip_comments(text):
    # Remove # comments from each line.
    # Assumes # never appears inside string values (safe for this config).
    return '\n'.join(re.sub(r'\s*#.*$', '', line) for line in text.splitlines())


def find_config(path=None):
    """
    Locate config.dict:
      1. explicit `path`
      2. environment variable WF_CONFIG
      3. ./config.dict (current working directory)
      4. config.dict next to this module
    """
    candidates = []
    if path is not None:
        candidates.append(path)
    if os.environ.get("WF_CONFIG"):
        candidates.append(os.environ["WF_CONFIG"])
    candidates.append(os.path.join(os.getcwd(), "config.dict"))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.dict"))

    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"config.dict not found; looked in {candidates}")


def resolve_by_mask(value, mask_type, key="value"):
    """A dict keyed by mask_type -> the entry for the active mask; a scalar passes through."""
    if isinstance(value, dict):
        if mask_type not in value:
            raise KeyError(f"'{key}' has no entry for mask_type='{mask_type}'")
        return value[mask_type]
    return value


def as_folder(root, folder):
    """Absolute folder path with trailing separator; relative paths hang from root."""
    if not os.path.isabs(folder):
        folder = os.path.join(root, folder)
    return os.path.join(os.path.normpath(folder), "")


def as_file(folder, name):
    """Absolute file path: absolute names pass through, otherwise join with folder."""
    if os.path.isabs(name):
        return name
    return os.path.join(folder, name)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def load_config(path=None, create_dirs=True):
    """
    Load and validate config.dict (see module docstring for the derived keys).

    create_dirs: create every output folder (OUTPUT_FOLDERS). Use False when the
    config is only inspected on a machine where those folders do not exist.
    """
    cfg_file = find_config(path)
    with open(cfg_file) as f:
        cfg = ast.literal_eval(_strip_comments(f.read()))

    for key, default in DEFAULTS.items():
        cfg.setdefault(key, default)
    for sub in ("binning", "cosmo_fid", "cosmo_sigma", "oqe"):
        for k, v in DEFAULTS[sub].items():
            cfg[sub].setdefault(k, v)

    # --- Validate required keys ---
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"config.dict is missing required keys: {missing}")

    mask_type = cfg["mask_type"]
    if mask_type not in ("circ", "rect"):
        raise ValueError(f"mask_type must be 'circ' or 'rect', got '{mask_type}'")

    if cfg["noise_type"] not in ("ho", "inho"):
        raise ValueError(f"noise_type must be 'ho' or 'inho', got '{cfg['noise_type']}'")

    for key in ("qubic", "so_hits"):
        if key not in cfg["mask_files"]:
            raise KeyError(f"mask_files is missing '{key}'")

    # --- Mask-type dependent scalars ---
    for key in ("map_rescale_factor", "inho2d_file", "apo_mask_file", "fisher_modes_file"):
        cfg[key] = resolve_by_mask(cfg[key], mask_type, key)
    cfg["apo_scale"] = resolve_by_mask(cfg["namaster"].get("scale", 0.8), mask_type, "namaster.scale")

    # --- Derived scalars ---
    cfg["fwhm_rad"] = math.radians(cfg["fwhm_arcmin"] / 60.0)
    cfg["bin_kwargs"] = dict(lmin=cfg["lmin"], lmax=cfg["lmax"],
                             frac=cfg["binning"]["frac"],
                             min_width=cfg["binning"]["min_width"],
                             extra_edges=cfg["binning"]["extra_edges"])

    # --- Folders ---
    cfg["config_file"] = cfg_file
    root = cfg.get("root_folder") or os.path.dirname(cfg_file)
    root = as_folder(os.path.dirname(cfg_file), root)
    cfg["root_folder"] = root
    for key in OUTPUT_FOLDERS:
        cfg[key] = as_folder(root, cfg[key])
    cfg["model_path"]    = cfg["model_folder"]
    cfg["loss_path"]     = cfg["loss_folder"]
    cfg["spectrum_path"] = cfg["spectrum_folder"]

    if create_dirs:
        for key in OUTPUT_FOLDERS:
            os.makedirs(cfg[key], exist_ok=True)

    # --- Files ---
    tag    = cfg["dataset_tag"]
    study  = cfg["study_name"]
    epochs = cfg["epochs"]

    cfg["name_train"]  = as_file(cfg["data_folder"], f"train_{tag}.pt")
    cfg["name_valid"]  = as_file(cfg["data_folder"], f"valid_{tag}.pt")
    cfg["result_path"] = as_file(cfg["result_folder"], f"results{epochs}_{study}.pt")
    cfg["study_db"]    = "sqlite:///" + as_file(cfg["study_folder"], f"{study}.db")

    cfg["qubic_path"]        = as_file(root, cfg["mask_files"]["qubic"])
    cfg["so_hits_path"]      = as_file(root, cfg["mask_files"]["so_hits"])
    cfg["inho2d_path"]       = as_file(cfg["aux_folder"], cfg["inho2d_file"])
    cfg["apo_mask_path"]     = as_file(cfg["aux_folder"], cfg["apo_mask_file"])
    cfg["fisher_modes_path"] = as_file(cfg["data_folder"], cfg["fisher_modes_file"])

    tm = cfg["true_maps"]
    cfg["true_maps_paths"] = {
        "file_sph": as_file(cfg["data_folder"], tm["file_sph"]),
        "file_pt":  as_file(cfg["data_folder"], tm["file_pt"]),
        "result":   as_file(cfg["spectrum_folder"], tm["result_name"]),
    }

    nm = cfg["namaster"]
    cfg["namaster_paths"] = {
        "data":       as_file(cfg["data_folder"], nm["name_data"]),
        "coupling":   as_file(cfg["aux_folder"], nm["name_coupling"]),
        "noise_bias": as_file(cfg["aux_folder"], nm["name_noise_bias"]),
        "result":     as_file(cfg["spectrum_folder"], nm["name_result"]),
    }

    if "covariance" in cfg:
        cv = cfg["covariance"]
        cfg["covariance_paths"] = {
            "oqe_El":            as_file(cfg["spectrum_folder"], cv["oqe_El"]),
            "oqe_fisher":        [as_file(cfg["spectrum_folder"], f) for f in cv["oqe_fisher"]],
            "oqe_bias":          as_file(cfg["spectrum_folder"], cv["oqe_bias"]),
            "master_estimation": as_file(cfg["spectrum_folder"], cv["master_estimation"]),
        }

    if "cobaya" in cfg:
        cob = cfg["cobaya"]
        cfg["cobaya_paths"] = {
            "datafile": as_file(cfg["aux_folder"], cob["datafile"]),
            "output":   as_file(cfg["chains_folder"], cob["output"]),
        }
        for key in ("master_estimation", "oqe_El", "oqe_bias"):
            if key in cob:
                cfg["cobaya_paths"][key] = as_file(cfg["spectrum_folder"], cob[key])
        if "oqe_fisher" in cob:
            cfg["cobaya_paths"]["oqe_fisher"] = [as_file(cfg["spectrum_folder"], f) for f in cob["oqe_fisher"]]

    return cfg


def bin_edges(cfg):
    """Bin edges shared by every estimator (thin wrapper around utilities)."""
    import utilities
    return utilities.compute_bins_fractional(**cfg["bin_kwargs"])


if __name__ == "__main__":
    import pprint
    pprint.pprint(load_config(create_dirs=False))
