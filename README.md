# DeepWiener-PyTorch
PyTorch implementation of neural network DeepWiener for Wiener filtering of Cosmic Microwave Background (CMB) polarization maps with inhomogeneous noise, including spherical-to-planar projections and power spectrum estimation.

# Description of the codes: 

Brief overview of some codes inclued in ``source``:
- ``projections.py``: Module with spherical-to-planar projections
- ``make_variance_map.py``: 2D variance map of the projected inhomogeneous noise
- ``make_dataset.py``: Module that creates the dataset
- ``network_2d.py``: neural network implementation in pytorch
- ``training_opt_beam_changed.py``: training of the neural network with Optuna
- ``losses.py``: Module with the loss functions used to train the neural networ
- ``PowerSpectrum.py``: Module used to generate noise bias and fisher matrix for the optimal quadratic power spectrum (OQE)
- ``compute_bias.py``: bias generation 
- ``compute_fisher.py``: fisher matrix generation   
- ``compute_namaster.py``: Pseudo-Cl (NaMaster) BB estimation on the sphere for the "true" simulations.
- ``run_cobaya_job.py``: cosmological parameter inference

# Requirements: 

- Pytorch 
- Optuna
- CAMB
- Healpy
- Namaster
- Cobaya

# Contact 

Feel free to contact me at belen@fcaglp.unlp.edu.ar for comments, questions and suggestions.

