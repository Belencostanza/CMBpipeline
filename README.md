# DeepWiener-PyTorch
PyTorch implementation of neural network DeepWiener for Wiener filtering of Cosmic Microwave Background (CMB) polarization maps with inhomogeneous noise, including spherical-to-planar projections and power spectrum estimation.


# Description of the codes: 

Brief overview of the codes included in ``source``:
- ``config.dict``: input dictionary with all the parameters of the pipeline.
- ``config_loader.py``: Module that reads and validates ``config.dict`` and resolves the paths.
- ``geometry.py``: Module that builds the sky mask, the projection to the plane and the noise variance maps.
- ``projections.py``: Module with spherical-to-planar projections.
- ``make_variance_map.py``: 2D variance map of the projected inhomogeneous noise.
- ``make_dataset.py``: Module that creates the dataset.
- ``run_dataset.py``: creates the training and validation sets.
- ``network_2d.py``: neural network implementation in pytorch.
- ``losses.py``: Module with the loss functions used to train the neural network.
- ``training_opt_beam_changed.py``: training of the neural network with Optuna.
- ``getbestmodel_beam.py``: evaluation of the best trained model.
- ``PowerSpectrum.py``: Module used to generate noise bias and fisher matrix for the optimal quadratic power spectrum (OQE).
- ``compute_bias.py``, ``reduce_bias.py``: noise bias generation and average over realizations.
- ``compute_fisher.py``, ``reduce_fisher.py``: fisher matrix generation and average over realizations.
- ``make_true_maps.py``: power spectrum estimation (OQE) of simulations with a given "true" power spectrum.
- ``make_noise_bias.py``, ``compute_namaster.py``: Pseudo-Cl (NaMaster) BB estimation on the sphere for the "true" simulations.
- ``make_covariance.py``: covariance matrices for the likelihood.
- ``run_cobaya_bb.py``: cosmological parameter inference.

The folder ``slurm`` contains the job scripts used to run the pipeline in a cluster, and ``tutorial`` contains a notebook with an example of the whole pipeline.

# Requirements: 

- Pytorch (https://pytorch.org)
- Optuna (https://optuna.org)
- CAMB (https://camb.readthedocs.io/en/latest/index.html)
- Healpy (https://healpy.readthedocs.io)
- NaMaster (https://namaster.readthedocs.io)
- Cobaya (https://cobaya.readthedocs.io)


# Usage: 

The implemented codes here perform the WF reconstruction of CMB polarization maps with inhomogeneous noise applied, on a sky patch projected from the sphere to the plane, and the estimation of the BB power spectrum and the cosmological parameters. Two masks are presented: a QUBIC-like circular patch and an SO-like patch built from the hits map.

The dictionary input ``source/config.dict`` include these parameters:

- ``mask_type``: put equal to "circ" for the QUBIC-like mask, put equal to "rect" for the SO-like mask.
- ``nside``, ``npixels_x``, ``npixels_y``, ``radius_deg``, ``fact``: HEALPix resolution, size of the plane grid, radius of the patch and margin factor of the projection.
- ``r``, ``lmin``, ``lmax``, ``cosmo_fid``: fiducial power spectrum.
- ``binning``: bins of the power spectrum.
- ``fwhm_arcmin``: beam.
- ``noise_type``, ``inho_type``, ``DEFAULT_NOISE_LEVEL``: noise model. ``noise_type`` put equal to "ho" for homogeneous noise, put equal to "inho" for inhomogeneous noise.
- ``map_rescale_factor``: normalization factor for the dataset.
- ``epochs``, ``batch_size``: training parameters.
- ``nsims_train``, ``nsims_valid``, ``nsims_test``: number of maps for each set.
- ``root_folder``, ``data_folder``, ``aux_folder``, ``study_folder``, ``chains_folder``: paths where the dataset, auxiliary files, Optuna studies and chains are stored.
- ``model_folder``, ``loss_folder``, ``result_folder``, ``spectrum_folder``: paths where the models, losses, results and power spectrum products are stored.
- ``study_name``, ``dataset_tag``, ``spectrum_tag``: names of the Optuna study, the dataset and the power spectrum products.
- ``mask_files``: path of the QUBIC-like coverage and the SO-like hits map.
- ``oqe``: number of simulations used to estimate the noise bias term and the fisher matrix.
- ``true_maps``: "true" power spectrum (``r_new``, ``Alens_new``), number of simulations and seeds used by ``make_true_maps.py``.
- ``namaster``: parameters of the NaMaster estimation.
- ``covariance``, ``cobaya``: inputs of the covariance matrices and parameters of the MCMC.

To run the software (from ``source``): 

1. Edit the input dictionary ``config.dict`` with the desired example.
2. Run ``make_variance_map.py`` to create the 2D variance map of the noise.
3. Run ``run_dataset.py`` to create the dataset.
4. Run ``training_opt_beam_changed.py`` to train the neural network.
5. Run ``compute_fisher.py`` and ``reduce_fisher.py`` to calculate the fisher matrix, and ``compute_bias.py`` and ``reduce_bias.py`` to calculate the noise bias for the power spectrum estimation.
6. Run ``make_true_maps.py`` to estimate the power spectrum of simulations.
7. Run ``make_noise_bias.py`` and ``compute_namaster.py`` to compare with the NaMaster estimation (optional).
8. Run ``make_covariance.py`` and ``run_cobaya_bb.py`` for the cosmological parameter inference.

Steps 3 to 5 require a cluster: the job scripts are in ``slurm`` (``data.sh``, ``train_opt_beam_changed.sh``, ``submit_all.sh``, ``submit_all_bias.sh``, ``reduce_fisher_job.sh``, ``reduce_bias_job.sh``). The fisher matrix and noise bias are computed with one realization per job, see ``submit_all.sh`` and ``submit_all_bias.sh``. If the sky patch, beam, noise or fiducial power spectrum are changed, all the steps must be run again.

# Tutorial: 

The notebook ``tutorial/tutorial.ipynb`` runs the whole pipeline on one simulation, in about a minute on a laptop. Training the network and computing the fisher matrix and noise bias require a cluster, so the tutorial uses the products already computed for the SO-like case. They are not stored in the repository: they are in the release [tutorial-data-v1](https://github.com/Belencostanza/CMBtorch/releases/tag/tutorial-data-v1) (``tutorial_data.tar.gz``, ~40 MB), which contains:

- the trained network (best Optuna trial) and its Optuna study,
- the fisher matrix and noise bias (2000 simulations each),
- the 2D variance map of the noise, the covariance matrices for cobaya and the NaMaster noise bias,
- the SO-like hits map.

Before running the notebook, download them into ``tutorial/data``:

```bash
cd tutorial
bash get_tutorial_data.sh
```

or download ``tutorial_data.tar.gz`` manually from the release and extract it inside ``tutorial`` (``tar -xzf tutorial_data.tar.gz``).

These files are only needed for the tutorial: the scripts in ``source`` generate their own products following the steps above.

# Contact 

Feel free to contact me at belen@fcaglp.unlp.edu.ar for comments, questions and suggestions.
