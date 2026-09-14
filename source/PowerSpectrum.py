#author: Belén Costanza


import numpy as np
import sys
import os
import scipy 
from scipy import interpolate

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn #provides all the building blocks to build the neural network
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models #just for debugging
from torchvision import transforms
import torch

import camb
from camb import model, initialpower
import utilities
import optuna
import healpy as hp
from network_2d import DeepWiener_twochannels, DeepWiener_threechannels


class PowerSpectrum:

    #DEFAULT_NOISE_LEVEL = 5e-7

    def __init__(self, nside, nx, ny, dx, dy, fwhm, nbins, bins, factor, DEFAULT_NOISE_LEVEL, mask_type="circ", noise_type="ho", nsamples_bias=None, nsamples_fisher=None):

        """Class to calculate the noise bias term and the fisher matrix

        Params:

        nx, ny: number of pixels along x and y (can differ for non-square grids)

        """

        self.nside = nside
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.fwhm = np.radians(fwhm/60)
        self.nbins = nbins # number of bins
        self.bins = bins # bin_edges
        self.FACTOR = factor
        #self.mask = mask
        self.DEFAULT_NOISE_LEVEL = DEFAULT_NOISE_LEVEL
        self.nsamples_bias = nsamples_bias
        self.nsamples_fisher = nsamples_fisher
        self.noise_type = noise_type
        self.mask_type = mask_type
        self.tfac = np.sqrt((self.dx*self.dy)/(self.nx*self.ny))
        self.lmax = 3*self.nside - 1

        self.lmin_grid = 2*np.pi/(self.nx*self.dx) #only exact along x; only works if dx = dy and nx = ny

        # cached quantities for performance
        self._ell_flat_bin_indices = None
        self._ell_flat_valid_mask = None
        self._beam_window = None
        self._alm_lmax = None
        self._alm_ell = None
        self._alm_m = None


    def get_models(self, model_path, study_name, storage=None): #charge models once 

        # storage: optuna sqlite URL (cfg["study_db"]); default keeps the legacy
        # behaviour of looking for <study_name>.db in the working directory
        if storage is None:
            storage = f"sqlite:///{study_name}.db"
        study = optuna.load_study(study_name=study_name, storage=storage)
        trial = study.best_trial

        filters0 = trial.params["filters0"]
        filters1 = trial.params["filters1"]
        filters2 = trial.params["filters2"]
        filters3 = trial.params["filters3"]
        filters4 = filters3
        filters5 = filters4
        lr = trial.params["lr"]
        wd = trial.params["wd"]

        hyperparameters = [filters0, filters1, filters2, filters3, lr, wd]#, lambda_weight]
        filters = [filters0, filters1, filters2, filters3, filters4, filters5]

        resultsname = utilities.hyper(hyperparameters)
        
        in_channels = 2
        out_channels = 2

        if self.noise_type == "ho":
            model = DeepWiener_twochannels(in_channels, out_channels, filters=filters)
        elif self.noise_type == "inho":
            model = DeepWiener_threechannels(in_channels, out_channels, filters=filters)

        model.load_state_dict(torch.load(model_path + resultsname, map_location=torch.device('cpu'))['model_state_dict'])

        return model

    def flat_spectrum_xy(self, cl_angular):

        lx,ly=np.meshgrid( np.fft.fftfreq(self.nx, d=self.dx)*2*np.pi, np.fft.fftfreq(self.ny, d=self.dy)*2*np.pi )
        l = np.sqrt(lx**2 + ly**2) 
        ell_flat = l.flatten()
    
        #ell_flat = ell_flat*self.lmin_grid#*18.
   
        ls = np.arange(len(cl_angular))
        inter = scipy.interpolate.interp1d(ls[2:], cl_angular[2:],bounds_error=False,fill_value=np.min(cl_angular[2:]))
        cl_plano = inter(ell_flat)
   
        return ell_flat, cl_plano


    def flat_spectrum(self, cl_angular):

        lx,ly=np.meshgrid( np.fft.fftfreq( self.nx, 1/float(self.nx) ),np.fft.fftfreq( self.ny, 1/float(self.ny) ) )
        l = np.sqrt(lx**2 + ly**2) 
        ell_flat = l.flatten()
    
        ell_flat = ell_flat*self.lmin_grid#*18.
   
        ls = np.arange(len(cl_angular))
        inter = scipy.interpolate.interp1d(ls[2:], cl_angular[2:],bounds_error=False,fill_value=np.min(cl_angular[2:]))
        cl_plano = inter(ell_flat)
   
        return ell_flat, cl_plano

    def flat_spectrum_real(self, cl_angular):

        lx,ly=np.meshgrid( np.fft.rfftfreq( self.nx, 1/float(self.nx) ),np.fft.fftfreq( self.ny, 1/float(self.ny) ) )
        l = np.sqrt(lx**2 + ly**2) 
        ell_flat = l.flatten()
    
        ell_flat = ell_flat*self.lmin_grid#*18.
   
        ls = np.arange(len(cl_angular))
        inter = scipy.interpolate.interp1d(ls[2:], cl_angular[2:],bounds_error=False,fill_value=np.min(cl_angular[2:]))
        cl_plano = inter(ell_flat)
   
        return ell_flat, cl_plano

    def spectrum_unique(self, ell_flat, cl_plane):

        ell_flat_unique, indices = np.unique(ell_flat, return_index=True)
        cl_flat_unique = cl_plane[indices]

        return ell_flat_unique, cl_flat_unique

    def bineado(self, ell, cl): 

        bin_indices = np.digitize(ell, self.bins, right=False)

        count = np.zeros(len(self.bins) - 1)
        cl_bin_sum = np.zeros(len(self.bins) - 1)
        el_med_sum = np.zeros(len(self.bins) - 1)

        # Calculate sum of ell values and cl values for each bin
        for i in range(1, len(self.bins)):
            mask = (bin_indices == i)
            count[i - 1] = np.sum(mask)
            cl_bin_sum[i - 1] = np.sum(cl[mask])
            el_med_sum[i - 1] = np.sum(ell[mask] * cl[mask])

        # Calculate the binned results
        el_med = (el_med_sum / cl_bin_sum).astype(int)
        cl_bin = cl_bin_sum / count
        
        return el_med, cl_bin, count
    
    def bineado_namaster_like(self, ell, cl):
        
        cl_bin = []
        ell_eff = []
        
        for i in range(len(self.bins)-1):
            mask = (ell >= self.bins[i]) & (ell < self.bins[i+1])
            
            l = ell[mask]
            c = cl[mask]
            
            w = 2*l + 1
            
            clb = np.sum(w * c) / np.sum(w)
            leff = np.sum(w * l) / np.sum(w)
            
            cl_bin.append(clb)
            ell_eff.append(leff)
            
        return np.array(ell_eff), np.array(cl_bin)


    def bineado_weights(self, ell, cl):

        bin_indices = np.digitize(ell, self.bins) - 1

        nb = len(self.bins) - 1
        cl_bin = np.zeros(nb)
        ell_eff = np.zeros(nb)
        weight_sum = np.zeros(nb)

        for b in range(nb):
            mask = (bin_indices == b)

            if np.sum(mask) == 0:
                continue

            w = ell[mask]

            cl_bin[b] = np.sum(w * cl[mask]) / np.sum(w)
            ell_eff[b] = np.sum(w * ell[mask]) / np.sum(w)
            weight_sum[b] = np.sum(w)

        return ell_eff, cl_bin, weight_sum


    def power(self, map):

        fft = np.fft.fft2(map[:,:])*self.tfac
        fft_shape = fft.shape 
        power = np.real(fft*np.conj(fft))
        power_reshape = np.reshape(power, [fft_shape[0]*fft_shape[1]]) 

        return power_reshape

    def power_real(self, map):

        rfft = np.fft.rfft2(map[:,:])*self.tfac
        rfft_shape = rfft.shape 
        power = np.real(rfft*np.conj(rfft))
        power_reshape = np.reshape(power, [rfft_shape[0]*rfft_shape[1]]) 
        
        return power_reshape, rfft

    def preprocess(self, data):

        if self.noise_type == "ho":
            qobs = data[:,0:1,:,:]
            uobs = data[:,1:2,:,:]
            mask = data[:,2:3,:,:]

            qobs_norm = (qobs - qobs.mean(dim=(2, 3), keepdim=True))*self.FACTOR
            uobs_norm = (uobs - uobs.mean(dim=(2, 3), keepdim=True))*self.FACTOR

            obs_norm = torch.cat((qobs_norm, uobs_norm, mask), dim=1)
        
        elif self.noise_type == "inho":
            qobs = (data[:,0:1,:,:])
            uobs = (data[:,1:2,:,:])
            mask = (data[:,2:3,:,:])
            inho = (data[:,3:4,:,:])

            qobs_norm = (qobs - qobs.mean(dim=(2, 3), keepdim=True))*self.FACTOR
            uobs_norm = (uobs - uobs.mean(dim=(2, 3), keepdim=True))*self.FACTOR
            obs_norm = torch.cat((qobs_norm, uobs_norm, mask, inho), dim=1)

        return obs_norm


    def prediction(self, data, model):

        """
        data: torch.Tensor (N, 3, npix, npix)

        """

        images = self.preprocess(data)

        model.eval()
        with torch.no_grad():
            result = model(images) / self.FACTOR

        qcnn = result[:,0,:,:].cpu().numpy()
        ucnn = result[:,1,:,:].cpu().numpy()

        return qcnn, ucnn


    def get_nn_outputs(self, data, model):

        qcnn, ucnn = self.prediction(data, model)  
        efft_cnn, bfft_cnn = utilities.transf_eb2_np(qcnn, ucnn, self.nx, self.dx, self.ny, self.dy)
        ecnn = np.fft.irfft2(efft_cnn)/self.tfac
        bcnn = np.fft.irfft2(bfft_cnn)/self.tfac

        return ecnn, bcnn # shape (N, nx, nx)


    def El_fid(self, ell_flat, cl_plane, power, cl_bin): 

        # Cache digitized bin indices for ell_flat (depends only on geometry)
        if (
            self._ell_flat_bin_indices is None
            or self._ell_flat_bin_indices.shape != ell_flat.shape
        ):
            bin_indices = np.digitize(ell_flat, self.bins) - 1
            valid_mask = (bin_indices >= 0) & (bin_indices < self.nbins)

            self._ell_flat_bin_indices = bin_indices
            self._ell_flat_valid_mask = valid_mask

        bin_indices = self._ell_flat_bin_indices
        valid_mask = self._ell_flat_valid_mask

        # Restrict to valid modes and accumulate per-bin with np.bincount
        weights = power[valid_mask] / cl_plane[valid_mask]

        El_sum = np.bincount(
            bin_indices[valid_mask],
            weights=weights,
            minlength=self.nbins,
        )

        El = 0.5 * El_sum / cl_bin

        return El

    def El_fid_weights(self, ell_flat, cl_plane, power, cl_bin):

        if (
            self._ell_flat_bin_indices is None
            or self._ell_flat_bin_indices.shape != ell_flat.shape):

            bin_indices = np.digitize(ell_flat, self.bins) - 1
            valid_mask = (bin_indices >= 0) & (bin_indices < self.nbins)

            self._ell_flat_bin_indices = bin_indices
            self._ell_flat_valid_mask = valid_mask

        bin_indices = self._ell_flat_bin_indices
        valid_mask = self._ell_flat_valid_mask

        w = ell_flat[valid_mask]
        weights = w * (power[valid_mask] / cl_plane[valid_mask])

        El_sum = np.bincount(
            bin_indices[valid_mask],
            weights=weights,
            minlength=self.nbins,
        )

        norm = np.bincount(
            bin_indices[valid_mask],
            weights=w,
            minlength=self.nbins,
            )

        El = 0.5 * El_sum / (norm * cl_bin)

        return El

    #constant perturbation on angular spectrum in each mode of the bin
    def perturbation(self, cl_ang, k):   
        cl_ang_pert = np.zeros((len(self.bins)-1,len(cl_ang)))
    
        for j in range(len(self.bins) -1):    
            cl_ang_pert[j,:] =  cl_ang[:]
            cl_ang_pert[j,self.bins[j]:self.bins[j+1]] = cl_ang[self.bins[j]:self.bins[j+1]] + k*cl_ang[self.bins[j]:self.bins[j+1]]

        return cl_ang_pert, k


    # perturbation proportional to cl 
    def perturbation_plane(self, ell_flat, cl_plane, k):   
    
        cl_flat_pert_prop = np.zeros((len(self.bins)-1, len(cl_plane)))
        for j in range(len(self.bins)-1):
            cl_flat_pert_prop[j] = cl_plane[:]
            for i in range(len(cl_plane)):
                el = ell_flat[i]
                if el>=self.bins[j] and el<self.bins[j+1]:
                    cl_flat_pert_prop[j,i]=cl_plane[i] + k*cl_plane[i]

        return cl_flat_pert_prop, k


    # pertubation proportional to cl (angular case)
    def perturbation_angular(self, ell, cl_angular, k): 

        """

        ell        : array de ell (0...lmax)
        cl_angular : C_ell fiducial (EE o BB)
        k          : amplitud de perturbación

        """

        cl_pert = np.zeros((len(self.bins)-1, len(cl_angular)))

        for j in range(len(self.bins)-1):
            cl_pert[j] = cl_angular.copy()

            lmin, lmax = self.bins[j], self.bins[j+1]
            mask = (ell >= lmin) & (ell < lmax)

            cl_pert[j, mask] *= (1.0 + k)

        return cl_pert, k


    #random perturbation, plane case
    def mode_perturbation(self, rfft_fiducial, ell_flat, cl_flat_pert):


        #rfft_fiducial is a real_rfft
        rfft_shape = rfft_fiducial.shape
        rfft_reshape = np.reshape(rfft_fiducial, [rfft_shape[0]*rfft_shape[1]])
 
        rfft_pert = np.zeros((len(self.bins)-1,len(rfft_reshape)), dtype=complex) 
        rfft_pert[:,:] = rfft_reshape.copy()
    
        for i in range(len(ell_flat)):
            el = ell_flat[i]    
            for j in range(len(self.bins) -1):
                    if el>=self.bins[j] and el<self.bins[j+1]:
                        ran1 = np.random.normal(scale=np.sqrt(cl_flat_pert[j,i]/2))
                        ran2 = np.random.normal(scale=np.sqrt(cl_flat_pert[j,i]/2))
                        rfft_pert[j,i] = ran1 + ran2*1j
    
        signal_pert = np.zeros((self.nbins, self.ny, self.nx))
        for i in range(len(self.bins)-1):
            rfft_pert_reshape = np.reshape(rfft_pert[i,:], [rfft_shape[0], rfft_shape[1]])
            signal_pert[i,:,:] = np.fft.irfft2(rfft_pert_reshape/self.tfac)#tfac

        return rfft_pert, signal_pert


    #keep phase fixed, plane case
    def mode_perturbation_phase(self, rfft_fiducial, ell_flat, cl_flat_pert):

    
        rfft_shape = rfft_fiducial.shape
        rfft_reshape = np.reshape(rfft_fiducial, [rfft_shape[0]*rfft_shape[1]])
 
        rfft_pert = np.zeros((len(self.bins)-1,len(rfft_reshape)), dtype=complex) #shape (25,len(ell_flat))
        rfft_pert[:,:] = rfft_reshape.copy()
    
        for i in range(len(ell_flat)):
            el = ell_flat[i]    
            for j in range(len(self.bins) -1):
                    if el>=self.bins[j] and el<self.bins[j+1]:
                        phase = np.arctan2(rfft_reshape[i].imag, rfft_reshape[i].real)
                        modulo = np.random.normal(scale=np.sqrt(cl_flat_pert[j,i]))
                        x = modulo*np.cos(phase)
                        y = modulo*np.sin(phase)
                        rfft_pert[j,i] = x + y*1j

        rfft_pert = np.reshape(rfft_pert, (len(self.bins)-1, rfft_shape[0], rfft_shape[1]))

        return rfft_pert

    def alm_perturbation2(self, alm_fiducial, cl_pert, lmax=1600): 

        # CHECK VERSION
        """
        Perturba los alm manteniendo la fase fija y modificando el módulo
        según C_ell perturbado (análogo a mode_perturbation_phase).

        """

        alm_pert = np.zeros((len(self.bins)-1,len(alm_fiducial)), dtype=complex) #shape (nbins,len(ell_flat))

        ell, m = hp.Alm.getlm(lmax)

        for i in range(len(alm_fiducial)):
            l = ell[i]

            # si usás bins
            for j in range(len(self.bins)-1):

                if self.bins[j] <= l < self.bins[j+1]:
                    Cl = cl_pert[j, l]
                    ran1 = np.random.normal(scale=np.sqrt(Cl / 2))
                    ran2 = np.random.normal(scale=np.sqrt(Cl / 2))
                    alm_pert[j,i] = ran1 + 1j * ran2

        return alm_pert

    def alm_perturbation_phase(self, alm_fiducial, cl_pert, lmax=1600):

        """
        Genera nuevos alm manteniendo la fase del fiducial,
        pero con amplitudes aleatorias consistentes con cl_pert.
        
        alm_fiducial : array (nalm,)
        cl_pert      : shape (nbins, lmax+1)
        """
        
        if self._alm_lmax != lmax or self._alm_ell is None:
            ell, m = hp.Alm.getlm(lmax)
            self._alm_lmax = lmax
            self._alm_ell = ell
            self._alm_m = m
        else:
            ell = self._alm_ell
            
        alm_pert = np.zeros((len(self.bins)-1, len(alm_fiducial)), dtype=complex)
        alm_pert[:,:] = alm_fiducial
        
        # fase del fiducial
        phase = np.arctan2(alm_fiducial.imag, alm_fiducial.real)
        
        for j in range(len(self.bins)-1):
            
            alm_j = np.zeros_like(alm_fiducial, dtype=complex)
            lmin, lmax_bin = self.bins[j], self.bins[j+1]
            mask = (ell >= lmin) & (ell < lmax_bin)
            
            # sampleo amplitud
            sigma = np.sqrt(cl_pert[j, ell[mask]])
            modulo = np.random.normal(scale=sigma)
            
            # reconstrucción con fase fija
            x = modulo * np.cos(phase[mask])
            y = modulo * np.sin(phase[mask])
            
            alm_j[mask] = x + 1j*y
            alm_pert[j] = alm_j
            
        return alm_pert

    # keep phase fixed, angular case
    def alm_perturbation(self, alm_fiducial, cl_fid, cl_pert, lmax=1600):

        """

        Perturba alm bin por bin manteniendo la fase fija.

        alm_fiducial : array (nalm,)
        cl_fid       : C_ell fiducial (len lmax+1)
        cl_pert      : shape (nbins, lmax+1)

        """

        if self._alm_lmax != lmax or self._alm_ell is None:
            ell, m = hp.Alm.getlm(lmax)
            self._alm_lmax = lmax
            self._alm_ell = ell
            self._alm_m = m
        else:
            ell = self._alm_ell

        alm_pert = np.zeros((len(self.bins)-1, len(alm_fiducial)), dtype=complex)

        for j in range(len(self.bins)-1):

            alm_j = alm_fiducial.copy()

            lmin, lmax_bin = self.bins[j], self.bins[j+1]
            mask = (ell >= lmin) & (ell < lmax_bin)

            alm_j[mask] *= np.sqrt(
                cl_pert[j, ell[mask]] / cl_fid[ell[mask]]
                )

            alm_pert[j] = alm_j

        return alm_pert

    def make_rot(self, q: np.ndarray, u: np.ndarray, psi: np.ndarray):

        """
        Rotate Q and U maps by angle psi.

        Args:
            q: Q map
            u: U map
            psi: Rotation angle array

        Returns:
            Tuple of (q_rot, u_rot) rotated maps
        """

        q_rot = q * np.cos(2*psi) + u * np.sin(2*psi)
        u_rot = -q * np.sin(2*psi) + u * np.cos(2*psi)

        return q_rot, u_rot


# function that calculates noise bias term
    def noise_bias(self, data, model, ell_flat, clee_flat, clbb_flat, clee_bin, clbb_bin):

        ecnn, bcnn = self.get_nn_outputs(data, model)
    
        El_e = np.zeros((self.nsamples_bias,len(self.bins)-1))
        El_b = np.zeros((self.nsamples_bias,len(self.bins)-1))

        for i in range(self.nsamples_bias):

            power_e = self.power(ecnn[i])
            power_b = self.power(bcnn[i])

            El_e[i] = self.El_fid(ell_flat, clee_flat, power_e, clee_bin) 
            El_b[i] = self.El_fid(ell_flat, clbb_flat, power_b, clbb_bin) 
    
        bl_e = np.mean(El_e, axis=0)
        bl_b = np.mean(El_b, axis=0)

        return bl_e, bl_b

    def generate_one_inho_pln(self, z, almE, almB, mask, noiseQ, noiseU, psi, valid_index, proj, inho2d, mask2d=None):


        data = torch.zeros((1, 4, self.ny, self.nx), dtype=torch.float32)

        if self._beam_window is None:
            self._beam_window = utilities.bl(self.fwhm, lmax=1600)
        B_ell = self._beam_window

            
        almE_beam = hp.almxfl(almE, B_ell)
        almB_beam = hp.almxfl(almB, B_ell)

        Q_sky, U_sky = hp.alm2map_spin(
            [almE_beam, almB_beam],
            nside=self.nside,
            spin=2, lmax=1600)

        dataQ = mask * (Q_sky + noiseQ)
        dataU = mask * (U_sky + noiseU) # noise realization from variance map

        Q_obs_rot, U_obs_rot = self.make_rot(
            dataQ[valid_index],
            dataU[valid_index],
            psi)

        if self.mask_type == "rect":
            gridQ_data = proj.grid_bins_mask_from_res_margin(z, Q_obs_rot, mask2d)
            gridU_data = proj.grid_bins_mask_from_res_margin(z, U_obs_rot, mask2d)
        else:
            gridQ_data  = proj.grid_bins_mask_margin(z, Q_obs_rot, mask2d)
            gridU_data  = proj.grid_bins_mask_margin(z, U_obs_rot, mask2d)

        #mask2d_index = np.where(gridQ_data != 0)
        #mask2d = np.zeros((self.npixels,self.npixels))
        #mask2d[mask2d_index] = 1

        data[:, 0, :, :] = torch.from_numpy(gridQ_data)
        data[:, 1, :, :] = torch.from_numpy(gridU_data)
        data[:, 2, :, :] = torch.from_numpy(mask2d)
        data[:, 3, :, :] = torch.from_numpy(inho2d)

        return data

    def generate_one_pln_binning(self, z, almE, almB, mask40, noiseQ, noiseU, psi, valid_index, proj, mask2d):


        data = torch.zeros((1, 3, self.ny, self.nx), dtype=torch.float32)

        if self._beam_window is None:
            self._beam_window = utilities.bl(self.fwhm, lmax=1600)
        B_ell = self._beam_window

        almE_beam = hp.almxfl(almE, B_ell)
        almB_beam = hp.almxfl(almB, B_ell)

        Q_sky, U_sky = hp.alm2map_spin(
            [almE_beam, almB_beam],
            nside=self.nside,
            spin=2, lmax=1600)

        dataQ = mask40 * (Q_sky + noiseQ)
        dataU = mask40 * (U_sky + noiseU)

        Q_obs_rot, U_obs_rot = self.make_rot(
            dataQ[valid_index],
            dataU[valid_index],
            psi)

        #gridQ_data, _, _ = proj.grid_bins(z, Q_obs_rot)
        #gridU_data, _, _ = proj.grid_bins(z, U_obs_rot)
        gridQ_data  = proj.grid_bins_binning(z, Q_obs_rot, mask2d)
        gridU_data  = proj.grid_bins_binning(z, U_obs_rot, mask2d)

        data[:, 0, :, :] = torch.from_numpy(gridQ_data)
        data[:, 1, :, :] = torch.from_numpy(gridU_data)
        data[:, 2, :, :] = torch.from_numpy(mask2d)

        return data


    def generate_one_pln(self, z, almE, almB, mask40, noiseQ, noiseU, psi, valid_index, proj, mask2d):


        data = torch.zeros((1, 3, self.ny, self.nx), dtype=torch.float32)

        if self._beam_window is None:
            self._beam_window = utilities.bl(self.fwhm, lmax=1600)
        B_ell = self._beam_window
            
        almE_beam = hp.almxfl(almE, B_ell)
        almB_beam = hp.almxfl(almB, B_ell)

        Q_sky, U_sky = hp.alm2map_spin(
            [almE_beam, almB_beam],
            nside=self.nside,
            spin=2, lmax=1600)

        dataQ = mask40 * (Q_sky + noiseQ)
        dataU = mask40 * (U_sky + noiseU)

        Q_obs_rot, U_obs_rot = self.make_rot(
            dataQ[valid_index],
            dataU[valid_index],
            psi)

        #gridQ_data, _, _ = proj.grid_bins(z, Q_obs_rot)
        #gridU_data, _, _ = proj.grid_bins(z, U_obs_rot)
        #gridQ_data  = proj.grid_bins_mask_margin(z, Q_obs_rot, mask2d)
        #gridU_data  = proj.grid_bins_mask_margin(z, U_obs_rot, mask2d)
        if self.mask_type == "rect":
            gridQ_data = proj.grid_bins_mask_from_res_margin(z, Q_obs_rot, mask2d)
            gridU_data = proj.grid_bins_mask_from_res_margin(z, U_obs_rot, mask2d)
        else:
            gridQ_data = proj.grid_bins_mask_margin_deg(z, Q_obs_rot, mask2d)
            gridU_data = proj.grid_bins_mask_margin_deg(z, U_obs_rot, mask2d)


        data[:, 0, :, :] = torch.from_numpy(gridQ_data)
        data[:, 1, :, :] = torch.from_numpy(gridU_data)
        data[:, 2, :, :] = torch.from_numpy(mask2d)

        return data

    def generate_pln_binning(self, z, almE, almB, mask40, noiseQ, noiseU, psi, valid_index, proj, all = True, mask2d = None):

        if all:
            nsamples = self.nsamples_fisher
        else:
            nsamples = self.nbins

        data = torch.zeros((nsamples, 3, self.ny, self.nx), dtype=torch.float32)

        if self._beam_window is None:
            self._beam_window = utilities.bl(self.fwhm, lmax=1600)
        B_ell = self._beam_window

        for i in range(nsamples):

            #print(hp.Alm.getlmax(len(almE[0])))
            almE_beam = hp.almxfl(almE[i], B_ell)
            almB_beam = hp.almxfl(almB[i], B_ell)

            Q_sky, U_sky = hp.alm2map_spin(
                [almE_beam, almB_beam],
                nside=self.nside,
                spin=2, lmax=1600)

            dataQ = mask40 * (Q_sky + noiseQ)
            dataU = mask40 * (U_sky + noiseU)

            Q_obs_rot, U_obs_rot = self.make_rot(
                dataQ[valid_index],
                dataU[valid_index],
                psi)

            # Keep the same projection/masking convention between fiducial and perturbed maps.
            if mask2d is not None:
                gridQ_data = proj.grid_bins_binning(z, Q_obs_rot, mask2d)
                gridU_data = proj.grid_bins_binning(z, U_obs_rot, mask2d)
                mask = mask2d
            else:
                gridQ_data, _, _ = proj.grid_bins(z, Q_obs_rot)
                gridU_data, _, _ = proj.grid_bins(z, U_obs_rot)

                mask_index = np.where(gridQ_data != 0)
                mask = np.zeros((self.ny,self.nx))
                mask[mask_index] = 1

            data[i, 0, :, :] = torch.from_numpy(gridQ_data)
            data[i, 1, :, :] = torch.from_numpy(gridU_data)
            data[i, 2, :, :] = torch.from_numpy(mask)

        return data


    def generate_pln(self, z, almE, almB, mask40, noiseQ, noiseU, psi, valid_index, proj, all = True, mask2d = None):

        if all: 
            nsamples = self.nsamples_fisher
        else: 
            nsamples = self.nbins

        data = torch.zeros((nsamples, 3, self.ny, self.nx), dtype=torch.float32)

        if self._beam_window is None:
            self._beam_window = utilities.bl(self.fwhm, lmax=1600)
        B_ell = self._beam_window

        for i in range(nsamples):

            #print(hp.Alm.getlmax(len(almE[0])))
            almE_beam = hp.almxfl(almE[i], B_ell)
            almB_beam = hp.almxfl(almB[i], B_ell)

            Q_sky, U_sky = hp.alm2map_spin(
                [almE_beam, almB_beam],
                nside=self.nside,
                spin=2, lmax=1600)

            dataQ = mask40 * (Q_sky + noiseQ)
            dataU = mask40 * (U_sky + noiseU)

            Q_obs_rot, U_obs_rot = self.make_rot(
                dataQ[valid_index],
                dataU[valid_index],
                psi)

            # Keep the same projection/masking convention between fiducial and perturbed maps.
            if mask2d is not None:
                if self.mask_type == "rect":
                    gridQ_data = proj.grid_bins_mask_from_res_margin(z, Q_obs_rot, mask2d)
                    gridU_data = proj.grid_bins_mask_from_res_margin(z, U_obs_rot, mask2d)
                else:
                    gridQ_data = proj.grid_bins_mask_margin_deg(z, Q_obs_rot, mask2d)
                    gridU_data = proj.grid_bins_mask_margin_deg(z, U_obs_rot, mask2d)
                mask = mask2d
            else:
                gridQ_data, _, _ = proj.grid_bins(z, Q_obs_rot)
                gridU_data, _, _ = proj.grid_bins(z, U_obs_rot)

                mask_index = np.where(gridQ_data != 0)
                mask = np.zeros((self.ny,self.nx))
                mask[mask_index] = 1

            data[i, 0, :, :] = torch.from_numpy(gridQ_data)
            data[i, 1, :, :] = torch.from_numpy(gridU_data)
            data[i, 2, :, :] = torch.from_numpy(mask)

        return data

    def generate_inho_pln(self, z, almE, almB, mask40, noiseQ, noiseU, psi, valid_index, proj, all = True, mask2d = None, inho2d = None):

        if all: 
            nsamples = self.nsamples_fisher
        else: 
            nsamples = self.nbins

        data = torch.zeros((nsamples, 4, self.ny, self.nx), dtype=torch.float32)

        if self._beam_window is None:
            self._beam_window = utilities.bl(self.fwhm, lmax=1600)
        B_ell = self._beam_window

        for i in range(nsamples):

            #print(hp.Alm.getlmax(len(almE[0])))
            almE_beam = hp.almxfl(almE[i], B_ell)
            almB_beam = hp.almxfl(almB[i], B_ell)

            Q_sky, U_sky = hp.alm2map_spin(
                [almE_beam, almB_beam],
                nside=self.nside,
                spin=2, lmax=1600)

            dataQ = mask40 * (Q_sky + noiseQ)
            dataU = mask40 * (U_sky + noiseU)

            Q_obs_rot, U_obs_rot = self.make_rot(
                dataQ[valid_index],
                dataU[valid_index],
                psi)

            # Keep the same projection/masking convention between fiducial and perturbed maps.
            if mask2d is not None:
                if self.mask_type == "rect":
                    gridQ_data = proj.grid_bins_mask_from_res_margin(z, Q_obs_rot, mask2d)
                    gridU_data = proj.grid_bins_mask_from_res_margin(z, U_obs_rot, mask2d)
                else:
                    gridQ_data = proj.grid_bins_mask_margin(z, Q_obs_rot, mask2d)
                    gridU_data = proj.grid_bins_mask_margin(z, U_obs_rot, mask2d)
                mask = mask2d
            else:
                gridQ_data, _, _ = proj.grid_bins(z, Q_obs_rot)
                gridU_data, _, _ = proj.grid_bins(z, U_obs_rot)

                mask_index = np.where(gridQ_data != 0)
                mask = np.zeros((self.ny,self.nx))
                mask[mask_index] = 1

            data[i, 0, :, :] = torch.from_numpy(gridQ_data)
            data[i, 1, :, :] = torch.from_numpy(gridU_data)
            data[i, 2, :, :] = torch.from_numpy(mask)
            data[i, 3, :, :] = torch.from_numpy(inho2d)

        return data

    def fiducial_one(self, data, model, ell_flat, clee_flat, clbb_flat, clee_bin, clbb_bin):


        ecnn, bcnn = self.get_nn_outputs(data, model)

        power_e = self.power(ecnn[0])
        power_b = self.power(bcnn[0])

        El_e = self.El_fid(ell_flat, clee_flat, power_e, clee_bin) 
        El_b = self.El_fid(ell_flat, clbb_flat, power_b, clbb_bin) 

        return El_e, El_b

    def fiducial_one_weights(self, data, model, ell_flat, clee_flat, clbb_flat, clee_bin, clbb_bin):


        ecnn, bcnn = self.get_nn_outputs(data, model)

        power_e = self.power(ecnn[0])
        power_b = self.power(bcnn[0])

        El_e = self.El_fid_weights(ell_flat, clee_flat, power_e, clee_bin) 
        El_b = self.El_fid_weights(ell_flat, clbb_flat, power_b, clbb_bin) 

        return El_e, El_b


    def fiducial(self, data, model, ell_flat, clee_flat, clbb_flat, clee_bin, clbb_bin, all=True):

        if all: 
            nsamples = self.nsamples_fisher
        else: 
            nsamples = self.nbins

        ecnn, bcnn = self.get_nn_outputs(data, model)

        El_e = np.zeros((nsamples,len(self.bins)-1))
        El_b = np.zeros((nsamples,len(self.bins)-1))

        for i in range(nsamples):

            power_e = self.power(ecnn[i])
            power_b = self.power(bcnn[i])

            El_e[i] = self.El_fid(ell_flat, clee_flat, power_e, clee_bin) 
            El_b[i] = self.El_fid(ell_flat, clbb_flat, power_b, clbb_bin) 

        return El_e, El_b

    def fiducial_weights(self, data, model, ell_flat, clee_flat, clbb_flat, clee_bin, clbb_bin, all=True):

        if all: 
            nsamples = self.nsamples_fisher
        else: 
            nsamples = self.nbins

        ecnn, bcnn = self.get_nn_outputs(data, model)

        El_e = np.zeros((nsamples,len(self.bins)-1))
        El_b = np.zeros((nsamples,len(self.bins)-1))

        for i in range(nsamples):

            power_e = self.power(ecnn[i])
            power_b = self.power(bcnn[i])

            El_e[i] = self.El_fid_weights(ell_flat, clee_flat, power_e, clee_bin) 
            El_b[i] = self.El_fid_weights(ell_flat, clbb_flat, power_b, clbb_bin) 

        return El_e, El_b


    def fisher(self, ele, Elpert_e, Elpert_b, Elfid_e, Elfid_b, c, clee_bin, clbb_bin, cl_pert_prop_e, cl_pert_prop_b):
        
        fisher_matrix_e = np.zeros((len(self.bins)-1,len(self.bins)-1))
        fisher_matrix_b = np.zeros((len(self.bins)-1,len(self.bins)-1))

        for j in range(len(self.bins)-1):

            _, clee_bin_pert, _ = self.bineado(ele, cl_pert_prop_e[j])
            _, clbb_bin_pert, _ = self.bineado(ele, cl_pert_prop_b[j])
            #_, clee_bin_pert = self.bineado_namaster_like(ele, cl_pert_prop_e[j])
            #_, clbb_bin_pert = self.bineado_namaster_like(ele, cl_pert_prop_b[j])
            
            delta_theta_e = clee_bin_pert[j] - clee_bin[j]
            delta_theta_b = clbb_bin_pert[j] - clbb_bin[j]
            
            fisher_matrix_e[:,j] = (Elpert_e[j] - Elfid_e)/(delta_theta_e)
            fisher_matrix_b[:,j] = (Elpert_b[j] - Elfid_b)/(delta_theta_b)
            
        return fisher_matrix_e, fisher_matrix_b


    def fisher_parallel_binning(self, z, mask40, model, almE_fid, almB_fid, Elfid_e, Elfid_b, ell_flat, clee_ang, clbb_ang, clee_flat, clbb_flat, clee_bin, clbb_bin, cl_pert_prop_e, cl_pert_prop_b, c, proj, noiseQ, noiseU, valid_index, psi, mask2d):

        almE_pert = self.alm_perturbation(almE_fid, clee_ang, cl_pert_prop_e)
        almB_pert = self.alm_perturbation(almB_fid, clbb_ang, cl_pert_prop_b)

        data_pert = self.generate_pln_binning(z, almE_pert, almB_pert, mask40, noiseQ, noiseU, psi, valid_index, proj, all = False, mask2d=mask2d)

        Elpert_e, Elpert_b = self.fiducial(data_pert, model, ell_flat, clee_flat, clbb_flat, clee_bin, clbb_bin, all=False) # (nbins, n_modes)
        fisher_value_e, fisher_value_b = self.fisher(Elpert_e, Elpert_b, Elfid_e, Elfid_b, c, clee_bin, clbb_bin)

        return fisher_value_e, fisher_value_b


    def fisher_parallel(self, ele, z, mask40, model, almE_fid, almB_fid, Elfid_e, Elfid_b, ell_flat, clee_ang, clbb_ang, clee_flat, clbb_flat, clee_bin, clbb_bin, cl_pert_prop_e, cl_pert_prop_b, c, proj, noiseQ, noiseU, valid_index, psi, mask2d, inho2d=None):

        almE_pert = self.alm_perturbation_phase(almE_fid, cl_pert_prop_e)
        almB_pert = self.alm_perturbation_phase(almB_fid, cl_pert_prop_b)

        if self.noise_type == "ho":
            data_pert = self.generate_pln(z, almE_pert, almB_pert, mask40, noiseQ, noiseU, psi, valid_index, proj, all = False, mask2d=mask2d)
        elif self.noise_type == "inho":
            data_pert = self.generate_inho_pln(z, almE_pert, almB_pert, mask40, noiseQ, noiseU, psi, valid_index, proj, all = False, mask2d=mask2d, inho2d=inho2d)

        Elpert_e, Elpert_b = self.fiducial(data_pert, model, ell_flat, clee_flat, clbb_flat, clee_bin, clbb_bin, all=False) # (nbins, n_modes)
        fisher_value_e, fisher_value_b = self.fisher(ele, Elpert_e, Elpert_b, Elfid_e, Elfid_b, c, clee_bin, clbb_bin, cl_pert_prop_e, cl_pert_prop_b)

        return fisher_value_e, fisher_value_b


    def fisher_parallel_weights(self, z, mask40, model, almE_fid, almB_fid, Elfid_e, Elfid_b, ell_flat, clee_ang, clbb_ang, clee_flat, clbb_flat, clee_bin, clbb_bin, cl_pert_prop_e, cl_pert_prop_b, c, proj, noiseQ, noiseU, valid_index, psi, mask2d):

        almE_pert = self.alm_perturbation(almE_fid, clee_ang, cl_pert_prop_e)
        almB_pert = self.alm_perturbation(almB_fid, clbb_ang, cl_pert_prop_b)

        data_pert = self.generate_pln(z, almE_pert, almB_pert, mask40, noiseQ, noiseU, psi, valid_index, proj, all = False, mask2d=mask2d)

        Elpert_e, Elpert_b = self.fiducial_weights(data_pert, model, ell_flat, clee_flat, clbb_flat, clee_bin, clbb_bin, all=False) # (nbins, n_modes)
        fisher_value_e, fisher_value_b = self.fisher(Elpert_e, Elpert_b, Elfid_e, Elfid_b, c, clee_bin, clbb_bin)

        return fisher_value_e, fisher_value_b
    

















 
            
















