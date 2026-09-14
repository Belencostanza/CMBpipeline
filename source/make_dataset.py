import sys, platform, os
import numpy as np
#import matplotlib.pyplot as plt
import scipy

import camb
from camb import model, initialpower

import healpy as hp

import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, ModuleList, LayerNorm
import torch.nn.functional as F

from healpy.projector import GnomonicProj
from scipy.stats import binned_statistic_2d
from scipy.interpolate import griddata
from scipy.interpolate import interpn
from scipy.linalg import orthogonal_procrustes
from scipy.ndimage import gaussian_filter

import utilities
import projections as pj

import h5py

class make_dataset:

    """
    Dataset generator for CMB polarization maps

    Creates training, validation and test datasets with CMB Q/U maps, 
    noise, and projection transformations

    """

    # Constants
    NUM_SPECTRUM_COMPONENTS = 6  # (cltt, clee, clbb, clte, cltb, ceb)
    
    #DEFAULT_NOISE_LEVEL = 5e-7

    #nlev= 2.1 uk.arcmin, ((nlev/60)*(np.pi/180))**2
    #DEFAULT_NOISE_LEVEL = 3.7315633923871807e-07
    
    #NOISE_PIX_PLANE = 0.093
    #NOISE_PIX_PLANE = 0.0935018906930219

    def __init__(
        self, 
        nsims_train: int, 
        nsims_valid: int, 
        nsims_test: int, 
        smooth: bool,
        apo: bool,
        nbins: int = 512, 
        nside: int = 512, 
        fwhm: float = 23, 
        DEFAULT_NOISE_LEVEL: float = 5e-7
    ):

        """    
        Initialize dataset generator.
        
        Args:
            nsims_train: Number of training simulations
            nsims_valid: Number of validation simulations
            nsims_test: Number of test simulations
            smooth: Whether to apply beam smoothing
            nside: HEALPix resolution parameter
            fwhm: Full width at half maximum in arcminutes
        """
    
        self.nsims_train = nsims_train
        self.nsims_valid = nsims_valid
        self.nsims_test = nsims_test
        self.smooth = smooth
        self.nside = nside
        self.nbins = nbins
        self.apo = apo
        self.fwhm = np.radians(fwhm/60) # from arcmin to rad
        self.DEFAULT_NOISE_LEVEL = DEFAULT_NOISE_LEVEL

    def get_CMB(self, beam: bool):

        """
        Generate one CMB E and B mode realization.
        
        Args:
            beam: Whether to apply beam smoothing
            
        Returns:
            Tuple of (mapE, mapB) or (mapE_smooth, mapB_smooth)
        """
        cltt,clee,clbb = utilities.signal_spectrum(r=0.034)
        mapE,_ = utilities.make_scalar_maps(clee, self.nside)
        mapB,_ = utilities.make_scalar_maps(clbb, self.nside)

        if beam == True: 
            mapE_smooth = hp.smoothing(mapE, self.fwhm)
            mapB_smooth = hp.smoothing(mapB, self.fwhm)

            return mapE_smooth, mapB_smooth
        else: 
            return mapE, mapB

    def _create_spectrum_array(self, cltt: np.ndarray, clee: np.ndarray, clbb: np.ndarray) -> np.ndarray:
        """
        Create spectrum array for QU maps generation.
        
        Args:
            cltt: TT power spectrum
            clee: EE power spectrum
            clbb: BB power spectrum
            
        Returns:
            Spectrum array with shape (6, lmax)
        """
        lmax = len(cltt)
        cls = np.zeros((self.NUM_SPECTRUM_COMPONENTS, lmax))
        cls[0] = cltt  # TT
        cls[1] = clee  # EE
        cls[2] = clbb  # BB
        # clte, cltb, ceb remain zero

        return cls

    def get_QUmaps(self, beam: bool):

        """
        Generate CMB T, Q, U maps.
        
        Args:
            beam: Whether to apply beam smoothing
            
        Returns:
            Tuple of (mapT, mapQ, mapU) or (mapT, mapQ_smooth, mapU_smooth)
        """

        cltt,clee,clbb = utilities.signal_spectrum(r=0.034)
        cls = self._create_spectrum_array(cltt, clee, clbb)

        qumaps,_ = utilities.make_maps(cls, self.nside)

        tmap = qumaps[0]
        qmap = qumaps[1]
        umap = qumaps[2]

        if beam == True: 
            mapQ_smooth = hp.smoothing(qmap, self.fwhm)
            mapU_smooth = hp.smoothing(umap, self.fwhm)
            return tmap, mapQ_smooth, mapU_smooth
        else:
            return tmap, qmap, umap

    def get_QUmaps_cl(self, cls: np.ndarray, beam: bool, cmb_seed=None):

        """
        Generate CMB T, Q, U maps.
        
        Args:
            beam: Whether to apply beam smoothing
            
        Returns:
            Tuple of (mapT, mapQ, mapU) or (mapT, mapQ_smooth, mapU_smooth)
        """

        qumaps,_ = utilities.make_maps(cls, self.nside, seed=cmb_seed)

        tmap = qumaps[0]
        qmap = qumaps[1]
        umap = qumaps[2]

        if beam == True:
            mapQ_smooth = hp.smoothing(qmap, self.fwhm)
            mapU_smooth = hp.smoothing(umap, self.fwhm)
            return tmap, mapQ_smooth, mapU_smooth
        else:
            return tmap, qmap, umap

    def get_homogenous_noise(self):

        """
        Get homogeneous noise power spectrum.
        
        Returns:
            Noise power spectrum array
        """
        cltt,_,_ = utilities.signal_spectrum()
        # cursor version
        nl = np.full(len(cltt), self.DEFAULT_NOISE_LEVEL)
        # old version
        #nl = np.zeros((len(cltt)))
        #nl[:] = 5e-7 #1e-2 # 2e-2

        return nl

    def get_inhomogenous_noise(self, mask_type, mask, inho_type=None, nhits=None, vec_center=None, theta0=None, lonc=None, latc=None):

        dx_sph = hp.nside2resol(self.nside)
        noise_pix_sph = self.DEFAULT_NOISE_LEVEL/dx_sph**2

        if mask_type == "circ":

            if inho_type == "elliptical":

                Nhits_masked = utilities.hits_model_elliptical_sphere(nside=self.nside,
                    lonc=lonc,
                    latc=latc,
                    radius_deg=40,
                    sigma_u_deg=30,
                    sigma_v_deg=15,
                    angle_deg=-25)

            else: # easiest hits model
                Nhits_masked, _ = utilities.hits_model(self.nside, vec_center, theta0, mask)
                
            variance_map = utilities.nhits_to_sigma2(noise_pix_sph, Nhits_masked, mask)
            
            return Nhits_masked, variance_map
        
        elif mask_type == "rect": # SO Hits
            
            variance_map = utilities.nhits_to_sigma2(noise_pix_sph, nhits, mask)
            Nhits_masked = nhits
            
            return Nhits_masked, variance_map

    def read_inhomogenous_noise(self, mask_type, inho_type=None, filename=None):

        """
        Read the 2D variance map of the projected inhomogeneous noise
        (generated by make_variance_map.py).

        filename: explicit path (config.dict -> cfg["inho2d_path"]). When None the
        legacy file names below are used, relative to the working directory.
        """
        if filename is not None:
            return np.load(filename)

        #print(mask_type)
        if mask_type == "circ":

            if inho_type == "elliptical":
                var_maps = np.load('var_map_sims100_elliptical.npy')
            else: 
                var_maps = np.load('var_map_sims100.npy')

        elif mask_type == "rect":
            if inho_type == "freq145":
                var_maps = np.load('var_map_sims100_SO.npy') # (512, 512)
            elif inho_type == "freq93":
                var_maps = np.load('var_map_sims100_SO_93GHz.npy') # (512, 512)
            elif inho_type == "inho2":
                #var_maps = np.load('var_map_sims100_SO_inho2.npy') # (512, 512)
                var_maps = np.load('var_map_sims100_SO_inho2_res_arcmin.npy')  #(416, 704)

        return var_maps


    def get_QUnoise(self, nl: np.ndarray):# -> tuple[np.ndarray, np.ndarray]:
        
        """
        Generate Q and U noise maps.
        
        Args:
            nl: Noise power spectrum
            
        Returns:
            Tuple of (qnoise, unoise) maps
        """
        nls = self._create_spectrum_array(nl, nl, nl)
        qumaps, _ = utilities.make_maps(nls, self.nside)

        qnoise = qumaps[1]
        unoise = qumaps[2]

        return qnoise, unoise

    def get_noise(self, nl: np.ndarray):
        
        """
        Generate scalar noise map.
        
        Args:
            nl: Noise power spectrum
            
        Returns:
            Noise map
        """
        noise_map,_ = utilities.make_scalar_maps(nl, self.nside)

        return noise_map

    def get_inho_noise(self, variance_map, mask, rng=None):
        
        """
        Generate scalar noise map.
        
        Args:
            variance_map: variance_map in the sphere
            mask: mask in the sphere
            
        Returns:
            Noise map
        """

        noise_map = utilities.sigma2_to_map(variance_map, mask, rng)

        return noise_map

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

    def make_inho_maps_pln(self, variance_map, proj, valid_index, mask, z, psi, mask_type, mask2d = None):

        mapT, mapQ, mapU = self.get_QUmaps(beam=self.smooth)
        noiseQ = self.get_inho_noise(variance_map, mask) 
        noiseU = self.get_inho_noise(variance_map, mask)

        dataQ = mask * (mapQ + noiseQ)
        dataU = mask * (mapU + noiseU)

        Q_obs_rot, U_obs_rot = self.make_rot(
            dataQ[valid_index],
            dataU[valid_index],
            psi)

        if mask_type == "rect":

            #gridQ_data = proj.grid_bins_mask_margin(z, Q_obs_rot, mask2d)
            #gridU_data = proj.grid_bins_mask_margin(z, U_obs_rot, mask2d)
            gridQ_data = proj.grid_bins_mask_from_res_margin(z, Q_obs_rot, mask2d)
            gridU_data = proj.grid_bins_mask_from_res_margin(z, U_obs_rot, mask2d)

        elif mask_type == "circ":
            
            gridQ_data = proj.grid_bins_mask_margin(z, Q_obs_rot, mask2d)#, margin_deg=5.0)
            gridU_data = proj.grid_bins_mask_margin(z, U_obs_rot, mask2d)#, margin_deg=5.0)

            #mask2d_index = np.where(gridQ_data != 0)
            #mask2d = np.zeros((self.nbins,self.nbins))
            #mask2d[mask2d_index] = 1

        return gridQ_data, gridU_data, mask2d

    def make_maps_pln_binning(self, nl, proj, valid_index, mask, z, psi, mask2d = None):

        mapT, mapQ, mapU = self.get_QUmaps(beam=self.smooth)
        noiseQ, noiseU = self.get_QUnoise(nl)

        dataQ = mask * (mapQ)#+ noiseQ)
        dataU = mask * (mapU) #+ noiseU)

        Q_obs_rot, U_obs_rot = self.make_rot(
            dataQ[valid_index],
            dataU[valid_index],
            psi)

        gridQ_data  = proj.grid_bins_binning(z, Q_obs_rot, mask2d)
        gridU_data  = proj.grid_bins_binning(z, U_obs_rot, mask2d)

        #mask2d_index = np.where(gridQ_data != 0)
        #mask2d = np.zeros((self.nbins,self.nbins))
        #mask2d[mask2d_index] = 1

        return gridQ_data, gridU_data

    def make_maps_pln(self, nl, proj, valid_index, mask, z, psi, mask2d = None, method='cubic'):

        mapT, mapQ, mapU = self.get_QUmaps(beam=self.smooth)
        noiseQ, noiseU = self.get_QUnoise(nl) 

        dataQ = mask * (mapQ + noiseQ)
        dataU = mask * (mapU + noiseU)

        Q_obs_rot, U_obs_rot = self.make_rot(
            dataQ[valid_index],
            dataU[valid_index],
            psi)

        gridQ_data  = proj.grid_bins_mask_margin_deg(z, Q_obs_rot, mask2d, method, margin_deg = 5.0)
        gridU_data  = proj.grid_bins_mask_margin_deg(z, U_obs_rot, mask2d, method, margin_deg = 5.0)

        #mask2d_index = np.where(gridQ_data != 0)
        #mask2d = np.zeros((self.nbins,self.nbins))
        #mask2d[mask2d_index] = 1

        return gridQ_data, gridU_data #, mask2d

    def make_maps_pln_cls_binning(self, cls, nl, proj, valid_index, mask, z, psi, mask2d = None):

        mapT, mapQ, mapU = self.get_QUmaps_cl(cls, beam=self.smooth)
        noiseQ, noiseU = self.get_QUnoise(nl)

        dataQ_nomask = mapQ + noiseQ
        dataU_nomask = mapU + noiseU

        dataQ = mask * (mapQ + noiseQ)
        dataU = mask * (mapU + noiseU)

        Q_obs_rot, U_obs_rot = self.make_rot(
            dataQ[valid_index],
            dataU[valid_index],
            psi)

        if mask2d is not None:
            gridQ_data = proj.grid_bins_binning(z, Q_obs_rot, mask2d)
            gridU_data = proj.grid_bins_binning(z, U_obs_rot, mask2d)
            mask = mask2d
        else:
            gridQ_data, _, _ = proj.grid_bins(z, Q_obs_rot)
            gridU_data, _, _ = proj.grid_bins(z, U_obs_rot)

            mask_index = np.where(gridQ_data != 0)
            mask = np.zeros((self.npixels,self.npixels))
            mask[mask_index] = 1

        return gridQ_data, gridU_data, mask2d, dataQ_nomask, dataU_nomask

    def make_maps_pln_from_sph(self, mapQ, mapU, proj, valid_index, mask, z, psi, mask2d=None): 

        dataQ = mask * (mapQ)
        dataU = mask * (mapU)
        
        Q_obs_rot, U_obs_rot = self.make_rot(
                dataQ[valid_index],
                dataU[valid_index],
                psi)

        if mask2d is not None:
            gridQ_data = proj.grid_bins_mask_margin(z, Q_obs_rot, mask2d)
            gridU_data = proj.grid_bins_mask_margin(z, U_obs_rot, mask2d)
            mask = mask2d

        return gridQ_data, gridU_data, mask2d

    def make_maps_inho_pln_from_sph(self, mapQ, mapU, proj, valid_index, mask, z, psi, mask_type, mask2d=None): 

        dataQ = mask * (mapQ)
        dataU = mask * (mapU)
        
        Q_obs_rot, U_obs_rot = self.make_rot(
                dataQ[valid_index],
                dataU[valid_index],
                psi)

        if mask_type == "rect":

            #gridQ_data = proj.grid_bins_mask_margin(z, Q_obs_rot, mask2d)
            #gridU_data = proj.grid_bins_mask_margin(z, U_obs_rot, mask2d)
            gridQ_data = proj.grid_bins_mask_from_res_margin(z, Q_obs_rot, mask2d)
            gridU_data = proj.grid_bins_mask_from_res_margin(z, U_obs_rot, mask2d)

        elif mask_type == "circ":

            gridQ_data = proj.grid_bins_mask_margin(z, Q_obs_rot, mask2d)#, margin_deg=5.0)
            gridU_data = proj.grid_bins_mask_margin(z, U_obs_rot, mask2d)#, margin_deg=5.0)

        return gridQ_data, gridU_data, mask2d

    def make_maps_inho_pln_cls(self, cls, variance_map, proj, valid_index, mask, z, psi, mask_type, mask2d = None, cmb_seed= None, noise_seed = None):

        mapT, mapQ, mapU = self.get_QUmaps_cl(cls, beam=self.smooth, cmb_seed=cmb_seed)

        rng_noise = np.random.default_rng(noise_seed)

        noiseQ = self.get_inho_noise(variance_map, mask, rng=rng_noise) 
        noiseU = self.get_inho_noise(variance_map, mask, rng=rng_noise)

        dataQ_nomask = mapQ + noiseQ
        dataU_nomask = mapU + noiseU

        dataQ = mask * (mapQ + noiseQ)
        dataU = mask * (mapU + noiseU)

        Q_obs_rot, U_obs_rot = self.make_rot(
            dataQ[valid_index],
            dataU[valid_index],
            psi)

        if mask_type == "rect":

            #gridQ_data = proj.grid_bins_mask_margin(z, Q_obs_rot, mask2d)
            #gridU_data = proj.grid_bins_mask_margin(z, U_obs_rot, mask2d)
            gridQ_data = proj.grid_bins_mask_from_res_margin(z, Q_obs_rot, mask2d)
            gridU_data = proj.grid_bins_mask_from_res_margin(z, U_obs_rot, mask2d)

        elif mask_type == "circ":

            gridQ_data = proj.grid_bins_mask_margin(z, Q_obs_rot, mask2d)#, margin_deg=5.0)
            gridU_data = proj.grid_bins_mask_margin(z, U_obs_rot, mask2d)#, margin_deg=5.0)

        return gridQ_data, gridU_data, mask2d, dataQ_nomask, dataU_nomask

    def make_maps_pln_cls(self, cls, nl, proj, valid_index, mask, z, psi, mask_type, mask2d = None):

        mapT, mapQ, mapU = self.get_QUmaps_cl(cls, beam=self.smooth)
        noiseQ, noiseU = self.get_QUnoise(nl)

        dataQ_nomask = mapQ + noiseQ
        dataU_nomask = mapU + noiseU

        dataQ = mask * (mapQ + noiseQ)
        dataU = mask * (mapU + noiseU)

        Q_obs_rot, U_obs_rot = self.make_rot(
            dataQ[valid_index],
            dataU[valid_index],
            psi)

        if mask_type == "rect":

            #gridQ_data = proj.grid_bins_mask_margin(z, Q_obs_rot, mask2d)
            #gridU_data = proj.grid_bins_mask_margin(z, U_obs_rot, mask2d)
            gridQ_data = proj.grid_bins_mask_from_res_margin(z, Q_obs_rot, mask2d)
            gridU_data = proj.grid_bins_mask_from_res_margin(z, U_obs_rot, mask2d)

        elif mask_type == "circ":

            gridQ_data = proj.grid_bins_mask_margin(z, Q_obs_rot, mask2d)#, margin_deg=5.0)
            gridU_data = proj.grid_bins_mask_margin(z, U_obs_rot, mask2d)#, margin_deg=5.0)

        return gridQ_data, gridU_data, mask2d, dataQ_nomask, dataU_nomask


    def make_train_dataset(
        self,
        radius_deg: float,
        filename_train: str,
        filename_valid: str,
        mask_type: str = "rect", 
        noise_type: str = "inho",
        inho_type: str = None, 
        model: bool = True,
        method: str = "cubic",
        inho2d_file: str = None,
        apo_mask_file: str = None,
        fact: int = 8,
        qubic_file: str = None,
        so_hits_file: str = None,
        **mask_kwargs
        ):

        """
        Generate and save training and validation datasets.
        
        Args:
            radius_deg: Radius of sky region in degrees
            nbins: Number of bins for grid projection
            filename_train: Filename to save training data
            filename_valid: Filename to save validation data
            inho2d_file: 2D variance map of the projected noise (cfg["inho2d_path"]);
                         None -> legacy file names in the working directory
            apo_mask_file: apodized sky mask used when self.apo is True (cfg["apo_mask_path"])
            fact: margin factor of proj_2d (cfg["fact"])
            qubic_file, so_hits_file: mask input files (cfg["qubic_path"], cfg["so_hits_path"]);
                         None -> taken from config.dict by projections.sph_mask
        """

        nsims = self.nsims_train + self.nsims_valid
        nl = self.get_homogenous_noise()
        
        sky_region = pj.sph_mask(nside=self.nside, radius=radius_deg,
                                 qubic_file=qubic_file, so_hits_file=so_hits_file)

        if mask_type == "rect":
            mask, nhits, valid_index, vecs, vec_center, theta_idx, phi_idx, lonc, latc = \
            sky_region.make_SO_mask(**mask_kwargs)

        elif mask_type == "circ":
            mask, valid_index, vecs, vec_center, theta_idx, phi_idx, lonc, latc = \
            sky_region.make_mask(**mask_kwargs)

        else:
            raise ValueError("mask_type debe ser 'rect' o 'circ'")

        if self.apo: 
            mask = np.load(apo_mask_file if apo_mask_file is not None else 'guassian_apodization0.8_mask.npy')
        
        theta0, phi0 = hp.vec2ang(vec_center)
        theta, phi = hp.vec2ang(vecs)
        
        proj = pj.proj_2d(lonc, latc, theta_idx, phi_idx, vec_center, nbins=self.nbins, fact=fact)
        
        z_eq, z_tan, z_str = proj.proj_conventions(vecs, vec_center)
        
        #dx, dy, dx_avg = utilities.calculate_res(z_eq, self.nbins, self.nbins)

        
        if noise_type == "inho":


            if mask_type == "rect":

                Nhits_masked, variance_map = self.get_inhomogenous_noise(mask_type, mask, inho_type, nhits)
                #mask2d_rect = proj.define_region_mask_nbins_margins(z_eq, mask, self.nside)
                mask2d_rect, nbins_x, nbins_y = proj.define_region_mask_from_res_margins(z_eq, mask, self.nside) # shape (ny, nx)

                if model:
                    Nhits_piece = Nhits_masked[valid_index]
                    sigma2_pln = utilities.nhitsproj_to_sigma2(self.NOISE_PIX_PLANE, nhits_pln, mask2d_rect)
                else: 
                    var_sims_pln = self.read_inhomogenous_noise(mask_type, inho_type, filename=inho2d_file)
                    sigma2_pln = var_sims_pln#['arr_1']

                X_train = torch.zeros((self.nsims_train, 4, nbins_y, nbins_x), dtype=torch.float32)
                X_valid = torch.zeros((self.nsims_valid, 4, nbins_y, nbins_x), dtype=torch.float32)

            elif mask_type == "circ":

                Nhits_masked, variance_map = self.get_inhomogenous_noise(mask_type, mask, inho_type, nhits=None, vec_center=vec_center, theta0=theta0, lonc=lonc, latc=latc)
                mask2d = proj.define_region_mask_nbins_margins(z_eq, mask, self.nside)

                if model:
                    Nhits_piece = Nhits_masked[valid_index]
                    nhits_pln = proj.grid_bins_mask_margin(z_eq, Nhits_piece, mask2d)
                    sigma2_pln = utilities.nhitsproj_to_sigma2(self.NOISE_PIX_PLANE, nhits_pln, mask2d)
                else: 
                    var_sims_pln = self.read_inhomogenous_noise(mask_type, inho_type, filename=inho2d_file)
                    sigma2_pln = var_sims_pln#['arr_0']

                X_train = torch.zeros((self.nsims_train, 4, self.nbins, self.nbins), dtype=torch.float32)
                X_valid = torch.zeros((self.nsims_valid, 4, self.nbins, self.nbins), dtype=torch.float32)

        elif noise_type == "ho":

            mask2d = proj.define_region_mask_nbins_margins(z_eq, mask, self.nside)#, margin_deg=5.0)

            X_train = torch.zeros((self.nsims_train, 3, self.nbins, self.nbins), dtype=torch.float32)
            X_valid = torch.zeros((self.nsims_valid, 3, self.nbins, self.nbins), dtype=torch.float32)

        phi_wrapped = (phi + np.pi) % (2*np.pi) - np.pi
        phi0_wrapped = (phi0 + np.pi) % (2*np.pi) - np.pi

        rt = pj.rotate_geo(theta, phi_wrapped, theta0, phi0_wrapped)
        psi = rt.psi_Q_to_P()

        for map_id in range(nsims):

            print('map', map_id)

            if noise_type == "inho":

                if mask_type == "rect":

                    gridQ_data, gridU_data, mask2d = self.make_inho_maps_pln(variance_map, proj, valid_index, mask, z_eq, psi, mask_type, mask2d = mask2d_rect)

                elif mask_type == "circ":

                    gridQ_data, gridU_data, mask2d = self.make_inho_maps_pln(variance_map, proj, valid_index, mask, z_eq, psi, mask_type, mask2d = mask2d)

                if map_id < self.nsims_train:

                    X_train[map_id, 0, :, :] = torch.from_numpy(gridQ_data)
                    X_train[map_id, 1, :, :] = torch.from_numpy(gridU_data)
                    X_train[map_id, 2, :, :] = torch.from_numpy(mask2d)
                    X_train[map_id, 3, :, :] = torch.from_numpy(sigma2_pln)

                if map_id>= self.nsims_train and map_id<(self.nsims_train+ self.nsims_valid):

                    map_id_valid = map_id - self.nsims_train
                    X_valid[map_id_valid, 0, :, :] = torch.from_numpy(gridQ_data)
                    X_valid[map_id_valid, 1, :, :] = torch.from_numpy(gridU_data)
                    X_valid[map_id_valid, 2, :, :] = torch.from_numpy(mask2d)
                    X_valid[map_id_valid, 3, :, :] = torch.from_numpy(sigma2_pln)


            elif noise_type == "ho":
                gridQ_data, gridU_data = self.make_maps_pln(nl, proj, valid_index, mask, z_eq, psi, mask2d, method)

                if map_id < self.nsims_train:

                    X_train[map_id, 0, :, :] = torch.from_numpy(gridQ_data)
                    X_train[map_id, 1, :, :] = torch.from_numpy(gridU_data)
                    X_train[map_id, 2, :, :] = torch.from_numpy(mask2d)

                if map_id>= self.nsims_train and map_id<(self.nsims_train+ self.nsims_valid):

                    map_id_valid = map_id - self.nsims_train
                    X_valid[map_id_valid, 0, :, :] = torch.from_numpy(gridQ_data)
                    X_valid[map_id_valid, 1, :, :] = torch.from_numpy(gridU_data)
                    X_valid[map_id_valid, 2, :, :] = torch.from_numpy(mask2d)

                
        torch.save(X_train, filename_train)
        torch.save(X_valid, filename_valid)


    def make_spectra_dataset(
        self,
        radius_deg: float,
        filename_train: str,
        filename_valid: str
        ):

        """
        Generate and save training and validation datasets.
        
        Args:
            radius_deg: Radius of sky region in degrees
            nbins: Number of bins for grid projection
            filename_train: Filename to save training data
            filename_valid: Filename to save validation data
        """

        nsims = self.nsims_train + self.nsims_valid
        nl = self.get_homogenous_noise()
        cltt,clee,clbb = utilities.signal_spectrum()

        sky_region = pj.sph_mask(nside=self.nside, radius=radius_deg)
        mask40, valid_index, vecs, vec_center, theta_idx, phi_idx, lonc, latc = sky_region.make_mask()
        theta0, phi0 = hp.vec2ang(vec_center)
        theta, phi = hp.vec2ang(vecs)
        proj = pj.proj_2d(lonc, latc, theta_idx, phi_idx, vec_center, nbins=self.nbins)
        
        z_eq, z_tan, z_str = proj.proj_conventions(vecs, vec_center)
        
        dx = utilities.calculate_res(z_eq, self.nbins)

        phi_wrapped = (phi + np.pi) % (2*np.pi) - np.pi
        phi0_wrapped = (phi0 + np.pi) % (2*np.pi) - np.pi

        rt = pj.rotate_geo(theta, phi_wrapped, theta0, phi0_wrapped)
        psi = rt.psi_Q_to_P()

        qnoise_fixed, unoise_fixed = self.get_QUnoise(nl)

        X_train = torch.zeros((self.nsims_train, 3, self.nbins, self.nbins), dtype=torch.float32)

        almE_fid_all = []
        almB_fid_all = []

        for map_id in range(nsims):

            print('map', map_id)

            if map_id < self.nsims_train:
                
                gridQ_data, gridU_data, mask = self.make_maps_pln(nl, proj, valid_index, mask40, z_eq, psi)

                X_train[map_id, 0, :, :] = torch.from_numpy(gridQ_data)
                X_train[map_id, 1, :, :] = torch.from_numpy(gridU_data)
                X_train[map_id, 2, :, :] = torch.from_numpy(mask)

            if map_id>= self.nsims_train and map_id<(self.nsims_train+ self.nsims_valid):

                almE_fid = hp.synalm(clee, lmax=1600)
                almB_fid = hp.synalm(clbb, lmax=1600)

                almE_fid_all.append(almE_fid)
                almB_fid_all.append(almB_fid)

        almE_fid_all = np.stack(almE_fid_all)  # shape = (Nmaps, Nalm)
        almB_fid_all = np.stack(almB_fid_all)
        print(almE_fid_all.shape)

        torch.save(X_train, filename_train)

        with h5py.File(filename_valid, "w") as f:
            f.create_dataset("almE_fid", data=almE_fid_all)
            f.create_dataset("almB_fid", data=almB_fid_all)
            f.create_dataset("noiseQ", data=np.array(qnoise_fixed))
            f.create_dataset("noiseU", data=np.array(unoise_fixed))
            f.create_dataset("psi", data=np.array(psi))














