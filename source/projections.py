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
import pickle
from astropy.io import fits

from scipy.interpolate import RegularGridInterpolator

# --------------------------------------------------------------------------- #
# Sky-mask input files.
# No path is hard-coded here: sph_mask takes `qubic_file` / `so_hits_file`
# explicitly, and when they are not given they are read from config.dict
# ("mask_files" section) through config_loader.load_config().
# --------------------------------------------------------------------------- #

def default_mask_files():
    """(qubic_path, so_hits_path) from config.dict (see config_loader.find_config)."""
    import os
    from config_loader import load_config
    cfg = load_config(create_dirs=False)
    for key in ("qubic_path", "so_hits_path"):
        if not os.path.isfile(cfg[key]):
            raise FileNotFoundError(
                f"{key} = {cfg[key]} (from {cfg['config_file']}) does not exist. "
                "Fix 'mask_files' in config.dict, point WF_CONFIG to a local config "
                "(e.g. config_local.dict) or pass qubic_file/so_hits_file to sph_mask.")
    return cfg["qubic_path"], cfg["so_hits_path"]

class sph_mask: 

    def __init__(self, nside, radius=None, side_deg=None, center=False, move=True,
                 qubic_file=None, so_hits_file=None): 

        """ Class to generate the region of the sky

        Params: 

        nside = nside of the healpy map
        radius = radius in deg of the circular region
        qubic_file = QUBIC coverage .pkl (centre of the circular mask); default: config.dict
        so_hits_file = SO hits .fits (rectangular/SO-like mask); default: config.dict

        """  

        self.nside = nside
        self.radius = radius
        self.side_deg = side_deg
        self.center = center
        self.move = move
        self.qubic_file = qubic_file
        self.so_hits_file = so_hits_file

    def _mask_file(self, which):
        """Resolve the qubic / so_hits input file (explicit argument first, then config.dict)."""
        value = self.qubic_file if which == "qubic" else self.so_hits_file
        if value is None:
            qubic_path, so_hits_path = default_mask_files()
            self.qubic_file, self.so_hits_file = qubic_path, so_hits_path
            value = qubic_path if which == "qubic" else so_hits_path
        return value

    def read_qubic(self):

        """
        Function that read qubic patch if you want to use QUBIC mask - like
        """

        maps_path = self._mask_file("qubic")
        with open(maps_path, 'rb') as f:
            data = pickle.load(f)

        if self.move: 
            lonc = data['center'][0] - 20
            latc = data['center'][1] + 30
        else:
            lonc = data['center'][0]
            latc = data['center'][1]
        coverage = data["coverage"]

        return lonc, latc, coverage

    def read_SO(self):

        """
        Function that read hit maps of Simons Observatory

        """
        maps_path = self._mask_file("so_hits")
        with fits.open(maps_path) as hdul:
            data = hdul[1].data
        hits = data['T']

        return hits


    def make_ra_dec_mask(self, ra_min, ra_max, dec_min, dec_max):

        """
        Function that given certain equatorial coordinates make a rectangular mask (SO - like)
        """

        npix = hp.nside2npix(self.nside)

        ra_corners = [ra_min, ra_max, ra_max, ra_min]
        dec_corners = [dec_min, dec_min, dec_max, dec_max]

        vec_corners = hp.ang2vec(ra_corners, dec_corners, lonlat=True)
        pixels_in_rect = hp.query_polygon(self.nside, vec_corners)

        valid_index = pixels_in_rect
        mask = np.zeros(npix)
        mask[valid_index] = 1.0

        return mask, valid_index, vec_corners


    def make_polygon_mask(self, lonc=None, latc=None):

        """
        Function make a polygon/square mask from a given center
        The cornes are defined from the center (center - d, center + d)

        """

        if self.center == False:
            lonc, latc, coverage = self.read_qubic()
        else: 
            lonc = lonc
            latc = latc
        
        npix = hp.nside2npix(nside=self.nside)
        vec_center = hp.ang2vec(lonc, latc, lonlat=True)

        d = self.side_deg / 2.0
        corners_lon = [lonc - d, lonc + d, lonc + d, lonc - d]
        corners_lat = [latc - d, latc - d, latc + d, latc + d]

        vec_corners = hp.ang2vec(corners_lon, corners_lat, lonlat=True)
        pixels_in_square = hp.query_polygon(self.nside, vec_corners)

        mask = np.zeros(npix)
        mask[pixels_in_square] = 1.0

        valid_index = pixels_in_square
        theta_idx, phi_idx = hp.pix2ang(self.nside, valid_index)
        vecs = hp.ang2vec(theta_idx, phi_idx)

        return mask, valid_index, vecs, vec_center, theta_idx, phi_idx, lonc, latc


    def make_square_mask(self, lonc=None, latc=None):

        if self.center == False:
            lonc, latc, coverage = self.read_qubic()
            
        else: 
            lonc = lonc
            latc = latc

        npix = hp.nside2npix(self.nside)
        side_rad = np.radians(self.side_deg)
        vec_center = hp.ang2vec(lonc, latc, lonlat=True)

        d = self.side_deg / 2.0
        lon_min, lon_max = lonc - d, lonc + d
        lat_min, lat_max = latc - d, latc + d
        
        theta, phi = hp.pix2ang(self.nside, np.arange(npix))
        lon = np.degrees(phi)
        lat = 90 - np.degrees(theta)  # healpy usa colatitud

        if lon_min < 0:
            lon_min += 360
            lon[lon < 0] += 360
        if lon_max > 360:
            lon_max -= 360
            lon[lon > 360] -= 360

        inside = (
            (lon >= lon_min) & (lon <= lon_max) &
            (lat >= lat_min) & (lat <= lat_max)
            )

        mask = np.zeros(npix)
        mask[inside] = 1.0

        valid_index = np.where(mask != 0)[0]
        theta_idx, phi_idx = hp.pix2ang(self.nside, valid_index)
        vecs = hp.ang2vec(theta_idx, phi_idx)

        return mask, valid_index, vecs, vec_center, theta_idx, phi_idx, lonc, latc

    def make_rec_mask(self, width_ra_deg=None, height_dec_deg=None, lonc=None, latc=None):

        npix = hp.nside2npix(self.nside)
        vec_center = hp.ang2vec(lonc, latc, lonlat=True)

        lonc = (lonc + 180) % 360 - 180 # cambiar de [0,360) a [-180, 180)
        dra = width_ra_deg / 2.0
        ddec = height_dec_deg / 2.0

        lon_min, lon_max = lonc - dra, lonc + dra
        lat_min, lat_max = latc - ddec, latc + ddec

        theta, phi = hp.pix2ang(self.nside, np.arange(npix))

        lon = np.degrees(phi)
        lon = (lon + 180) % 360 - 180  # cambiar de [0,360) a [-180, 180)
        lat = 90 - np.degrees(theta)  # healpy usa colatitud

        inside = (
            (lon >= lon_min) & (lon <= lon_max) &
            (lat >= lat_min) & (lat <= lat_max)
            )

        mask = np.zeros(npix)
        mask[inside] = 1.0

        valid_index = np.where(mask != 0)[0]
        theta_idx, phi_idx = hp.pix2ang(self.nside, valid_index)
        vecs = hp.ang2vec(theta_idx, phi_idx)

        return mask, valid_index, vecs, vec_center, theta_idx, phi_idx, lonc, latc

    def make_mask(self):


        lonc, latc, coverage = self.read_qubic()
        npix = hp.nside2npix(self.nside)
        radius_rad = np.radians(self.radius)

        vec_center = hp.ang2vec(lonc, latc, lonlat=True)

        pixels_in_disc = hp.query_disc(nside=self.nside, vec=vec_center, radius=radius_rad)

        mask = np.zeros(npix)
        mask[pixels_in_disc] = 1.0

        valid_index = np.array(np.where(mask != 0)[0])
        theta_idx, phi_idx =hp.pix2ang(self.nside, valid_index) #colalitude and longitude from healpy (at the center of the sphere)
        vecs = hp.ang2vec(theta_idx, phi_idx)

        return mask, valid_index, vecs, vec_center, theta_idx, phi_idx, lonc, latc

    def make_SO_mask(self):

        hits = self.read_SO()
        hits = hits.flatten()
        vec_center = hp.pix2vec(self.nside, np.argmax(hits))
        vecs_all = np.array(hp.pix2vec(self.nside, np.arange(hp.nside2npix(self.nside))))

        cosang = np.dot(vec_center, vecs_all)
        theta = np.arccos(cosang)

        # SO mask
        theta_max = 100  # ver si dejo fijo o parametro
        thr = 0.25
        mask = (theta < np.radians(theta_max)) & (hits > thr)

        # nhits SO map
        nhits = mask*hits

        lonc, latc = hp.pix2ang(self.nside, np.argmax(hits), lonlat=True)
        valid_index = np.where(mask != 0)[0]
        theta_idx, phi_idx = hp.pix2ang(self.nside, valid_index)
        vecs = hp.ang2vec(theta_idx, phi_idx)
        vec_center = hp.ang2vec(lonc, latc, lonlat=True)

        return mask, nhits, valid_index, vecs, vec_center, theta_idx, phi_idx, lonc, latc



class proj_2d: 

    def __init__(self, lonc, latc, theta_idx, phi_idx, vec_center, nbins, reso_arcmin=9.6, fact=4):

        self.vec_center = vec_center
        self.lonc = lonc
        self.latc = latc
        self.theta_idx = theta_idx
        self.phi_idx = phi_idx
        self.nbins = nbins
        self.reso_arcmin = reso_arcmin
        self.fact = fact

    def polar_coord(self, vecs):

        cos_theta = np.clip(vecs @ self.vec_center, -1.0, 1.0)
        theta_new = np.arccos(cos_theta)

        z = self.vec_center/np.linalg.norm(self.vec_center) # (en realidad el centro ya es unitatio)
        u = np.cross(z, [0,0,1])
        u /= np.linalg.norm(u)
        v = np.cross(z,u)

        vecs_proj = vecs - theta_new[:,None]*self.vec_center[None,:]
        x_ = np.sum(vecs_proj * u[None, :], axis=1)
        y_ = np.sum(vecs_proj * v[None, :], axis=1)
        phi_proj = np.arctan2(y_, x_)

        return theta_new, phi_proj

    def proj_conventions(self, vecs, vec_center):
        # gnomonic and equidistant projection with conventions


        z = vec_center / np.linalg.norm(vec_center)
        k = np.array([0., 0., 1.])

        north = k - (k @ z) * z # proyeccion sobre el plano con normal z
        norm_n = np.linalg.norm(north)
        north = north / norm_n

        west = np.cross(z, north)  # producto vectorial de z con y 
        west /= np.linalg.norm(west)

        cos_theta = np.clip(vecs @ vec_center, -1.0, 1.0)
        theta_proj1 = np.arccos(cos_theta)

        vecs_proj = vecs - (vecs @ vec_center)[:, None] * vec_center[None, :]

        x_ = np.sum(vecs_proj * west[None, :], axis=1)
        y_ = np.sum(vecs_proj * north[None, :], axis=1)
        phi_proj1 = np.arctan2(x_,y_)

        r_eq = theta_proj1
        r_tan = np.tan(theta_proj1)
        r_str = 2*np.tan(theta_proj1/2)

        x_eq1 = r_eq * np.sin(phi_proj1)
        y_eq1 = r_eq * np.cos(phi_proj1)

        x_tan1 = r_tan * np.sin(phi_proj1)
        y_tan1 = r_tan * np.cos(phi_proj1)

        x_str1 = r_str * np.sin(phi_proj1)
        y_str1 = r_str * np.cos(phi_proj1)

        z_eq1 = np.stack([x_eq1, y_eq1], axis=1)
        z_tan1 = np.stack([x_tan1, y_tan1], axis=1)
        z_str1 = np.stack([x_str1, y_str1], axis=1)

        return z_eq1, z_tan1, z_str1


    def get_gnomonic(self): 

        lat = 90 - np.degrees(self.theta_idx)      # convertir colatitud (theta) a latitud
        lon = np.degrees(self.phi_idx)             # convertir phi a grados

        proj = GnomonicProj(rot=[self.lonc, self.latc])
        x_gnom, y_gnom = proj.ang2xy(lon, lat, lonlat=True)
        z_gnom = np.stack([x_gnom, y_gnom], axis=1)

        return z_gnom 

    def get_equidistant(self, vecs, z_ref = None, rotate = False): 

        theta_proj, phi_proj = self.polar_coord(vecs)
        x_plane = theta_proj * np.cos(phi_proj)
        y_plane = theta_proj * np.sin(phi_proj)

        z_eq = np.stack([x_plane, y_plane], axis=1)

        if rotate == True:
            R, _ = orthogonal_procrustes(z_eq, z_ref)
            z_eq_rot = z_eq @ R

            return z_eq_rot
        else: 
            return z_eq

    def get_NN_proj(self, z_ref, nn_file):

        # nn_file: .npy with the learned 2D embedding (formerly a hard-coded NN_path)
        z_NN = np.load(nn_file)
        R, _ = orthogonal_procrustes(z_NN, z_ref)
        z_NN_rot = z_NN @ R

        return z_NN_rot

    def get_cart(self, vecs):

        theta_proj, phi_proj = self.polar_coord(vecs)
        x_car = phi_proj
        y_car = theta_proj
        z_car = np.stack([x_car, y_car], axis=1)

        return z_car

    def inv_projection(self, z, projection='eq'): 

        x, y = z[:,0], z[:,1]

        if projection == 'eq':
            theta = np.sqrt(x**2 + y**2)
        elif projection == 'gnomonic': 
            theta = np.arctan(np.sqrt(x**2 + y**2))

        phi = np.arctan2(y, x)

        z = self.vec_center / np.linalg.norm(self.vec_center)
        u = np.cross(z, [0, 0, 1])
        if np.linalg.norm(u) < 1e-10:  # caso especial si vec_center ~ eje z
            u = np.cross(z, [0, 1, 0])
        u /= np.linalg.norm(u)
        v = np.cross(z, u)

        vecs_recon = (
            np.cos(theta)[:, None] * z[None, :]
            + np.sin(theta)[:, None] * (
            np.cos(phi)[:, None] * u[None, :] +
            np.sin(phi)[:, None] * v[None, :]
            ))

        return vecs_recon

    def define_region_mask_nbins_margins_deg(self, z, mask, nside, margin_deg):

        # define region mask in the grid

        x, y = z[:, 0], z[:, 1]

        margin = np.radians(margin_deg)

        x_max, x_min = x.max(), x.min()
        y_max, y_min = y.max(), y.min()

        #x_vals = np.linspace(x_min, x_max, self.nbins)
        #y_vals = np.linspace(y_min, y_max, self.nbins)

        x_vals = np.linspace(x_min - margin, x_max + margin, self.nbins)
        y_vals = np.linspace(y_min - margin, y_max + margin, self.nbins)   

        grid_x, grid_y = np.meshgrid(x_vals, y_vals)

        z_axis = self.vec_center/ np.linalg.norm(self.vec_center)
        k = np.array([0., 0., 1.])

        north = k - (k @ z_axis) * z_axis # proyeccion sobre el plano con normal z
        norm_n = np.linalg.norm(north)
        north = north / norm_n

        west = np.cross(z_axis, north)  # producto vectorial de z con y 
        west /= np.linalg.norm(west)

        r = np.sqrt(grid_x**2 + grid_y**2)
        phi = np.arctan2(grid_x, grid_y)
        theta = r

        vec = (
            np.cos(theta)[...,None]*self.vec_center +
            np.sin(theta)[...,None]*(
                np.cos(phi)[...,None]*north +
                np.sin(phi)[...,None]*west
            ))

        theta_sph = np.arccos(vec[...,2])
        phi_sph = np.arctan2(vec[...,1], vec[...,0])
        phi_sph = np.mod(phi_sph, 2*np.pi)
        mask2d = mask[hp.ang2pix(nside, theta_sph, phi_sph)]

        return mask2d

    def define_region_mask_nbins_margins(self, z, mask, nside):

        # define region mask in the grid

        x, y = z[:, 0], z[:, 1]

        x_max, x_min = x.max(), x.min()
        y_max, y_min = y.max(), y.min()

        #x_vals = np.linspace(x_min, x_max, self.nbins)
        #y_vals = np.linspace(y_min, y_max, self.nbins)

        dx = (x_max - x_min) / self.nbins
        dy = (y_max - y_min) / self.nbins
        margin = self.fact * max(dx, dy) 

        x_vals = np.linspace(x_min - margin, x_max + margin, self.nbins)
        y_vals = np.linspace(y_min - margin, y_max + margin, self.nbins)   

        grid_x, grid_y = np.meshgrid(x_vals, y_vals)

        z_axis = self.vec_center/ np.linalg.norm(self.vec_center)
        k = np.array([0., 0., 1.])

        north = k - (k @ z_axis) * z_axis # proyeccion sobre el plano con normal z
        norm_n = np.linalg.norm(north)
        north = north / norm_n

        west = np.cross(z_axis, north)  # producto vectorial de z con y 
        west /= np.linalg.norm(west)

        r = np.sqrt(grid_x**2 + grid_y**2)
        phi = np.arctan2(grid_x, grid_y)
        theta = r

        vec = (
            np.cos(theta)[...,None]*self.vec_center +
            np.sin(theta)[...,None]*(
                np.cos(phi)[...,None]*north +
                np.sin(phi)[...,None]*west
            ))

        theta_sph = np.arccos(vec[...,2])
        phi_sph = np.arctan2(vec[...,1], vec[...,0])
        phi_sph = np.mod(phi_sph, 2*np.pi)
        mask2d = mask[hp.ang2pix(nside, theta_sph, phi_sph)]

        return mask2d

    def define_region_mask_from_res_margins(self, z, mask, nside):

        x, y = z[:, 0], z[:, 1]

        x_max, x_min = x.max(), x.min()
        y_max, y_min = y.max(), y.min()

        Lx = x_max - x_min
        Ly = y_max - y_min

        pix = np.radians(self.reso_arcmin / 60.0)
        margin = self.fact * pix

        def _round_up(n, mult=32):
            return int(np.ceil(n / mult) * mult)

        nbins_x = _round_up(np.ceil((Lx + 2 * margin) / pix))
        nbins_y = _round_up(np.ceil((Ly + 2 * margin) / pix))

        # Number of bins adapted to each side -> square pixels, rectangular image
        #nbins_x = int(np.ceil((Lx + 2 * margin) / pix))
        #nbins_y = int(np.ceil((Ly + 2 * margin) / pix))

        x_vals = x_min - margin + pix * np.arange(nbins_x)
        y_vals = y_min - margin + pix * np.arange(nbins_y)

        grid_x, grid_y = np.meshgrid(x_vals, y_vals)   # shape (nbins_y, nbins_x)

        # ... rest identical ...
        z_axis = self.vec_center / np.linalg.norm(self.vec_center)        
        k = np.array([0., 0., 1.])
        north = k - (k @ z_axis) * z_axis
        north /= np.linalg.norm(north)
        west = np.cross(z_axis, north)
        west /= np.linalg.norm(west)


        r = np.sqrt(grid_x**2 + grid_y**2)
        phi = np.arctan2(grid_x, grid_y)
        theta = r
        vec = (np.cos(theta)[..., None] * self.vec_center +
            np.sin(theta)[..., None] * (np.cos(phi)[..., None] * north +
                np.sin(phi)[..., None] * west))

        theta_sph = np.arccos(vec[..., 2])
        phi_sph = np.mod(np.arctan2(vec[..., 1], vec[..., 0]), 2 * np.pi)
        mask2d = mask[hp.ang2pix(nside, theta_sph, phi_sph)]

        return mask2d, nbins_x, nbins_y

    def define_region_mask_nbins(self, z, mask, nside):

        # define region mask in the grid

        x, y = z[:, 0], z[:, 1]

        x_max, x_min = x.max(), x.min()
        y_max, y_min = y.max(), y.min()

        x_vals = np.linspace(x_min, x_max, self.nbins)
        y_vals = np.linspace(y_min, y_max, self.nbins)

        grid_x, grid_y = np.meshgrid(x_vals, y_vals)

        z_axis = self.vec_center/ np.linalg.norm(self.vec_center)
        k = np.array([0., 0., 1.])

        north = k - (k @ z_axis) * z_axis # proyeccion sobre el plano con normal z
        norm_n = np.linalg.norm(north)
        north = north / norm_n

        west = np.cross(z_axis, north)  # producto vectorial de z con y 
        west /= np.linalg.norm(west)

        r = np.sqrt(grid_x**2 + grid_y**2)
        phi = np.arctan2(grid_x, grid_y)
        theta = r

        vec = (
            np.cos(theta)[...,None]*self.vec_center +
            np.sin(theta)[...,None]*(
                np.cos(phi)[...,None]*north +
                np.sin(phi)[...,None]*west
            ))

        theta_sph = np.arccos(vec[...,2])
        phi_sph = np.arctan2(vec[...,1], vec[...,0])
        phi_sph = np.mod(phi_sph, 2*np.pi)
        mask2d = mask[hp.ang2pix(nside, theta_sph, phi_sph)]

        return mask2d


    def define_region_mask(self, z, mask, dx, dy, nside): 

        # define region mask in the grid

        x, y = z[:, 0], z[:, 1]

        x_max, x_min = x.max(), x.min()
        y_max, y_min = y.max(), y.min()

        nx = int((x_max-x_min)/dx)
        ny = int((y_max-y_min)/dy)

        x_vals = np.linspace(x_min, x_max, nx)
        y_vals = np.linspace(y_min, y_max, ny)

        grid_x, grid_y = np.meshgrid(x_vals, y_vals)

        z_axis = self.vec_center/ np.linalg.norm(self.vec_center)
        k = np.array([0., 0., 1.])

        north = k - (k @ z_axis) * z_axis # proyeccion sobre el plano con normal z
        norm_n = np.linalg.norm(north)
        north = north / norm_n

        west = np.cross(z_axis, north)  # producto vectorial de z con y 
        west /= np.linalg.norm(west)

        r = np.sqrt(grid_x**2 + grid_y**2)
        phi = np.arctan2(grid_x, grid_y)
        theta = r

        vec = (
            np.cos(theta)[...,None]*self.vec_center +
            np.sin(theta)[...,None]*(
                np.cos(phi)[...,None]*north +
                np.sin(phi)[...,None]*west
            ))

        theta_sph = np.arccos(vec[...,2])
        phi_sph = np.arctan2(vec[...,1], vec[...,0])
        phi_sph = np.mod(phi_sph, 2*np.pi)
        mask2d = mask[hp.ang2pix(nside, theta_sph, phi_sph)]

        return mask2d


    def grid_mesh(self, z):

        x, y = z[:, 0], z[:, 1]

        x_max, x_min = x.max(), x.min()
        y_max, y_min = y.max(), y.min()

        x_vals = np.linspace(x_min, x_max, self.nbins)
        y_vals = np.linspace(y_min, y_max, self.nbins)

        grid_x, grid_y = np.meshgrid(x_vals, y_vals)

        return grid_x, grid_y 

    def grid_bins(self, z, values):

        # function that interpolate and make grid (nbins, nbins) = (nx, nx)
        
        x, y = z[:, 0], z[:, 1]

        x_max, x_min = x.max(), x.min()
        y_max, y_min = y.max(), y.min()

        x_vals = np.linspace(x_min, x_max, self.nbins)
        y_vals = np.linspace(y_min, y_max, self.nbins)

        grid_x, grid_y = np.meshgrid(x_vals, y_vals)

        grid = griddata(points=z, values=values, xi=(grid_x, grid_y), fill_value=0, method='cubic')

        return grid, grid_x, grid_y

    
    def grid_bins_mask_margin_deg(self, z, values, mask, method='cubic', margin_deg = 5.0):

        # function that interpolate and make grid (nbins, nbins) = (nx, nx) considering mask region

        x, y = z[:, 0], z[:, 1]

        x_max, x_min = x.max(), x.min()
        y_max, y_min = y.max(), y.min()

        margin = np.radians(margin_deg)

        x_vals = np.linspace(x_min - margin, x_max + margin, self.nbins)
        y_vals = np.linspace(y_min - margin, y_max + margin, self.nbins)   

        grid_x, grid_y = np.meshgrid(x_vals, y_vals)

        grid = griddata(points=z, values=values, xi=(grid_x, grid_y), fill_value=0, method=method)
        grid[mask == 0] = 0

        return grid



    def grid_bins_mask_margin(self, z, values, mask, method='cubic'):

        # function that interpolate and make grid (nbins, nbins) = (nx, nx) considering mask region

        x, y = z[:, 0], z[:, 1]

        x_max, x_min = x.max(), x.min()
        y_max, y_min = y.max(), y.min()

        dx = (x_max - x_min) / self.nbins
        dy = (y_max - y_min) / self.nbins
        margin = self.fact * max(dx, dy) 

        x_vals = np.linspace(x_min - margin, x_max + margin, self.nbins)
        y_vals = np.linspace(y_min - margin, y_max + margin, self.nbins)   

        grid_x, grid_y = np.meshgrid(x_vals, y_vals)

        grid = griddata(points=z, values=values, xi=(grid_x, grid_y), fill_value=0, method=method)
        grid[mask == 0] = 0

        return grid


    def grid_bins_mask_from_res_margin(self, z, values, mask, method='cubic'):
        
        # function that interpolate and make grid (nbins, nbins) = (nx, nx) considering mask region

        x, y = z[:, 0], z[:, 1]

        x_max, x_min = x.max(), x.min()
        y_max, y_min = y.max(), y.min()

        Lx = x_max - x_min
        Ly = y_max - y_min

        pix = np.radians(self.reso_arcmin / 60.0)
        margin = self.fact * pix

        def _round_up(n, mult=32):
            return int(np.ceil(n / mult) * mult)

        nbins_x = _round_up(np.ceil((Lx + 2 * margin) / pix))
        nbins_y = _round_up(np.ceil((Ly + 2 * margin) / pix))

        # Number of bins adapted to each side -> square pixels, rectangular image
        #nbins_x = int(np.ceil((Lx + 2 * margin) / pix))
        #nbins_y = int(np.ceil((Ly + 2 * margin) / pix))

        x_vals = x_min - margin + pix * np.arange(nbins_x)
        y_vals = y_min - margin + pix * np.arange(nbins_y)

        grid_x, grid_y = np.meshgrid(x_vals, y_vals)

        grid = griddata(points=z, values=values, xi=(grid_x, grid_y), fill_value=0, method=method)
        grid[mask == 0] = 0

        return grid


    def grid_bins_mask(self, z, values, mask):

        # function that interpolate and make grid (nbins, nbins) = (nx, nx) considering mask region

        x, y = z[:, 0], z[:, 1]

        x_max, x_min = x.max(), x.min()
        y_max, y_min = y.max(), y.min()

        x_vals = np.linspace(x_min, x_max, self.nbins)
        y_vals = np.linspace(y_min, y_max, self.nbins)

        grid_x, grid_y = np.meshgrid(x_vals, y_vals)

        grid = griddata(points=z, values=values, xi=(grid_x, grid_y), fill_value=0, method='cubic')
        grid[mask == 0] = 0

        return grid

    def grid_bins_binning(self, z, values, mask2d):

        x, y = z[:, 0], z[:, 1]

        x_max, x_min = x.max(), x.min()
        y_max, y_min = y.max(), y.min()

        dx = (x_max - x_min) / self.nbins
        dy = (y_max - y_min) / self.nbins
        margin = self.fact * max(dx, dy)

        xmin = x_min - margin
        ymin = y_min - margin

        grid = np.zeros((self.nbins, self.nbins))
        counts = np.zeros((self.nbins, self.nbins))

        # indices de píxel
        ix = ((x - xmin) / (dx)).astype(int)
        iy = ((y - ymin) / (dy)).astype(int)

        # evitar out of bounds
        valid = (ix >= 0) & (ix < self.nbins) & (iy >= 0) & (iy < self.nbins)

        ix = ix[valid]
        iy = iy[valid]
        vals = values[valid]

        # acumular
        for i in range(len(ix)):
            grid[iy[i], ix[i]] += vals[i]
            counts[iy[i], ix[i]] += 1

        # promedio
        mask_nonzero = counts > 0
        grid[mask_nonzero] /= counts[mask_nonzero]
        grid[mask2d == 0] = 0

        return grid


    def sub_grid(self, z, value=None):

        x, y = z[:, 0], z[:, 1]

        if value == None:
            r = np.sqrt(x**2 + y**2)
            L = r.max()/np.sqrt(2)
        else:
            r = np.sqrt(x**2 + y**2)
            L = r.max()/np.sqrt(value)

        inside = (x >= -L) & (x <= L) & (y >= -L) & (y <= L)
        z_inside = z[inside]

        return z_inside, inside

    def grid_cuad(self, z, values, value = None):

        x, y = z[:, 0], z[:,1]

        if value == None:
            r = np.sqrt(x**2 + y**2)
            L = r.max()/np.sqrt(2)
        else: 
            r = np.sqrt(x**2 + y**2)
            L = r.max()/np.sqrt(value)

        x_vals = np.linspace(-L, L, self.nbins)
        y_vals = np.linspace(-L, L, self.nbins)
        grid_x, grid_y = np.meshgrid(x_vals, y_vals)
        grid = griddata(points=z, values=values, xi=(grid_x, grid_y), fill_value=0, method='cubic')

        inside = (x >= -L) & (x <= L) & (y >= -L) & (y <= L)
        z_inside = z[inside]

        return grid, grid_x, grid_y, inside, z_inside

    def deprojection(self, z, grid, grid_x, grid_y):
        x = grid_x[0,:]
        y = grid_y[:, 0]
        interpolator = RegularGridInterpolator((y, x), grid.T, method='cubic', bounds_error=False, fill_value=0)
        mapped_values = interpolator(z)
        return mapped_values

    def deprojection_old(self, z, grid, grid_x, grid_y):
        
        flat_img = grid.flatten()
        grid_coords = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
        mapped_values = griddata(grid_coords, flat_img, z, fill_value=0, method='cubic')  # shape: (Npix,)

        return mapped_values

class rotate_geo: 

    def __init__(self, theta, phi, theta0, phi0): 

        self.theta_Q = theta
        self.phi_Q = phi 
        self.theta_P = theta0
        self.phi_P = phi0

    def basis(self, theta, phi):
        e_theta = np.array([np.cos(theta)*np.cos(phi),
            np.cos(theta)*np.sin(phi),
            -np.sin(theta)])
        e_phi = np.array([-np.sin(phi), np.cos(phi), np.zeros_like(phi)])

        return e_theta, e_phi


    def psi_Q_to_P(self):
        
        """
        Angulo psi tal que:
        (Q ± iU)_P = exp\mp 2 i psi) (Q ± iU)_Q

        Convención:
        x oeste  = -e_phi
        y norte  = -e_theta

        """

        et_Q, ep_Q = self.basis(self.theta_Q, self.phi_Q)
        et_P, ep_P = self.basis(self.theta_P, self.phi_P)

        # base compleja consistente con (x,y)
        m_Q = -et_Q - 1j*ep_Q
        m_P = -et_P - 1j*ep_P

        z = np.sum(m_Q * np.conj(m_P), axis=0)
        psi = np.angle(z)

        return psi


class rotate: 

    def __init__(self, theta, phi, theta0, phi0): 

        self.theta_Q = theta
        self.phi_Q = phi 
        self.theta_P = theta0
        self.phi_P = phi0

    def basis(self, theta, phi):
        e_theta = np.array([np.cos(theta)*np.cos(phi),
            np.cos(theta)*np.sin(phi),
            -np.sin(theta)])
        e_phi = np.array([-np.sin(phi), np.cos(phi), np.zeros_like(phi)])

        return e_theta, e_phi

    def delta_alpha(self, theta_Q, phi_Q, phi_P):

        return (phi_P - phi_Q) * np.cos(theta_Q)

    def rotate_in_plane(self, v_theta, v_phi, dalpha):
        c = np.cos(dalpha); s = np.sin(dalpha)

        return (c*v_theta - s*v_phi, s*v_theta + c*v_phi)

    def components_to_cartesian(self, v_theta, v_phi, theta, phi):
        e_theta, e_phi = self.basis(theta, phi)

        return v_theta*e_theta + v_phi*e_phi

    def psi_Q_to_P(self):
        
        #1) norte en Q en componentes locales

        vQ_theta, vQ_phi = -1.0, 0.0

        #2) rotar componentes por delta_alpha (transporte sobre paralelo theta_Q)

        dalpha = self.delta_alpha(self.theta_Q, self.phi_Q, self.phi_P)

        v_mid_theta, v_mid_phi = self.rotate_in_plane(vQ_theta, vQ_phi, dalpha)

        #3) reconstruir vector en R3 pero en el punto (theta_Q, phi_P) (punto intermedio)

        v_mid_cart = self.components_to_cartesian(v_mid_theta, v_mid_phi, self.theta_Q, self.phi_P)

        #4) proyectar en la base del centro P (theta_P, phi_P)

        e_theta_P, e_phi_P = self.basis(self.theta_P, self.phi_P)
        north_P = -e_theta_P   # +y local
        east_P  =  e_phi_P     # +x local

        a = np.dot(v_mid_cart.T, east_P[:,0])
        b = np.dot(v_mid_cart.T, north_P[:,0])

        # 5) ángulo final

        psi = np.arctan2(a, b)

        return -psi, dalpha


    




























