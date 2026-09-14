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
from scipy.signal.windows import tukey

import pickle
import nifty7 as ift
#import pymaster as nmt


def components_spectrum_cosmo(
    
    r=0.034,
    H0=67.5,
    ombh2=0.022,
    omch2=0.122,
    tau=0.06,
    ns=0.965,
    As=2e-9,
    mnu=0.06):
    
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=0, tau=0.06)
    pars.InitPower.set_params(As=As, ns=ns, r=r)
    pars.set_for_lmax(7000, lens_potential_accuracy=1)
    pars.WantTensors = True

    results = camb.get_results(pars)
    
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')

    totCL          = powers['total']
    unlensedCL     = powers['unlensed_scalar']
    unlensed_Tot   = powers['unlensed_total']
    lensed_SC      = powers['lensed_scalar']
    tensorCL       = powers['tensor']
    lens_potential = powers['lens_potential']  

    lmax=7000
    ls = np.arange(lmax+1)
    factor = ls*(ls+1)/2./np.pi
    #lmax=7000

    cl_TT_tot = np.copy(totCL[:lmax+1,0])
    cl_EE_tot = np.copy(totCL[:lmax+1,1])
    cl_BB_tot = np.copy(totCL[:lmax+1,2])

    cl_TT_tot[2:] = cl_TT_tot[2:] / factor[2:]
    cl_EE_tot[2:] = cl_EE_tot[2:] / factor[2:]
    cl_BB_tot[2:] = cl_BB_tot[2:] / factor[2:]

    cl_TT_for_map = np.copy(cl_TT_tot)
    cl_EE_for_map = np.copy(cl_EE_tot)
    cl_BB_for_map = np.copy(cl_BB_tot)

    #cl_TT_for_map[lmax+1:len(cl_TT_unlensed)] = 0.
    
    return cl_TT_for_map, cl_EE_for_map, cl_BB_for_map


def bb_spectrum_cosmo(
    
    r=0.034,
    H0=67.5,
    ombh2=0.022,
    omch2=0.122,
    tau=0.06,
    ns=0.965,
    As=2e-9,
    mnu=0.06):
    
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=0, tau=0.06)
    pars.InitPower.set_params(As=As, ns=ns, r=r)
    pars.set_for_lmax(7000, lens_potential_accuracy=1)
    pars.WantTensors = True

    results = camb.get_results(pars)
    
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')

    totCL          = powers['total']
    unlensedCL     = powers['unlensed_scalar']
    unlensed_Tot   = powers['unlensed_total']
    lensed_SC      = powers['lensed_scalar']
    tensorCL       = powers['tensor']
    lens_potential = powers['lens_potential']  

    lmax=2000
    ls = np.arange(lmax+1)
    factor = ls*(ls+1)/2./np.pi
    #lmax=7000
    cl_BB_tot = np.copy(totCL[:lmax+1,2])
    cl_BB_tot[2:] = cl_BB_tot[2:] / factor[2:]

    cl_BB_for_map = np.copy(cl_BB_tot)

    #cl_TT_for_map[lmax+1:len(cl_TT_unlensed)] = 0.
    
    return cl_BB_for_map


def get_templates_fiducial(H0=67.5, ombh2=0.022, omch2=0.122, tau=0.06,
                           ns=0.965, As=2e-9, mnu=0.06, r_ref=0.034,
                           lmax=7000):
    """
    Templates fiduciales para el modelo Cl_th = r * clBB_prim + Alens * clBB_lens.

    clTT_fid  : TT lensed scalar fiducial (para generar realizaciones).
    clEE_fid  : EE lensed scalar fiducial (insensible a r y Alens a estos
                niveles; se usa fijo para generar las realizaciones Q/U).
    clBB_prim : tensores con amplitud r=1 y tilt nt = -r_ref/8, para que
                coincida con la forma tensorial de espectros generados con
                la relacion de consistencia en r = r_ref.
    clBB_lens : escalares lensados (equivale a r=0, Alens=1).
    """
    nt = -r_ref / 8.0
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=0, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=1.0, nt=nt, ntrun=0)   # r=1, nt fijo
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.WantTensors = True
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')

    ls = np.arange(lmax + 1)
    factor = ls * (ls + 1) / 2. / np.pi

    # TT y EE fiduciales: escalares lensados
    clTT_fid = np.copy(powers['lensed_scalar'][:lmax + 1, 0])
    clTT_fid[2:] /= factor[2:]

    clEE_fid = np.copy(powers['lensed_scalar'][:lmax + 1, 1])
    clEE_fid[2:] /= factor[2:]

    # Template primordial BB: SOLO tensores, amplitud r=1, tilt de r_ref
    clBB_prim = np.copy(powers['tensor'][:lmax + 1, 2])
    clBB_prim[2:] /= factor[2:]

    # Template lensing BB: escalares lensados (equivale a r=0, Alens=1)
    clBB_lens = np.copy(powers['lensed_scalar'][:lmax + 1, 2])
    clBB_lens[2:] /= factor[2:]

    return clTT_fid, clEE_fid, clBB_prim, clBB_lens


def clbb_theory(r, Alens, clBB_prim, clBB_lens):
    return r * clBB_prim + Alens * clBB_lens


def correction_modified(El_true, bl, fisher, cl_bin_fid, nsamples, nbins):

    fisher_inverse = np.linalg.inv(fisher)
    correction = np.zeros((nsamples, nbins))
    for i in range(nsamples):
        nuevo = El_true[i] - bl
        nuevo = np.reshape(nuevo, (nbins, 1))
        correction_shape = np.dot(fisher_inverse, nuevo)
        correction[i, :] = np.reshape(correction_shape, (nbins))

    correction_mean = np.mean(correction, axis=0)
    correction_desv = np.std(correction, axis=0)

    estimation = cl_bin_fid + correction_mean

    return estimation, correction_desv, correction


def signal_spectrum(r=0.034):

    #use CAMB to simulate the fiducial power spectrum

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=r)
    pars.set_for_lmax(7000, lens_potential_accuracy=1)
    pars.WantTensors = True

    results = camb.get_results(pars)
    
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')

    totCL          = powers['total']
    unlensedCL     = powers['unlensed_scalar']
    unlensed_Tot   = powers['unlensed_total']
    lensed_SC      = powers['lensed_scalar']
    tensorCL       = powers['tensor']
    lens_potential = powers['lens_potential']  

    lmax=7000
    ls = np.arange(lmax+1)
    factor = ls*(ls+1)/2./np.pi
    #lmax=7000
    
    cl_TT_unlensed = np.copy(totCL[:lmax+1,0])
    cl_EE_unlensed = np.copy(totCL[:lmax+1,1])
    cl_BB_unlensed = np.copy(totCL[:lmax+1,2])

    cl_TT_unlensed[2:] = cl_TT_unlensed[2:] / factor[2:]
    cl_EE_unlensed[2:] = cl_EE_unlensed[2:] / factor[2:]
    cl_BB_unlensed[2:] = cl_BB_unlensed[2:] / factor[2:]

    cl_TT_for_map = np.copy(cl_TT_unlensed)
    cl_EE_for_map = np.copy(cl_EE_unlensed)
    cl_BB_for_map = np.copy(cl_BB_unlensed)

    #cl_TT_for_map[lmax+1:len(cl_TT_unlensed)] = 0.
    
    return cl_TT_for_map, cl_EE_for_map, cl_BB_for_map

def bl(fwhm, lmax):

    """
    Beam gaussiano en la esfera.
    
    fwhm: ancho a mitad de potencia [radianes]
    lmax: maximo multipolo
    
    return: B(l)
    """

    ls = np.arange(0,lmax+1)
    sigma = fwhm/(np.sqrt(8*np.log(2)))
    return np.exp(-(ls*(ls+1)*sigma**2/2))

def beam_flat(fwhm, k):
    """
    Beam gaussiano en el plano (flat-sky).
    
    fwhm: ancho a mitad de potencia [radianes]
    k: array 2D o 1D de frecuencias angulares (≈ multipolos)
    
    return: B(k)
    """
    sigma = fwhm / np.sqrt(8*np.log(2))
    return np.exp(-0.5 * (k*sigma)**2)


def apodize_map(m, nx, alpha=0.1):
    
    ny = nx
    wx = tukey(nx, alpha)
    wy = tukey(ny, alpha)
    w2d = np.outer(wx, wy)
    return m * w2d, w2d



def hyper(hyperparameters):
    return "n_filters0_" + str(hyperparameters[0]) +  "_n_filters1_" + str(hyperparameters[1]) + "_n_filters2_" + str(hyperparameters[2]) + "_n_filters3_" + "_lr_" + "{:.3e}".format(hyperparameters[3]) + "_wd_" + "{:.3e}".format(hyperparameters[4])# + "_lambda_" + "{:.2e}".format(hyperparameters[7])


def correlated_noise(l, nl_white, l_knee, alfa_knee): 
    #simulate correlated power spectrum
    return nl_white + nl_white*(l/l_knee)**alfa_knee


def hits_model_elliptical_sphere(
    nside,
    lonc,
    latc,
    radius_deg=40,
    sigma_u_deg=10,
    sigma_v_deg=5,
    angle_deg=0,
    normalize=True
):
    """
    Generate an elliptical hits map directly on the sphere.

    Parameters
    ----------
    nside : int
        HEALPix nside.

    lonc, latc : float
        Center coordinates in degrees.

    radius_deg : float
        Hard cutoff radius of the observed patch.

    sigma_u_deg : float
        Width along major axis.

    sigma_v_deg : float
        Width along minor axis.

    angle_deg : float
        Rotation angle of ellipse in tangent plane.

    normalize : bool
        Normalize maximum to 1.

    Returns
    -------
    hits : ndarray
        Simulated hits map.
    """

    npix = hp.nside2npix(nside)

    # -----------------------------------
    # Pixel vectors
    # -----------------------------------

    vecs = np.array(
        hp.pix2vec(nside, np.arange(npix))
    ).T

    # -----------------------------------
    # Center vector
    # -----------------------------------

    center = hp.ang2vec(lonc, latc, lonlat=True)

    # -----------------------------------
    # Build tangent basis
    # -----------------------------------

    z = np.array([0., 0., 1.])

    # Avoid numerical issues near poles
    if np.abs(np.dot(center, z)) > 0.99:
        z = np.array([1., 0., 0.])

    e1 = np.cross(z, center)
    e1 /= np.linalg.norm(e1)

    e2 = np.cross(center, e1)
    e2 /= np.linalg.norm(e2)

    # -----------------------------------
    # Rotate tangent basis
    # -----------------------------------

    alpha = np.radians(angle_deg)

    e1_rot = np.cos(alpha) * e1 + np.sin(alpha) * e2
    e2_rot = -np.sin(alpha) * e1 + np.cos(alpha) * e2

    # -----------------------------------
    # Local coordinates
    # -----------------------------------

    u = vecs @ e1_rot
    v = vecs @ e2_rot

    # -----------------------------------
    # Angular distance to center
    # -----------------------------------

    cosang = vecs @ center
    ang = np.arccos(np.clip(cosang, -1, 1))

    # -----------------------------------
    # Elliptical Gaussian
    # -----------------------------------

    su = np.sin(np.radians(sigma_u_deg))
    sv = np.sin(np.radians(sigma_v_deg))

    hits = np.exp(
        -0.5 * (
            (u / su)**2 +
            (v / sv)**2
        )
    )

    # -----------------------------------
    # Apply hard radius mask
    # -----------------------------------

    radius_rad = np.radians(radius_deg)

    hits[ang > radius_rad] = 0.

    # -----------------------------------
    # Normalize
    # -----------------------------------

    if normalize:
        hits /= hits.max()

    return hits


def hits_model(nside, center, th0, mask, sig=30):
    #simulate hits maps

    npix = hp.nside2npix(nside)
    vecs_all = hp.pix2vec(nside, np.arange(npix))
    theta_all, phi_all = hp.pix2ang(nside, np.arange(npix))  # theta: co-lat (0=North pole), phi: RA-like longitude

    cosang = np.dot(center, vecs_all)
    ang = np.arccos(np.clip(cosang, -1, 1))

    sigma = np.radians(sig)
    N0 = 1.0
    
    Nhits = N0 * np.exp(-0.5 * (ang/sigma)**2)
    dec_rad = (0.5 * np.pi - theta_all)
    dec0_rad = (0.5 * np.pi - th0)

    mod = np.cos(dec_rad - dec0_rad)**2
    mod = np.clip(mod, 0, None)
    Nhits_mod = Nhits * mod
    Nhits_mod_masked = Nhits_mod * mask

    return Nhits_mod_masked, Nhits_mod


def nhits_to_sigma2(noise_pix_sph, Nhits, mask):
    # build variance maps from Nhits

    inv_hits = np.zeros_like(Nhits)
    inv_hits[mask > 0] = 1.0 / (Nhits[mask > 0] + 1e-6)
    
    norm = np.mean(inv_hits[mask > 0])
    variance_map = noise_pix_sph * inv_hits / norm
    
    return variance_map


def sigma2_to_map(variance_map, mask, rng=None):
    # buil noise realization given variance map

    if rng is None:
        rng = np.random.default_rng()

    noise_map = np.zeros_like(variance_map)
    noise_map[mask > 0] = rng.normal(loc=0.0, scale=np.sqrt(variance_map[mask > 0]))

    return noise_map

def nhitsproj_to_sigma2(noise_pix_plane, Nhits_plane, mask):

    #noise_pix_plane = 0.093

    # máscara efectiva: solo donde hay hits reales
    valid = (mask > 0) & (Nhits_plane > 0)

    inv_hits = np.zeros_like(Nhits_plane)
    inv_hits[valid] = 1.0 / Nhits_plane[valid]

    norm = np.mean(inv_hits[valid])

    sigma2_plane = np.zeros_like(Nhits_plane)
    sigma2_plane[valid] = noise_pix_plane * inv_hits[valid] / norm

    return sigma2_plane

def make_maps(cls, nside, seed=None):

    resolution = hp.nside2resol(nside, arcmin=True)

    if seed is None:
        maps = hp.synfast(cls,nside=nside,new=True,pol=True)

    else:
        # Save current NumPy random state
        random_state = np.random.get_state()

        try:
            # Set seed only for hp.synfast
            np.random.seed(seed)

            maps = hp.synfast(cls,nside=nside,new=True,pol=True)

        finally:
            # Restore original NumPy random state
            np.random.set_state(random_state)

    return maps, resolution


def make_scalar_maps(cls, nside):
    #np.random.seed(1234)
    resolution = hp.nside2resol(nside, arcmin = True)
    maps = hp.synfast(cls, nside=nside, new=True, pol=False) # acá esta pol == False
    return maps, resolution


def transf_eb(qmap, umap, nx, ny, dx, dy):

    lx,ly = np.meshgrid( np.fft.fftfreq( nx, dx )[0:int(nx/2+1)]*2.*np.pi, np.fft.fftfreq( ny, dy )*2.*np.pi )
    #tpi  = 2.*np.arctan2(lx, -ly)
    tpi = 2.*np.arctan2(ly, -lx)
    tpi = torch.Tensor(tpi).cuda()
    tfac = np.sqrt((dx * dy) / (nx * ny))
    qfft = torch.fft.rfft2(qmap)*tfac
    ufft = torch.fft.rfft2(umap)*tfac

    efft = (+torch.cos(tpi) * qfft + torch.sin(tpi) * ufft)
    bfft = (-torch.sin(tpi) * qfft + torch.cos(tpi) * ufft)
    return efft, bfft

def transf_eb2(qmap, umap, nx, dx, ny, dy, yes=False):

    # Normalización
    tfac = np.sqrt((dx * dy) / (nx * ny))

    # FFTs
    qfft = torch.fft.rfft2(qmap) * tfac
    ufft = torch.fft.rfft2(umap) * tfac

    # Modos de Fourier físicos
    lx = np.fft.rfftfreq(nx, d=dx) * 2.0 * np.pi
    ly = np.fft.fftfreq(ny, d=dy) * 2.0 * np.pi

    # Malla de Fourier
    lx, ly = np.meshgrid(lx, ly)

    # angulo del modo Fourier
    if yes:
        tpi = 2.0 * np.arctan2(lx, -ly)
        tpi = torch.Tensor(tpi).cuda()
    else:
        tpi = 2.0 * np.arctan2(ly, -lx)
        tpi = torch.Tensor(tpi).cuda()

    # Rotación Q/U -> E/B
    efft = torch.cos(tpi) * qfft + torch.sin(tpi) * ufft
    bfft = -torch.sin(tpi) * qfft + torch.cos(tpi) * ufft

    # Transformada inversa
    #emap = np.fft.irfft2(efft, s=(ny, nx)) / tfac
    #bmap = np.fft.irfft2(bfft, s=(ny, nx)) / tfac

    return efft, bfft

def transf_eb_np(qmap, umap, nx, ny, dx, dy):

    lx,ly = np.meshgrid( np.fft.fftfreq( nx, dx )[0:int(nx/2+1)]*2.*np.pi, np.fft.fftfreq( ny, dy )*2.*np.pi )
    #tpi  = 2.*np.arctan2(lx, -ly)
    tpi = 2.*np.arctan2(ly, -lx)
    #tpi = torch.Tensor(tpi).cuda()
    tfac = np.sqrt((dx * dy) / (nx * ny))
    qfft = np.fft.rfft2(qmap)*tfac
    ufft = np.fft.rfft2(umap)*tfac

    efft = (+np.cos(tpi) * qfft + np.sin(tpi) * ufft)
    bfft = (-np.sin(tpi) * qfft + np.cos(tpi) * ufft)

    return efft, bfft

def transf_eb2_np(qmap, umap, nx, dx, ny, dy, yes=False):

    # Normalización
    tfac = np.sqrt((dx * dy) / (nx * ny))

    # FFTs
    qfft = np.fft.rfft2(qmap) * tfac
    ufft = np.fft.rfft2(umap) * tfac

    # Modos de Fourier físicos
    lx = np.fft.rfftfreq(nx, d=dx) * 2.0 * np.pi
    ly = np.fft.fftfreq(ny, d=dy) * 2.0 * np.pi

    # Malla de Fourier
    lx, ly = np.meshgrid(lx, ly)

    # angulo del modo Fourier
    if yes:
        tpi = 2.0 * np.arctan2(lx, -ly)
    else:
        tpi = 2.0 * np.arctan2(ly, -lx)

    # Rotación Q/U -> E/B
    efft = np.cos(tpi) * qfft + np.sin(tpi) * ufft
    bfft = -np.sin(tpi) * qfft + np.cos(tpi) * ufft

    # Transformada inversa
    #emap = np.fft.irfft2(efft, s=(ny, nx)) / tfac
    #bmap = np.fft.irfft2(bfft, s=(ny, nx)) / tfac

    return efft, bfft


def transf_qu_eb(emap, bmap, nx, ny, dx, dy, only_e = False, only_b = False):
    
    lx,ly = np.meshgrid( np.fft.fftfreq( nx, dx )[0:int(nx/2+1)]*2.*np.pi, np.fft.fftfreq( ny, dy )*2.*np.pi )  
    tpi  = 2.*np.arctan2(lx, -ly)
    tfac = np.sqrt((dx * dy) / (nx * ny))
    efft = np.fft.rfft2(emap)*tfac
    bfft = np.fft.rfft2(bmap)*tfac
    
    if only_e == True: 
        qmap = np.fft.irfft2(np.cos(tpi)*efft)/tfac
        umap = np.fft.irfft2(np.sin(tpi)*efft)/tfac
    elif only_b == True: 
        qmap = np.fft.irfft2(-np.sin(tpi)*bfft)/tfac
        umap = np.fft.irfft2(np.cos(tpi)*bfft)/tfac
    else:
        qmap = np.fft.irfft2(np.cos(tpi)*efft - np.sin(tpi)*bfft)/tfac
        umap = np.fft.irfft2(np.sin(tpi)*efft + np.cos(tpi)*bfft)/tfac

    return qmap, umap

def QU_to_EB_fft(Q_plane, U_plane, dx, eps=1e-15):
    """
    Q_plane, U_plane : 2D arrays shape (ny, nx)
    dx, dy : pixel size in x,y (arbitrary units). Only used for fftfreq scale;
             angle phi_k = atan2(ky, kx) is scale-invariant.
    Returns: E_map, B_map (real 2D arrays), and optionally E_k, B_k in Fourier domain.
    """
    # 1) forward FFTs (complex)
    Qk = torch.fft.fft2(Q_plane)
    Uk = torch.fft.fft2(U_plane)
    Pk = Qk + 1j * Uk  # complex Fourier field

    ny, nx = Q_plane.shape

    # 2) build kx, ky grids (Hz units). We don't need 2pi for the angle.
    kx = np.fft.fftfreq(nx, d=dx)   # shape (nx,)
    ky = np.fft.fftfreq(ny, d=dx)   # shape (ny,)
    kxg, kyg = np.meshgrid(kx, ky)  # shape (ny, nx)

    # 3) angle phi_k
    phi_k = np.arctan2(kyg, -kxg)    # shape (ny, nx)
    phi_k = torch.Tensor(phi_k).cuda()


    # 4) rotation factor e^{-2 i phi_k}
    rot = torch.exp(-2j * phi_k)

    # 5) compute E_k + i B_k
    EBk = rot * Pk

    Ek = EBk.real
    Bk = EBk.imag

    # 6) inverse FFT to real space
    E_map = torch.fft.ifft2(Ek).real
    B_map = torch.fft.ifft2(Bk).real


    return E_map, B_map, phi_k


def eth2_fourier(phiE, phiB, nx, dx, grid_x, grid_y):
    """
    Compute Q, U from scalar potentials phiE, phiB
    using flat-sky spin-2 operator in Fourier space (Torch).
    
    phiE, phiB: (..., nx, nx)
    grid_x, grid_y: (nx, nx)
    """
    device = phiE.device

    z  = grid_x + 1j * grid_y
    r2 = grid_x**2 + grid_y**2

    #z  = z.to(device)
    #r2 = r2.to(device)

    # ----------------------------------
    # Complex scalar potential
    # ----------------------------------
    Psi = torch.complex(phiE, phiB)   # (..., nx, nx)

    # ----------------------------------
    # Fourier transform
    # ----------------------------------
    Psi_k = torch.fft.fft2(Psi)

    # ----------------------------------
    # Fourier grid
    # ----------------------------------
    freq = torch.fft.fftfreq(nx, d=dx, device=device) * 2.0 * np.pi
    lx, ly = torch.meshgrid(freq, freq, indexing="ij")

    l  = lx + 1j * ly
    d2 = l**2

    # ----------------------------------
    # Derivatives in Fourier space
    # ----------------------------------
    d2Psi_k = -d2 * Psi_k
    d2Psi   = torch.fft.ifft2(d2Psi_k)

    dPsi_k = 1j * l * Psi_k
    dPsi   = torch.fft.ifft2(dPsi_k)

    # ----------------------------------
    # Spin connection (stereographic)
    # ----------------------------------
    connection = z / (4.0 + r2)

    # ----------------------------------
    # Spin-2 field
    # ----------------------------------
    P = -(d2Psi - connection * dPsi)

    Q = P.real
    U = P.imag

    return Q, U

def eth2_correct(phiE, phiB, nx, dx, grid_x, grid_y):
    """
    Compute Q, U from scalar potentials phiE, phiB
    using flat-sky spin-2 operator in Fourier space (Torch).
    
    phiE, phiB: (..., nx, nx)
    grid_x, grid_y: (nx, nx)
    """
    device = phiE.device

    z  = grid_x + 1j * grid_y
    r2 = grid_x**2 + grid_y**2

    #z  = z.to(device)
    #r2 = r2.to(device)

    # ----------------------------------
    # Complex scalar potential
    # ----------------------------------
    Psi = torch.complex(phiE, phiB)   # (..., nx, nx)

    # ----------------------------------
    # Fourier transform
    # ----------------------------------
    Psi_k = torch.fft.fft2(Psi)

    # ----------------------------------
    # Fourier grid
    # ----------------------------------
    freq = torch.fft.fftfreq(nx, d=dx, device=device) * 2.0 * np.pi
    lx, ly = torch.meshgrid(freq, freq, indexing="ij")

    l  = lx + 1j * ly

    # ----------------------------------
    # Derivatives in Fourier space
    # ----------------------------------

    dPsi_k = 1j * l * Psi_k
    dPsi   = torch.fft.ifft2(dPsi_k)

    # ----------------------------------
    # Spin connection (stereographic)
    # ----------------------------------
    connection = z / (4.0 + r2)
    geom_k = torch.fft.fft2(connection*dPsi)

    # ----------------------------------
    # Spin-2 field
    # ----------------------------------
    P_k = -(- (lx + 1j*ly)**2 * Psi_k - geom_k)


    Q = torch.fft.ifft2(P_k).real
    U = torch.fft.ifft2(P_k).imag

    return Q, U



# make the spin-2 operator for stereographic coordinates 
def d_dx(f, dx):
    # axis=1 → x direction
    return (torch.roll(f, shifts=-1, dims=1) -
            torch.roll(f, shifts= 1, dims=1)) / (2.0 * dx)

def d_dy(f, dy):
    # axis=0 → y direction
    return (torch.roll(f, shifts=-1, dims=0) -
            torch.roll(f, shifts= 1, dims=0)) / (2.0 * dy)

def d_dzbar(f, dx, dy):
    
    return 0.5 * (d_dx(f, dx) + 1j * d_dy(f, dy))


def eth2_from_scalar_torch(phi_E, phi_B, grid_x, grid_y, dx):
    
    """
    Compute Q, U from scalar potentials on a stereographic plane.

    Parameters
    ----------
    phi_E, phi_B : torch.Tensor (N, N)
        Scalar potentials (NN outputs).
    grid_x, grid_y : torch.Tensor (N, N)
        Physical stereographic coordinates (not pixel indices).

    Returns
    -------
    Q, U : torch.Tensor (N, N)
        Stokes parameters.
    """

    # Pixel size in physical coordinates
    #dx = grid_x[0, 1] - grid_x[0, 0]
    #dy = grid_y[1, 0] - grid_y[0, 0]
    dx = dx
    dy = dx

    device = phi_E.device
    dtype  = phi_E.dtype

    grid_x = torch.tensor(grid_x, device=device, dtype=dtype)
    grid_y = torch.tensor(grid_y, device=device, dtype=dtype)

    # Complex scalar potential
    Psi = phi_E + 1j * phi_B

    # Complex plane coordinates
    z    = grid_x + 1j * grid_y
    zbar = grid_x - 1j * grid_y
    r2   = grid_x**2 + grid_y**2

    # derivatives
    dPsi  = d_dzbar(Psi, dx, dy)
    d2Psi = d_dzbar(dPsi, dx, dy)

    # Spin connection (stereographic geometry)
    connection = z / (4.0 + r2)

    # Spin-2 field
    P = -(d2Psi - connection * dPsi)# * (dx**2)

    Q = P.real
    U = P.imag

    return Q, U

def eth2_from_scalar_pseudospec(phi_E, phi_B, grid_x, grid_y, nx, dx, map_rescale_factor_q):
    device = phi_E.device
    dtype  = phi_E.dtype

    #print(phi_E)

    grid_x = torch.as_tensor(grid_x, device=device, dtype=dtype)
    grid_y = torch.as_tensor(grid_y, device=device, dtype=dtype)

    Psi = phi_E + 1j * phi_B

    #print(Psi)

    # Fourier grid
    k = torch.fft.fftfreq(nx, d=dx, device=device) * 2 * torch.pi
    kx, ky = torch.meshgrid(k, k, indexing='xy')
    k_plus = kx + 1j * ky

    # FFT of Psi
    Psi_k = torch.fft.fft2(Psi)

    # ∂ Psi
    dPsi = torch.fft.ifft2(1j * k_plus * Psi_k)

    # ∂² Psi
    d2Psi = torch.fft.ifft2(-(k_plus**2) * Psi_k)

    # Geometry
    z  = grid_x + 1j * grid_y
    r2 = grid_x**2 + grid_y**2
    connection = z / (4.0 + r2)

    # eth^2 operator (CORRECT)
    P = -(
        0.25*d2Psi
        -0.5*connection * dPsi
    )
    #print(P)

    #P = P*map_rescale_factor_q
    #print(P)

    return P.real, P.imag


def eth2_from_scalar_batch_torch(phiE, phiB, dx, grid_x=None, grid_y=None):
    """
    Spin-2 operator (eth^2) in stereographic coordinates.
    PyTorch, batched, autograd-safe.

    Inputs
    ------
    phiE, phiB : (B, N, N) real tensors
    dx : float
    grid_x, grid_y : (N, N) real tensors (optional)

    Outputs
    -------
    Q, U : (B, N, N) real tensors
    """

    B, N, _ = phiE.shape
    device = phiE.device
    dtype  = phiE.dtype

    # --------------------------------------------------
    # Coordinate grids (shared, no grad)
    # --------------------------------------------------
    if grid_x is None or grid_y is None:
        x = torch.arange(N, device=device, dtype=dtype) * dx - (N * dx) / 2.0
        y = torch.arange(N, device=device, dtype=dtype) * dx - (N * dx) / 2.0
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

    grid_x = torch.tensor(grid_x, device=device, dtype=dtype)
    grid_y = torch.tensor(grid_y, device=device, dtype=dtype)

    z  = grid_x + 1j * grid_y
    r2 = grid_x**2 + grid_y**2

    # Add batch dim
    z  = z.unsqueeze(0)
    r2 = r2.unsqueeze(0)

    # --------------------------------------------------
    # Finite differences
    # --------------------------------------------------
    def d_dx(f):
        df = torch.zeros_like(f, dtype=torch.complex64)
        df[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2.0 * dx)
        df[:, :, 0]    = (f[:, :, 1] - f[:, :, 0]) / dx
        df[:, :, -1]   = (f[:, :, -1] - f[:, :, -2]) / dx
        return df

    def d_dy(f):
        df = torch.zeros_like(f, dtype=torch.complex64)
        df[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2.0 * dx)
        df[:, 0, :]    = (f[:, 1, :] - f[:, 0, :]) / dx
        df[:, -1, :]   = (f[:, -1, :] - f[:, -2, :]) / dx
        return df

    def d_dzbar(f):
        return 0.5 * (d_dx(f) + 1j * d_dy(f))

    # --------------------------------------------------
    # Spin-2 operator
    # --------------------------------------------------
    Psi = phiE + 1j * phiB

    dPsi  = d_dzbar(Psi)
    d2Psi = d_dzbar(dPsi)

    connection = z / (4.0 + r2)

    P = -(d2Psi - connection * dPsi)

    Q = P.real
    U = P.imag

    return Q, U


def spin2_from_potentials(phiE, phiB, grid_x, grid_y, dx, dy, eps=1e-6):
    """
    Compute Q,U from scalar potentials phiE, phiB using the exact spin-2 operator
    in azimuthal equidistant projection (r = theta).

    Parameters
    ----------
    phiE, phiB : (Ny, Nx) arrays
        Scalar E and B potentials (network outputs)
    grid_x, grid_y : (Ny, Nx)
        Plane coordinates of each pixel
    dx, dy : float
        Pixel size in x and y
    eps : float
        Small regularization near theta=0

    Returns
    -------
    Q, U : (Ny, Nx)
    """

    # --------------------------------------------------
    # Geometry
    # --------------------------------------------------
    r = np.sqrt(grid_x**2 + grid_y**2)
    theta = r
    phi = np.arctan2(grid_x, grid_y)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    sin_theta = np.where(sin_theta < eps, eps, sin_theta)
    sin2_theta = sin_theta**2
    cot_theta = cos_theta / sin_theta

    sinphi = np.sin(phi)
    cosphi = np.cos(phi)

    # --------------------------------------------------
    # Cartesian derivatives
    # --------------------------------------------------
    def grad_xy(f):
        df_dy, df_dx = np.gradient(f, dy, dx)
        return df_dx, df_dy

    # First derivatives
    dE_dx, dE_dy = grad_xy(phiE)
    dB_dx, dB_dy = grad_xy(phiB)

    # --------------------------------------------------
    # Angular derivatives
    # --------------------------------------------------
    def d_theta(d_dx, d_dy):
        return cosphi * d_dy + sinphi * d_dx

    def d_phi(d_dx, d_dy):
        return -theta * sinphi * d_dy + theta * cosphi * d_dx

    dE_dtheta = d_theta(dE_dx, dE_dy)
    dE_dphi   = d_phi(dE_dx, dE_dy)

    dB_dtheta = d_theta(dB_dx, dB_dy)
    dB_dphi   = d_phi(dB_dx, dB_dy)

    # --------------------------------------------------
    # Second derivatives
    # --------------------------------------------------
    # theta-theta
    d2E_dtheta2 = d_theta(*grad_xy(dE_dtheta))
    d2B_dtheta2 = d_theta(*grad_xy(dB_dtheta))

    # phi-phi
    d2E_dphi2 = d_phi(*grad_xy(dE_dphi))
    d2B_dphi2 = d_phi(*grad_xy(dB_dphi))

    # mixed
    d2E_dtheta_dphi = d_theta(*grad_xy(dE_dphi))
    d2B_dtheta_dphi = d_theta(*grad_xy(dB_dphi))

    # --------------------------------------------------
    # Spin-2 operator
    # --------------------------------------------------
    # Real part (E-like)
    ReE = (
        d2E_dtheta2
        - cot_theta * dE_dtheta
        - d2E_dphi2 / sin2_theta
    )

    ReB = (
        d2B_dtheta2
        - cot_theta * dB_dtheta
        - d2B_dphi2 / sin2_theta
    )

    # Imaginary part (B-like)
    ImE = 2.0 * d2E_dtheta_dphi / sin_theta
    ImB = 2.0 * d2B_dtheta_dphi / sin_theta

    # --------------------------------------------------
    # Q, U
    # --------------------------------------------------
    Q = -ReE + ImB
    U = -ImE - ReB

    return Q, U


def power_spectrum_flat(cl, nx, ny, dx, dy):


    lx,ly=np.meshgrid( np.fft.fftfreq( nx, dx )[0:int(nx/2+1)]*2.*np.pi,np.fft.fftfreq( ny, dy )*2.*np.pi )
    #lx,ly=np.meshgrid( np.fft.fftfreq( nx, 1/float(nx) ),np.fft.fftfreq( nx, 1/float(nx) ) )
    l = np.sqrt(lx**2 + ly**2)
    ell_flat = l.flatten()#*72.

    ell_ql = np.arange(0,cl.shape[0])

    cl_plane = np.copy(cl)
    cl_plane[0:2] = cl[2]
    interp = scipy.interpolate.interp1d(np.log(ell_ql[1:]),np.log(cl_plane[1:]), kind='linear',fill_value=0,bounds_error=False)
    cl_plane = np.zeros(ell_flat.shape[0])
    cl_plane[1:] = np.exp(interp(np.log(ell_flat[1:])))
    cl_plane[0] = cl[1]


    return ell_flat, cl_plane

def power_average(ell_flat, cl_flat):

    ell_unique = np.unique(ell_flat)
    cl_mean = np.zeros(len(ell_unique))
    index_list = []
    for i in range(len(ell_unique)):
        index = np.where((ell_flat == ell_unique[i]))[0]
        cl_mean[i] = np.mean(cl_flat[index])
        index_list.append(index)

    return cl_mean, index_list

def calculate_res(z, nx, ny):

    res_x = (z[:, 0].max() - z[:, 0].min()) / nx
    res_y = (z[:, 1].max() - z[:, 1].min()) / ny
    res = (res_x + res_y) / 2

    return res_x, res_y, res

def calculate_res_margin(z, dx, dy, nx, ny, fact=4):

    margin = fact*max(dx, dy)
    res_x = (z[:, 0].max() - z[:, 0].min() + 2*margin) / (nx-1)
    res_y = (z[:, 1].max() - z[:, 1].min() + 2*margin) / (ny-1)
    res = (res_x + res_y) / 2

    return res_x, res_y, res

def calculate_res_margin_binning(z, dx, dy, nx, ny, fact=4):

    margin = fact*max(dx, dy)
    res_x = (z[:, 0].max() - z[:, 0].min() + 2*margin) / (nx)
    res_y = (z[:, 1].max() - z[:, 1].min() + 2*margin) / (ny)
    res = (res_x + res_y) / 2

    return res_x, res_y, res

def calculate_res_margin_deg(z, nx, ny, margin_deg):

    margin = np.deg2rad(margin_deg)

    res_x = (
        z[:,0].max() - z[:,0].min() + 2*margin
    ) / (nx - 1)

    res_y = (
        z[:,1].max() - z[:,1].min() + 2*margin
    ) / (ny - 1)

    res = 0.5 * (res_x + res_y)

    return res_x, res_y, res


def compute_pixel_window(nx, dx):

    kx_grid,ky_grid=np.meshgrid( np.fft.fftfreq( nx, dx )[0:int(nx/2+1)]*2.*np.pi,np.fft.fftfreq( nx, dx )*2.*np.pi )

    k = np.sqrt(kx_grid**2 + ky_grid**2)

    # evitar división por cero
    sinc = lambda x: np.sinc(x/np.pi)

    Wx = sinc(kx_grid * dx / 2)
    Wy = sinc(ky_grid * dx / 2)

    W2 = (Wx**2) * (Wy**2)

    ell_flat = k.flatten()
    W2_flat = W2.flatten()

    return ell_flat, W2_flat

def cl_nifty(map_grid, npix_grid, pixel_size):
    
    s_space = ift.RGSpace([npix_grid,npix_grid])
    h_space = s_space.get_default_codomain()
    HT = ift.HartleyOperator(h_space,s_space)
    size_map = npix_grid*pixel_size
    kfun=2.0*np.pi/size_map
    grid_ift = ift.Field.from_raw(s_space,np.asarray(map_grid))
    power_grid = ift.power_analyze(HT.inverse(grid_ift)).val
    kvals=ift.PowerSpace(h_space).k_lengths
    cls_grid=power_grid*(size_map)**2
    ls=kvals*kfun

    return ls, cls_grid



def powerauto(map, nx, ny, dx, dy):
    tfac = np.sqrt((dx*dy)/(nx*ny))
    fft = np.fft.rfft2(map[:,:])*tfac
    fft_shape = fft.shape
    power = np.real(fft*np.conj(fft))
    power_reshape = np.reshape(power, [fft_shape[0]*fft_shape[1]]) #coef de fourier mapa
    return power_reshape 


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx   #,array[idx]


def compute_bins(lmin, lmax, Nbins):

    ls = np.arange(lmax+1)

    num_modes = np.zeros(lmax+1)
    cumulative_num_modes = np.zeros(lmax+1)

    bin_edges = np.zeros(Nbins+1)
    bin_edges[0] = lmin
    
    cumulative = 0
    for i in range(lmin,lmax+1):
        num_modes[i] = 2*i +1
        
        cumulative += num_modes[i]
        cumulative_num_modes[i] = cumulative

            
    Num_modes_total = num_modes.sum()
    print("Total number of modes in (l_min,l_max) = ", Num_modes_total)   
    Num_modes_per_bin = Num_modes_total / Nbins
    print("Number of modes in each bin = ", Num_modes_per_bin)

    for i in range(1,Nbins+1):
        
        #Num_modes_per_bin*i #cumulative modes up to bin "i"
        
        bin_edges[i] = find_nearest(cumulative_num_modes, Num_modes_per_bin*i)
    
    bin_edges = np.asarray(bin_edges,int)
   
    return Num_modes_per_bin, cumulative_num_modes, bin_edges


def compute_bins_primordial(lmin=2, lmax=300):

    # -------- Reionización --------
    #edges_reion = np.arange(lmin, 21, 5)

    # -------- Recombination bump --------
    edges_recomb = np.arange(lmin, 201, 20)  

    # -------- Lensing tail --------
    edges_lensing = np.arange(200, lmax+1, 50)  # bins grandes

    # combinar evitando duplicados
    #edges = np.unique(
    #    np.concatenate([edges_reion,
    #                    edges_recomb,
    #                    edges_lensing])
    #)

    edges = np.unique(
        np.concatenate([edges_recomb,
                        edges_lensing])
    )

    return edges

def compute_bins_uniform(lmin=2, lmax=700, delta_ell=20):
    """
    Binning uniforme  con ancho constante delta_ell.
    """
    edges = np.arange(lmin, lmax + delta_ell, delta_ell)
    return edges



def compute_bins_hybrid(lmin, lmax, Nbins, l_transition=100):

    # -------- Parte lineal --------
    # usamos un bin por multipolo hasta l_transition
    linear_edges = np.arange(lmin, l_transition + 1, 4)
    #print(linear_edges)
    
    # número de bins ya usados
    N_linear = len(linear_edges) - 1
    
    # bins restantes
    N_remaining = Nbins - N_linear

    if N_remaining <= 0:
        raise ValueError("Nbins demasiado chico para la parte lineal")

    # -------- Parte mismo numero de modos --------
    _,_, equal_edges = compute_bins(l_transition + 1, lmax, N_remaining)

    #print(equal_edges)
    # combinamos evitando repetir l_transition
    bin_edges = np.concatenate([
        linear_edges,
        equal_edges[1:]
    ])

    # asegurar unicidad y orden
    bin_edges = np.unique(bin_edges)

    return bin_edges


def compute_bins_fractional(lmin=2, lmax=700, frac=0.1, min_width=10, extra_edges=None):
    edges = [lmin]
    ell = float(lmin)

    while ell < lmax:
        delta = max(frac * ell, min_width)
        ell = ell + delta
        edges.append(ell)

    edges = np.array(edges)
    if extra_edges is not None:
        edges = np.concatenate([edges, extra_edges])
    edges = np.unique(np.round(edges).astype(int))

    return edges

def compute_bins_physical(
    lmin=2,
    lsplit=120,
    lmax=700,
    low_width=4,
    frac=0.1,
    min_width_high=10):

    edges = [lmin]

    ell = float(lmin)

    while ell < lsplit:

        ell += low_width
        edges.append(ell)

    while ell < lmax:

        delta = max(frac * ell, min_width_high)

        ell += delta
        edges.append(ell)

    edges = np.array(edges)

    edges = np.unique(np.round(edges).astype(int))

    return edges


def bineado(ell, cl, bins): 
 
    bin_indices = np.digitize(ell, bins, right=False)

    # Initialize arrays to store results
    count = np.zeros(len(bins) - 1)
    cl_bin_sum = np.zeros(len(bins) - 1)
    el_med_sum = np.zeros(len(bins) - 1)


    # Calculate sum of ell values and cl values for each bin
    for i in range(1, len(bins)):
        mask = (bin_indices == i)
        count[i - 1] = np.sum(mask)
        cl_bin_sum[i - 1] = np.sum(cl[mask])
        el_med_sum[i - 1] = np.sum(ell[mask])# * cl[mask])

    # Calculate the binned results
    #el_med = (el_med_sum / cl_bin_sum).astype(int)
    el_med = (el_med_sum/count).astype(int)
    cl_bin = cl_bin_sum / count

    return el_med, cl_bin, count

def bineado_namaster_like(ell, cl, bins):

    cl_bin = []
    ell_eff = []

    for i in range(len(bins)-1):

        mask = (ell >= bins[i]) & (ell < bins[i+1])

        l = ell[mask]
        c = cl[mask]

        w = 2*l + 1

        clb = np.sum(w * c) / np.sum(w)
        leff = np.sum(w * l) / np.sum(w)

        cl_bin.append(clb)
        ell_eff.append(leff)

    return np.array(ell_eff), np.array(cl_bin)


def make_ell_grid(nx, dx):
    # devuelve ell_grid numpy shape (nx, nx//2+1)
    fx = np.fft.fftfreq(nx, d=dx)           # (nx,)
    fy = np.fft.rfftfreq(nx, d=dx)          # (nx//2+1,)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')  # (nx, nx//2+1)
    ell = 2.0 * np.pi * np.sqrt(FX**2 + FY**2)
    return ell

def make_lowpass_mask(nx, dx, ell_cut=1500, smooth_width=0.0, device='cpu', dtype=torch.float32):
    """
    smooth_width = 0.0 -> hard cutoff (ell <= ell_cut)
    smooth_width > 0.0 -> smooth roll-off over [ell_cut - smooth_width/2, ell_cut + smooth_width/2]
    returns torch tensor mask shape (1,1,nx,nx//2+1) on device
    """
    ell = make_ell_grid(nx, dx)  # numpy
    ell_t = torch.tensor(ell, device=device, dtype=dtype)  # (nx, nx//2+1)

    if smooth_width is None or smooth_width <= 0.0:
        mask = (ell_t <= ell_cut).to(dtype)
    else:
        l1 = ell_cut - smooth_width/2.0
        l2 = ell_cut + smooth_width/2.0
        mask = torch.ones_like(ell_t)
        above = ell_t >= l2
        below = ell_t <= l1
        between = (~above) & (~below)
        mask[above] = 0.0
        # raised-cosine in the transition
        mask[between] = 0.5 * (1.0 + torch.cos(np.pi * (ell_t[between] - l1) / (l2 - l1)))
        mask = mask.to(dtype)

    return mask.unsqueeze(0).unsqueeze(0)  # shape (1,1,nx,nx//2+1)

def lowpass_batch_torch(maps, dx, ell_cut=1500, smooth_width=0.0):
    """
    maps: torch.Tensor shape (B, C, nx, nx)  (C can be 1)
    dx: pixel size in radians (float)
    ell_cut: cutoff in ell
    smooth_width: width of transition (0 -> hard cutoff). Recommend 100-300 for smooth.
    returns: filtered maps same shape (B, C, nx, nx), real dtype same as input
    """
    assert maps.ndim == 4
    B, C, nx, _ = maps.shape
    device = maps.device
    dtype = maps.dtype

    # create mask once
    mask = make_lowpass_mask(nx, dx, ell_cut=ell_cut, smooth_width=smooth_width, device=device, dtype=dtype)
    # mask shape (1,1,nx,nx//2+1) -> will broadcast to (B,C,nx,nx//2+1)

    # FFT
    # torch.fft.rfft2 handles batch and channel dims
    Tfft = torch.fft.rfft2(maps)  # complex tensor shape (B,C,nx,nx//2+1)

    # apply mask (real multiplier)
    Tfft_filtered = Tfft * mask  # broadcasts on (1,1,...)

    # inverse FFT
    maps_filtered = torch.fft.irfft2(Tfft_filtered, s=(nx, nx))  # returns real (float) tensor

    return maps_filtered

def use_namaster(mask, map1, map2):

    mask_apod = nmt.mask_apodization(mask, aposize=0.5, apotype="Smooth")


    f1 = nmt.NmtField(mask_apod, [mask_apod*map1])
    f2 = nmt.NmtField(mask_apod, [mask_apod*map2])

    #b = nmt.NmtBin.from_edges(bin_edges[:-1], bin_edges[1:])
    b = nmt.NmtBin.from_nside_linear(nside, 100)
    ells_uncoupled = b.get_effective_ells()
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f1, f2, b)

# calcular pseudo-Cl
    cl_pseudo = nmt.compute_coupled_cell(f1, f2)

# decoupled Cl
    cl_decoupled = w.decouple_cell(cl_pseudo)

    return cl_decoupled, ells_uncoupled


def sph_to_cart(theta, phi):
    """theta: colatitude (0..pi), phi: longitude (radians)"""
    st = np.sin(theta); ct = np.cos(theta)
    cp = np.cos(phi); sp = np.sin(phi)
    x = st * cp
    y = st * sp
    z = ct
    return np.stack([x, y, z], axis=-1)  # shape (...,3)

def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return v / n

def rotate_QU_gnomonic(theta, phi, theta0, phi0):
    """
    Q,U : 2D arrays of same shape (maps on sphere patch)
    theta, phi : 2D arrays of pixel coords in radians (colatitude, longitude)
    theta0, phi0 : scalars - centre of gnomonic projection (colat, lon) in radians
    returns: Q_map, U_map rotated to projection plane basis
    """

    th = theta.ravel()
    ph = phi.ravel()

    # unit vectors of each point on sphere
    r = sph_to_cart(th, ph)           # (...,3)
    r0 = sph_to_cart(theta0, phi0)[0] # (3,)

    # tangent-plane basis (fixed) at projection center:
    # choose x_axis = "weast at center", y_axis = "north at center"
    # weast at center:
    ex_center = np.array([np.sin(phi0), -np.cos(phi0), np.zeros_like(phi0)])[:,0]
    ex_center = unit(ex_center)
    ey_center = np.cross(r0, ex_center)   # north at center in tangent plane
    ey_center = unit(ey_center)

    # local "north" and "east" unit vectors at each pixel (on sphere)
    # using spherical coordinate unit vectors:
    # e_theta = d r / d theta  (points toward increasing theta = south), 
    # so local north = -e_theta_unit
    e_theta = np.stack([
        np.cos(th) * np.cos(ph),
        np.cos(th) * np.sin(ph),
        -np.sin(th)
    ], axis=-1)   # not unit, but its norm = 1
    e_theta = unit(e_theta)
    local_north = -e_theta  # unit vector pointing toward local north

    # e_phi (not normalized): d r / d phi = [-sinθ sinφ, sinθ cosφ, 0]
    e_phi = np.stack([
        -np.sin(th) * np.sin(ph),
         np.sin(th) * np.cos(ph),
         np.zeros_like(th)
    ], axis=-1)
    # unit east:
    local_east = -unit(e_phi)

    e_phi = np.stack([
        -np.sin(th) * np.sin(ph),
         np.sin(th) * np.cos(ph),
         np.zeros_like(th)
    ], axis=-1)
    # unit east:
    local_east = -unit(e_phi)

    # project local_north onto projection plane basis (ex_center, ey_center)
    # coordinates in projection-basis:
    a = np.dot(local_north, ex_center)  # component along x
    b = np.dot(local_north, ey_center)  # component along y

    # angle psi: angle from map y-axis (north on plane) to local north, 
    # measured towards +x (weast)
    # i.e. local_north in plane has coords (a,b) and psi = atan2(a, b)
    psi = np.arctan2(a, b)  # shape (Npix,)

    return psi
