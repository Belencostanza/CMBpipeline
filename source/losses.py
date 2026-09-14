#funcion loss j3, no la puedo programar con quicklens ya que no podria usar tensorflow gpu

import numpy as np
import sys
import os
import scipy.interpolate

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn #provides all the building blocks to build the neural network
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models #just for debugging
from torchvision import transforms
#from torchsummary import summary #just for debugging

from scipy.interpolate import RegularGridInterpolator

#import pickle
import utilities



def lossj2_plane(y_true, y_pred, device, temp=True):

    y_true_Q = y_true[:,0,:,:].to(device)
    y_true_U = y_true[:,1,:,:].to(device)
    y_pred_Q = y_pred[:,0:,:].to(device)
    y_pred_U = y_pred[:,1:,:].to(device)

    loss_Q = (y_pred_Q - y_true_Q)*(y_pred_Q-y_true_Q)
    loss_U = (y_pred_U - y_true_U)*(y_pred_U-y_true_U)

    loss = torch.mean(loss_Q) + torch.mean(loss_U)

    return loss

def beam_filter(map, beam, nx, ny, dx, dy, device):

    beam = torch.tensor(beam, dtype=torch.float32, device=device)
    tfac = np.sqrt((dx*dy)/(nx*ny))
    fft = torch.fft.rfft2(map)*tfac
    fft_beam = beam*fft
    beam_map = torch.fft.irfft2(fft_beam)/tfac
    
    return beam_map

def realspace_loss(y_true, y_pred):

    y_obs_Q = y_true[:,0,:,:].to(device)
    y_pred_Q = y_pred[:,0,:,:].to(device)
    y_obs_U = y_true[:,1,:,:].to(device)
    y_pred_U = y_pred[:,1,:,:].to(device)

    
    loss_Q = (y_obs_Q - y_pred_Q)*(y_obs_Q-y_pred_Q)/(noise_pix*map_rescale_factor_q**2)
    loss_U = (y_obs_U - y_pred_U)*(y_obs_U-y_pred_U)/(noise_pix*map_rescale_factor_q**2)

    return torch.mean(loss_Q) + torch.mean(loss_U)

def realspace_loss_beam_corr(y_true, y_pred, mask, nl_flat_w2, ell_flat, nx, ny, dx, dy, factor, fwhm, device):

    beam = utilities.beam_flat(fwhm, ell_flat)
    y_obs_Q = y_true[:,0,:,:]#.to(device)
    y_pred_Q = y_pred[:,0,:,:]#.to(device)
    y_obs_U = y_true[:,1,:,:]#.to(device)
    y_pred_U = y_pred[:,1,:,:]#.to(device)
    mask = mask.to(device)

    nl_flat_w2 = torch.tensor(nl_flat_w2, dtype=torch.float32, device=device)
    epsilon = 1e-8 * torch.max(nl_flat_w2)
    nl_flat_w2 = torch.clamp(nl_flat_w2, min=epsilon)
    inv_noise = 1.0 / nl_flat_w2
    inv_noise = inv_noise.reshape(ny, nx//2+1)

    y_pred_Qf = beam_filter(y_pred_Q, beam.reshape(ny, nx//2+1), nx, ny, dx, dy, device)
    y_pred_Uf = beam_filter(y_pred_U, beam.reshape(ny, nx//2+1), nx, ny, dx, dy, device)

    res_Q = mask*(y_obs_Q - y_pred_Qf)
    res_U = mask*(y_obs_U - y_pred_Uf)

    tfac = np.sqrt((dx*dy)/(nx*ny))
    res_Q_fft = torch.fft.rfft2(res_Q)*tfac
    res_U_fft = torch.fft.rfft2(res_U)*tfac

    power_Q = torch.real(res_Q_fft * torch.conj(res_Q_fft))
    power_U = torch.real(res_U_fft * torch.conj(res_U_fft))

    #res_Q_weighted = torch.fft.irfft2(res_Q_fft * inv_noise) / tfac
    #res_U_weighted = torch.fft.irfft2(res_U_fft * inv_noise) / tfac
    
    loss_Q = power_Q*inv_noise
    loss_U = power_U*inv_noise

    return torch.mean(loss_Q) + torch.mean(loss_U)

def realspace_loss_beam(y_true, y_pred, mask, ell_flat, nx, ny, dx, dy, factor, fwhm, noise_pix, device):

    beam = utilities.beam_flat(fwhm, ell_flat)
    y_obs_Q = y_true[:,0,:,:]#.to(device)
    y_pred_Q = y_pred[:,0,:,:]#.to(device)
    y_obs_U = y_true[:,1,:,:]#.to(device)
    y_pred_U = y_pred[:,1,:,:]#.to(device)
    mask = mask.to(device)

    y_pred_Qf = beam_filter(y_pred_Q, beam.reshape(ny, nx//2+1), nx, ny, dx, dy, device)
    y_pred_Uf = beam_filter(y_pred_U, beam.reshape(ny, nx//2+1), nx, ny, dx, dy, device)

    diff_Q = (y_obs_Q - y_pred_Qf)**2
    diff_U = (y_obs_U - y_pred_Uf)**2

    #promedio sobe pixeles validos

    #loss_Q = (mask * diff_Q).sum() / (mask.sum() * noise_pix * map_rescale_factor_q**2)
    #loss_U = (mask * diff_U).sum() / (mask.sum() * noise_pix * map_rescale_factor_q**2)

    loss_Q = (mask * diff_Q)/ (noise_pix * factor**2)
    loss_U = (mask * diff_U)/ (noise_pix * factor**2)


    return torch.mean(loss_Q) + torch.mean(loss_U)

def realspace_loss_beam_inho(y_true, y_pred, mask, inho, ell_flat, nx, ny, dx, dy, factor, fwhm, device):

    beam = utilities.beam_flat(fwhm, ell_flat)
    y_obs_Q = y_true[:,0,:,:]#.to(device)
    y_pred_Q = y_pred[:,0,:,:]#.to(device)
    y_obs_U = y_true[:,1,:,:]#.to(device)
    y_pred_U = y_pred[:,1,:,:]#.to(device)
    mask = mask.to(device)
    inho = inho.to(device)

    y_pred_Qf = beam_filter(y_pred_Q , beam.reshape(ny, nx//2+1), nx, ny, dx, dy, device)
    y_pred_Uf = beam_filter(y_pred_U , beam.reshape(ny, nx//2+1), nx, ny, dx, dy, device)

    inv_var = torch.zeros_like(inho)
    valid = (mask > 0) & (inho > 0)
    inv_var[valid] = 1.0 / inho[valid]

    diff_Q = mask*(y_obs_Q - y_pred_Qf)**2
    diff_U = mask*(y_obs_U - y_pred_Uf)**2

    #promedio sobe pixeles validos

    #loss_Q = (mask * diff_Q).sum() / (mask.sum() * noise_pix * map_rescale_factor_q**2)
    #loss_U = (mask * diff_U).sum() / (mask.sum() * noise_pix * map_rescale_factor_q**2)

    loss_Q = (inv_var * diff_Q)/ (factor**2)
    loss_U = (inv_var * diff_U)/ (factor**2)


    return torch.mean(loss_Q) + torch.mean(loss_U)

def realspace_loss_pot(y_true, y_pred, mask, grid_x, grid_y, ell_flat, nx, dx, device, w2d, fwhm):

    beam = utilities.beam_flat(fwhm, ell_flat)
    y_obs_Q = y_true[:,0,:,:]#.to(device)
    y_pred_potE = y_pred[:,0,:,:]*dx**2
    y_obs_U = y_true[:,1,:,:]#.to(device)
    y_pred_potB = y_pred[:,1,:,:]*dx**2
    mask = mask.to(device)

    #print(y_pred_potE)
    phiE_apo = y_pred_potE*w2d
    phiB_apo = y_pred_potB*w2d

    alpha=0.05
    scale = int(alpha * nx)
    mask_inner = torch.zeros((nx, nx), device=device)
    mask_inner[scale:nx-scale, scale:nx-scale] = 1.0

    y_pred_Q, y_pred_U = utilities.eth2_correct(phiE_apo, phiB_apo, nx, dx, grid_x, grid_y)#, map_rescale_factor_q)

    mask_loss = mask * mask_inner

    y_pred_Qf = beam_filter(y_pred_Q * mask_loss, beam.reshape(nx, nx//2+1), nx, dx, device)
    y_pred_Uf = beam_filter(y_pred_U * mask_loss, beam.reshape(nx, nx//2+1), nx, dx, device)

    diff_Q = (y_obs_Q - y_pred_Qf)**2
    diff_U = (y_obs_U - y_pred_Uf)**2

    #promedio sobe pixeles validos

    #loss_Q = (mask * diff_Q).sum() / (mask.sum() * noise_pix * map_rescale_factor_q**2)
    #loss_U = (mask * diff_U).sum() / (mask.sum() * noise_pix * map_rescale_factor_q**2)

    loss_Q = (mask_loss * diff_Q)/ (noise_pix * map_rescale_factor_q**2)
    loss_U = (mask_loss * diff_U)/ (noise_pix * map_rescale_factor_q**2)


    return torch.mean(loss_Q) + torch.mean(loss_U)


def inverse_cl(data, cltt_flat):

    # se divide por el cltt interpolado en el plano
    #data_div[:, 1:] = data[:, 1:] / cltt_flat[1:]
    data_div = data[:, 1:] / cltt_flat[1:]
    #data_div = data/ cltt_flat
    return data_div


def fourier_loss(y_pred, nx, ny, dx, dy, cl_flat, ell_flat, device): 
    
    y_pred_Q = y_pred[:,0,:,:]#.to(device)
    y_pred_U = y_pred[:,1,:,:]#.to(device)

    clee_flat = cl_flat[0]
    clbb_flat = cl_flat[1]

    clee_flat = torch.tensor(clee_flat, dtype=torch.float32, device=device)
    clbb_flat = torch.tensor(clbb_flat, dtype=torch.float32, device=device)

    # calculamos espectro del mapa 
    tfac = np.sqrt((dx * dy) / (nx * ny))
    efft, bfft = utilities.transf_eb2(y_pred_Q, y_pred_U, nx, dx, ny, dy)

    power_e = torch.real((efft * torch.conj(efft)))
    power_b = torch.real((bfft * torch.conj(bfft)))
    #print(np.shape(power_e))

    efft_shape = efft.shape
    #print(efft_shape)
    #print(clee_flat.shape)
    power_e = torch.reshape(power_e,(-1,efft_shape[1]*efft_shape[2]))
    power_b = torch.reshape(power_b,(-1,efft_shape[1]*efft_shape[2]))

    #loss_E = torch.sum(power_e/clee_flat)
    #loss_B = torch.sum(power_b/clbb_flat)

    #Nmodes = power_e.numel()

    #return (loss_E + loss_B)/Nmodes

    ratio_e = inverse_cl(power_e, clee_flat)
    ratio_b = inverse_cl(power_b, clbb_flat)
    
    return torch.mean(ratio_e) + torch.mean(ratio_b)

def fourier_loss_pot(y_pred, grid_x, grid_y, nx, dx, cl_flat, device, w2d): 
   
    L = nx * dx
    dell = 2.0 * np.pi / L
    w_ell = dell**2

    y_pred_potE = y_pred[:,0,:,:]*dx**2#.to(device)
    y_pred_potB = y_pred[:,1,:,:]*dx**2#.to(device)

    phiE_apo = y_pred_potE*w2d
    phiB_apo = y_pred_potB*w2d

    clpotE_flat = cl_flat[0]#[1:]#/(ell_flat[1:]**4)
    clpotB_flat = cl_flat[1]#[1:]#/(ell_flat[1:]**4)

    #CphiE = torch.tensor(clpotE_flat, dtype=torch.float32, device=device)
    #CphiB = torch.tensor(clpotB_flat, dtype=torch.float32, device=device)

    # calculamos espectro del mapa 
    tfac = np.sqrt((dx * dx) / (nx * nx))
    efft, bfft = torch.fft.rfft2(phiE_apo)*tfac, torch.fft.rfft2(phiB_apo)*tfac
    #efft, bfft = utilities.transf_eb(y_pred_Q, y_pred_U, nx, dx)

    #print(efft)
    #PkE = torch.abs(ectfft)**2
    #PkB = torch.abs(bfft)**2

    power_e = torch.real((efft * torch.conj(efft)))
    power_b = torch.real((bfft * torch.conj(bfft)))
    
    efft_shape = efft.shape
    
    power_e = torch.reshape(power_e,(-1,efft_shape[1]*efft_shape[2]))
    power_b = torch.reshape(power_b,(-1,efft_shape[1]*efft_shape[2]))

    #print(y_pred_potE.std(), torch.sqrt(torch.mean(clpotE_flat)))
    #PkE = PkE.reshape(PkE.shape[0], -1)
    #PkB = PkB.reshape(PkB.shape[0], -1)

    CphiE = clpotE_flat[1:]
    CphiB = clpotB_flat[1:]
    PkE   = power_e[:, 1:]
    PkB   = power_b[:, 1:]

    #print(PkE)

    n_modes = PkE.shape[1]
    #print(n_modes)

    eps = 1e-30
    #scaleE = n_modes * torch.mean(1.0 / (CphiE + eps))
    #scaleB = n_modes * torch.mean(1.0 / (CphiB + eps))
    #scale = torch.max(scaleE, scaleB)  # usar máximo para estabilidad
    #print(scale)
    
    loss_E = torch.mean((PkE/CphiE))#/scale
    loss_B = torch.mean((PkB/CphiB))#/scale


    return (loss_E + loss_B)
    #return torch.mean(ratio_e) + torch.mean(ratio_b)

def lossj3_plane(y_true, y_pred, nx, dx, cl_flat, ell_flat, device):

    term1 = realspace_loss(y_true, y_pred)
    term2 = fourier_loss(y_pred, nx, dx, cl_flat, ell_flat, device)
    #print(term1, term2)

    return term1 + term2

def lossj3_plane_beam(y_true, y_pred, mask, nx, ny, dx, dy, cl_flat, ell_flat, factor, fwhm, noise_pix, device):

    term1 = realspace_loss_beam(y_true, y_pred, mask, ell_flat, nx, ny, dx, dy, factor, fwhm, noise_pix, device)
    term2 = fourier_loss(y_pred, nx, ny, dx, dy, cl_flat, ell_flat, device) # el fourier queda igual

    #print('term1', term1)
    #print('term2', term2)

    return term1 + term2

def lossj3_plane_beam_corr(y_true, y_pred, mask, nl_flat_w2, nx, ny, dx, dy, cl_flat, ell_flat, factor, fwhm, device):

    term1 = realspace_loss_beam_corr(y_true, y_pred, mask, nl_flat_w2, ell_flat, nx, ny, dx, dy, factor, fwhm, device)
    term2 = fourier_loss(y_pred, nx, ny, dx, dy, cl_flat, ell_flat, device) # el fourier queda igual

    #print('term1', term1)
    #print('term2', term2)

    return term1 + term2

def lossj3_plane_beam_inho(y_true, y_pred, mask, inho, nx, ny, dx, dy, cl_flat, ell_flat, factor, fwhm, device):

    term1 = realspace_loss_beam_inho(y_true, y_pred, mask, inho, ell_flat, nx, ny, dx, dy, factor, fwhm, device)
    term2 = fourier_loss(y_pred, nx, ny, dx, dy, cl_flat, ell_flat, device) # el fourier queda igual

    #print("term1:", term1, "term2:", term2)

    return term1 + term2


def lossj3_potential(y_true, y_pred, mask, nx, dx, grid_x, grid_y, cl_flat, ell_flat, device, w2d, fwhm):#, epoch, anneal_epochs=30):

    term1 = realspace_loss_pot(y_true, y_pred, mask, grid_x, grid_y, ell_flat, nx, dx, device, w2d, fwhm)
    term2 = fourier_loss_pot(y_pred, grid_x, grid_y, nx, dx, cl_flat, device, w2d) # el fourier queda igual

    #print('term1', term1)
    #print('term2', term2)
    lambda_weight = 1.# min(1.0, epoch / anneal_epochs)
    return term1 + lambda_weight*term2


def deproject_batch(pred_batch, x_axis, y_axis, z):

    """
    pred_batch: torch.Tensor, shape (B, nbins, nbins)
    x_axis, y_axis: 1D arrays con las coordenadas de la grilla en el plano
    z: array de shape (Npix, 2) con coords (x_plane, y_plane) a donde
                querés evaluar en la esfera (por ejemplo los pixeles válidos).

    Devuelve: torch.Tensor shape (B, Npix)
    """
    if isinstance(x_axis, torch.Tensor):
        x_axis = x_axis.detach().cpu().numpy()
    if isinstance(y_axis, torch.Tensor):
        y_axis = y_axis.detach().cpu().numpy()
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()

    B = pred_batch.shape[0]
    N = z.shape[0]
    out = torch.zeros((B, N), dtype=torch.float32, device = pred_batch.device)
    
    # Creamos el interpolador una sola vez por mapa del batch
    for i in range(B):
        interp = RegularGridInterpolator(
            (y_axis, x_axis),
            (pred_batch[i].detach().cpu().numpy()).T,
            method='cubic',
            bounds_error=False,
            fill_value=0.0
        )
        out[i] = torch.from_numpy(interp(z)).to(pred_batch.device)
    return out

        
def lossj2_sph(y_true, y_pred, info, device): 

    """
    y_true: torch.Tensor, shape (B, Npix)
    y_pred: torch.Tensor, shape (B, 1, nbins, nbins)

    y_pred_reproj: torch.Tensor, shape (B, Npix)
    """

    z_inside = info[0]
    x      = info[1][0,:]
    y      = info[2][:,0]

    Npix = len(z_inside)

    #grid = torch.stack((x, y), dim=1)  # (Npix, 2)
    grid = z_inside.view(1, Npix, 1, 2)
    
    values = F.grid_sample(
            y_pred,             # (1,1,512,512)
            grid,             # (1,Npix,1,2)
            mode="bicubic",  # o "bicubic"
            align_corners=True
            )

    y_pred_reproj = values.view(Npix)
    #print('reproj', y_pred_reproj.shape)
    #print('true', y_true.shape)
    #y_pred_reproj = deproject_batch(y_pred.squeeze(1), grid_x, grid_y, z_inside)

    return F.mse_loss(y_pred_reproj.unsqueeze(0), y_true)




    
