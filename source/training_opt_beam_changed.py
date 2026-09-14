
import numpy as np
import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.nn as nn #provides all the building blocks to build the neural network
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torchvision import models #just for debugging
from torchvision import transforms
from torchsummary import summary #just for debugging
import time
#from torch.nn.parallel import DistributedDataParallel as DDP
#import torch.multiprocessing as mp
import pickle
import math, gc
import optuna
from scipy.special import factorial
from scipy.special import gammaln

import losses
from network_2d import DeepWiener_threechannels, DeepWiener_twochannels, DeepWiener_dilation
import projections as pj
import utilities
from scipy.ndimage import gaussian_filter1d
#https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-model-building-6ab09d6a0862

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device", device)


#############################################READ CONFIG#################################################

from config_loader import load_config
from geometry import build_geometry
cfg = load_config()

epochs               = cfg["epochs"]
loss_j3              = cfg["loss_j3"]
nx                   = cfg["npixels_x"]
ny                   = cfg["npixels_y"]
mask_type            = cfg["mask_type"]
noise_type           = cfg["noise_type"]
map_rescale_factor_q = cfg["map_rescale_factor"]
fwhm_rad             = cfg["fwhm_rad"]
noise_pix            = cfg["noise_pix"]
name_train           = cfg["name_train"]
name_valid           = cfg["name_valid"]
model_path           = cfg["model_path"]
loss_path            = cfg["loss_path"]
study_name           = cfg["study_name"]

cltt, clee, clbb = utilities.signal_spectrum(r=cfg["r"])
ele = np.arange(len(clee))

# plane resolution (dx, dy) of the projected grid, same set-up as the dataset
# generation (geometry.build_geometry: circ -> dx = dy from calculate_res_margin
# with nx x nx; rect -> dx, dy from calculate_res_margin with nx x ny)
geo = build_geometry(cfg, load_inho2d=False, variance=False)
dx, dy = geo.dx, geo.dy

clee = (clee[:]) * map_rescale_factor_q**2
clbb = (clbb[:]) * map_rescale_factor_q**2
ell_flat, clee_flat = utilities.power_spectrum_flat(clee, nx, ny, dx, dy)
ell_flat, clbb_flat = utilities.power_spectrum_flat(clbb, nx, ny, dx, dy)
cl_flat = [clee_flat, clbb_flat]


######################### DATASET #############################################

class CMBDataset(Dataset):
    def __init__(self, inputs, targets):
        self.X = inputs # (N, 1, ngrid, ngrid)
        self.Y = targets # (N, 1, ngrid, ngrid)
        #self.Y = torch.stack(data["Y"])*map_rescale_factor # (N, npix) # para caso esférico

    def __getitem__(self, i):
        return self.X[i], self.Y[i]
    def __len__(self):
        return len(self.Y)


def create_dataloader(filename, noise_type, shuffle=True, batch_size=1):

    if noise_type == "inho":
        data_array = torch.load(filename)
        qobs = data_array[:,0:1,:,:]#*map_rescale_factor
        uobs = data_array[:,1:2,:,:]#*map_rescale_factor
        mask = data_array[:,2:3,:,:]
        inho = data_array[:,3:4,:,:]

        qobs_norm = (qobs - qobs.mean(dim=(2, 3), keepdim=True))*map_rescale_factor_q
        uobs_norm = (uobs - uobs.mean(dim=(2, 3), keepdim=True))*map_rescale_factor_q
        inho_norm = inho#/torch.max(inho)

        del data_array

        obs_norm = torch.cat((qobs_norm, uobs_norm, mask, inho_norm), dim=1)
        target_norm = torch.cat((qobs_norm, uobs_norm), dim=1)

        dataset = CMBDataset(obs_norm, target_norm)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            )

        return dataloader, mask[0], inho[0]

    elif noise_type == "ho":
        data_array = torch.load(filename)
        qobs = data_array[:,0:1,:,:]#*map_rescale_factor
        uobs = data_array[:,1:2,:,:]#*map_rescale_factor
        mask = data_array[:,2:3,:,:]

        qobs_norm = (qobs - qobs.mean(dim=(2, 3), keepdim=True))*map_rescale_factor_q
        uobs_norm = (uobs - uobs.mean(dim=(2, 3), keepdim=True))*map_rescale_factor_q

        del data_array

        obs_norm = torch.cat((qobs_norm, uobs_norm, mask), dim=1)
        target_norm = torch.cat((qobs_norm, uobs_norm), dim=1)

        dataset = CMBDataset(obs_norm, target_norm)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            )

        return dataloader, mask[0], None       

train_loader, mask, inho = create_dataloader(name_train, noise_type)
valid_loader, _, _ = create_dataloader(name_valid, noise_type)


################################################ TRAINING #########################################
def hyper(hyperparameters):
    return "n_filters0_" + str(hyperparameters[0]) +  "_n_filters1_" + str(hyperparameters[1]) + "_n_filters2_" + str(hyperparameters[2]) + "_n_filters3_" + "_lr_" + "{:.3e}".format(hyperparameters[3]) + "_wd_" + "{:.3e}".format(hyperparameters[4])# + "_lambda_" + "{:.2e}".format(hyperparameters[7])


def train(train_loader, model, optimizer, criterion, scheduler):

    train_loss = 0.0
    model.train()

    for inputs, targets in train_loader:

        inputs = inputs.to(device)
        targets = targets.to(device)
        #print('input', inputs.shape)
        optimizer.zero_grad()  #clear the gradients
        pred = model(inputs) #make a forward pass

        loss_wf = criterion(targets, pred)        
        loss_wf.backward()  #perform a backward pass to calculate the gradients

        optimizer.step()#optimizer step to update the weights
        scheduler.step()
        train_loss += loss_wf.item()

    last_loss = train_loss/len(train_loader)
    return last_loss

def eval(valid_loader, model, optimizer, criterion, min_valid_loss, hyperparameters):

    valid_loss = 0.0 
    model.eval()
    for data, labels in valid_loader:
        with torch.no_grad():

            data = data.to(device)
            labels = labels.to(device)
            pred = model(data)
            loss_wf = criterion(labels, pred)#, nx, dx, cltt_flat, ell_flat, device)
            valid_loss += loss_wf.item()

    val_loss = valid_loss/len(valid_loader)

    if val_loss < min_valid_loss:
        min_valid_loss = val_loss
        # Saving State Dict
        print('Best model, saving...')
        torch.save({'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict()}, 
            model_path + hyper(hyperparameters))

    return val_loss, min_valid_loss


def objective(trial):

    filters0 = trial.suggest_int("filters0", 8, 32, step=8)
    filters1 = trial.suggest_int("filters1", filters0, 64, step=8)
    filters2 = trial.suggest_int("filters2", filters1, 128, step=8)
    filters3 = trial.suggest_int("filters3", filters2, 192, step=8)
    filters4 = filters3
    filters5 = filters4
    filters = [filters0, filters1, filters2, filters3, filters4, filters5]
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    wd = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
    #lambda_weight = trial.suggest_float("lambda_weight", 0.1, 10.0, log=True)

    #kernel_size = 5 # 3
    #padding = 2 # 1

    in_channels = 2
    out_channels = 2
    if noise_type == "ho":
        model = DeepWiener_twochannels(in_channels, out_channels, filters)
    elif noise_type == "inho":
        model = DeepWiener_threechannels(in_channels, out_channels, filters)
    model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=wd)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr*3, total_steps=total_steps, pct_start=0.3, anneal_strategy='cos')
    #scheduler = torch.optim.lr_scheduler.CyclicLR(
    #    optimizer, base_lr=lr, max_lr=1.e-3, cycle_momentum=False, step_size_up=1000
    #)

    if loss_j3:
        if noise_type == "ho":
            def criterion(y_true, y_pred):
                return losses.lossj3_plane_beam(
                    y_true, y_pred, mask, nx, ny, dx, dy,
                    cl_flat, ell_flat, map_rescale_factor_q, fwhm_rad, noise_pix, device)
        elif noise_type == "inho":
            def criterion(y_true, y_pred):
                return losses.lossj3_plane_beam_inho(
                    y_true, y_pred, mask, inho, nx, ny, dx, dy,
                    cl_flat, ell_flat, map_rescale_factor_q, fwhm_rad, device)
    else:
        criterion = losses.lossj2_plane

    hyperparameters = [filters0, filters1, filters2, filters3, lr, wd]#, lambda_weight]

    trainLoss_history = []
    validLoss_history = []
    min_valid_loss = 1e7    

    for epoch in range(epochs):

        print('epoch:', epoch)

        t0 = time.time()

        train_loss = train(train_loader, model, optimizer, criterion, scheduler)

        t1 = time.time()
        print('trained', t1-t0)

        if(math.isnan(train_loss)):
            return 10000

        valid_loss, min_valid_loss = eval(valid_loader, model, optimizer, criterion, min_valid_loss, hyperparameters)

            
        print(f"train loss {train_loss}, valid loss {valid_loss}", flush = True)

        trainLoss_history.append(train_loss)
        validLoss_history.append(valid_loss)


    np.savez(loss_path + hyper(hyperparameters), trainLoss_history, validLoss_history)

    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()

    return min_valid_loss


if __name__ == '__main__':

    study = optuna.create_study(
    study_name=cfg["study_name"],
    storage=cfg["study_db"],
    direction="minimize",
    load_if_exists=True
    )
    study.optimize(objective, n_trials=10)  # podes poner más trials

    print("Best trial:")
    trial = study.best_trial
    print("value:", trial.value)
    
    print("Params:")
    for key, value in trial.params.items():
        print("{}:{}".format(key, value))
















							  




