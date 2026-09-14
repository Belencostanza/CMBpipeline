import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn as nn #provides all the building blocks to build the neural network
import torch.nn.functional as F
import torch.optim as optim
#from tensorflow import keras #just for downloading dataset
from torchvision import models #just for debugging
from torchvision import transforms
from torchsummary import summary #just for debugging
import sys
import pickle
import projections as pj

#from losses import deproject_batch
from network_2d import DeepWiener_twochannels, DeepWiener_threechannels
import optuna
import utilities
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device", device)

#############################################READ CONFIG#################################################

from config_loader import load_config
cfg = load_config()

mask_type            = cfg["mask_type"]
noise_type           = cfg["noise_type"]
nx                   = cfg["npixels_x"]
ny                   = cfg["npixels_y"]
map_rescale_factor_q = cfg["map_rescale_factor"]
name_valid           = cfg["name_valid"]
model_path           = cfg["model_path"]
result_path          = cfg["result_path"]
study_name           = cfg["study_name"]

# (the sky projection is not needed for inference; see geometry.build_geometry if it is)

study = optuna.load_study(study_name=study_name, storage=cfg["study_db"])   # study_folder/<study_name>.db

trial = study.best_trial

print("\nTrial number {}".format(trial.number), flush=True)
print("Value: %.5e"%trial.value, flush=True)
print(" Params: ", flush=True)
for key, value in trial.params.items():
    print("    {}: {}".format(key, value), flush=True)

filters0 = trial.params["filters0"]
filters1 = trial.params["filters1"]
filters2 = trial.params["filters2"]
filters3 = trial.params["filters3"]
#filters4 = trial.params["filters4"]
filters4 = filters3
filters5 = filters4
lr = trial.params["lr"]
wd = trial.params["wd"]
#lambda_weight = trial.params["lambda_weight"]

hyperparameters = [filters0, filters1, filters2, filters3, lr, wd]#, lambda_weight]
filters = [filters0, filters1, filters2, filters3, filters4, filters5]

resultsname = utilities.hyper(hyperparameters)

######################### DATASET #############################################

#test_loader = create_dataloader(name_valid)
test = torch.load(name_valid)
qobs = (test[:,0:1,:,:]).to(device)
uobs = (test[:,1:2,:,:]).to(device)
mask = (test[:,2:3,:,:]).to(device)
inho = (test[:,3:4,:,:]).to(device)
inho_norm = inho#/torch.max(inho)


qobs_norm = (qobs - qobs.mean(dim=(2, 3), keepdim=True))*map_rescale_factor_q#/true.std()
uobs_norm = (uobs - uobs.mean(dim=(2, 3), keepdim=True))*map_rescale_factor_q#/true.std()

obs_norm = torch.cat((qobs_norm, uobs_norm, mask, inho_norm), dim=1)
#obs_norm = torch.cat((qobs_norm, uobs_norm, mask), dim=1)
print(obs_norm.shape)

####################################### inference ##########################################


in_channels = 2
out_channels = 2
if noise_type == "ho":
    model = DeepWiener_twochannels(in_channels, out_channels, filters=filters)
elif noise_type == "inho":
    model = DeepWiener_threechannels(in_channels, out_channels, filters)
model.load_state_dict(torch.load(model_path + resultsname, map_location=torch.device('cpu'))['model_state_dict'])
model.to(device)

model.eval()
t1 = time.time()
pred = model(obs_norm[0:10])
t2 = time.time()
print(t2-t1)
#phi_pred = pred * (dx**2) #/ map_rescale_factor_q


#torch.save(phi_pred.cpu(), result_path)
torch.save(pred.cpu()/map_rescale_factor_q, result_path)
