import os
import argparse
import optuna
import torch
import pdb
import numpy as np
import random
import os.path as osp
import logging
import time
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from torch import nn, optim, utils
import torch.nn as nn
import models.BiMamba4RUL 
import models.BiMamba4RULplus
import models.TSMamba
# from tensorboardX import SummaryWriter
from sklearn import preprocessing
from torch.utils.data import DataLoader
import rulutils
import torch.nn.functional as F
logging.basicConfig(level=logging.WARNING)
def build_dir(name):
    model_dir = osp.join("./experiments",name)
    os.makedirs(model_dir,exist_ok=True)
    return model_dir
class RULDataset(Dataset):
    def __init__(self,x_train,y_train):
        self.x=x_train
        self.y=y_train
        self.len = len(self.x)
    def __getitem__(self, index):
        history=np.array(self.x[index])
        # print("history shape:",history.shape)
        # future=np.array(self.y[index])
        # print("future shape:",future.shape)
        
        # data=np.concatenate((history,future), axis=0)
        # history=torch.Tensor(history)
        # future=torch.Tensor(future)
        # return self.x[index],self.y[index]
        # print(self.y[index].shape)
        return history,self.y[index]
    def __len__(self):
        return self.len
def train_model(params,datasetname, train_loader,val_loader,test_loader, x_train,x_val):
    # Define the model based on params
    model = models.TSMamba.Model(
        num_scales=params['num_scales'],
        seq_len=params['sequence_length'],
        ch_ind=1,
        stride=params['sequence_length'],
        patch_len=params['sequence_length'],
        d_model=params['d_model'],
        d_state=8,
        d_conv=2,
        e_fact=1,
        d_ff=params['d_ff'],
        dropout=0.1,
        bi_dir=1,
        residual=1,
        e_layers=2,
        enc_in=1,
        sp_num=14,
        sp_d_model=params['sp_d_model']
    )
    # Training and validation process
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    model = model.cuda()
    patience=20
    patience_c=0
    minavgvaloss = float('inf')
    best_val_loss = float('inf')
    model_dir=r"./checkpoint"
    for epoch in range(0,100):
        model.train() 
        sumvalloss=0
        sumtrainloss=0
        for i,(history,rul) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model.forward(history)
            train_loss = model.loss_function(output,rul)
            train_loss.backward()
            optimizer.step()                                                                                                                                                                                                                                        
            sumtrainloss+=train_loss.item()*rul.size()[0]
        sumtrainloss = sumtrainloss / x_train.shape[0]
        model.eval()
        with torch.no_grad():
            for i,(history,rul) in enumerate(val_loader):
                prediction = model.forward(history)
                val_loss = model.loss_function(prediction,rul)
                sumvalloss += val_loss.item()*rul.size(0)
            sumvalloss = sumvalloss / x_val.shape[0]
            patience_c+=1
            if(sumvalloss<minavgvaloss):
                patience_c=0
                torch.save(model.state_dict(),osp.join(model_dir,f"{datasetname}.pt"))
                minavgvaloss=sumvalloss
            if patience_c>patience:
                    break
            torch.cuda.empty_cache()
    print(f"best model found with val_loss: {minavgvaloss:.5f}")
    model.eval()    
    test_rmse = 0.0
    test_score = 0.0
    model.load_state_dict(torch.load(osp.join(model_dir,f"{datasetname}.pt"), weights_only=True))
    with torch.no_grad():
        for i, (history, rul) in enumerate(test_loader):
            history, rul = history.cuda(), rul.cuda()  # Move both tensors to CUDA
            output = model.forward(history)  # Forward pass with GPU data
            rmse_loss = torch.sqrt(F.mse_loss(output[0], rul[0], reduction='mean'))
            temp=rulutils.score(rul.cpu().detach().numpy(),output.cpu().detach().numpy())
            # print(f"no.{i} socre : {temp}")
            test_score += temp[0]
            test_rmse += rmse_loss.item()
        test_rmse /= len(test_loader)
    print(f"Test score: {test_score:.5f} Test rmse:{test_rmse:.5f}")
    return test_score
def objective(trial):
    # sequence_length = 32
    params = {
        'sequence_length': trial.suggest_categorical('sequence_length', [i*16 for i in range(2, 4)]),
        'num_scales': trial.suggest_categorical('num_scales', [i for i in range(1, 5)]),  # [1, 2, 4,8]
        'd_model': trial.suggest_categorical('d_model', [2**i for i in range(5, 10)]),  # [64, 128, 256, 512]
        'd_ff': trial.suggest_categorical('d_ff', [2**i for i in range(5, 10)]),  # [128, 256, 512, 1024]
          # Keep as float, not 2's power
        'sp_d_model': trial.suggest_categorical('sp_d_model', [2**i for i in range(5, 10)]),  # [64, 128, 256, 512]  # [64, 128]
         # Keep as float, not 2's power
    }
    # Ensure d_model is even
    datasetname="FD002"
    sensors = ['s_2','s_3', 's_4', 's_7','s_8','s_9','s_11', 's_12','s_13','s_14','s_15','s_17','s_20','s_21']
    alpha = 0.1
    threshold = 120
    # Create data loaders
    # model.load_state_dict(torch.load('.//checkpoint//FD001_epoch'+str(epoch)+'.pt'))
    x_train, y_train, x_val, y_val, x_test, y_test = rulutils.get_data(datasetname, sensors, params['sequence_length'], alpha, threshold)
    train_dataset = RULDataset(x_train,y_train)
    val_dataset = RULDataset(x_val,y_val)
    test_dataset = RULDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=48, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    val_loss = train_model(params,datasetname, train_loader,val_loader,test_loader,x_train,x_val)
    return val_loss

if __name__ == '__main__':
    datasetname="FD002"
    study = optuna.create_study(study_name=f"{datasetname}_study", storage=f"sqlite:///{datasetname}_study.db", load_if_exists=True,direction='minimize')
    study.optimize(objective, n_trials=150)
    print("Best parameters:", study.best_params)