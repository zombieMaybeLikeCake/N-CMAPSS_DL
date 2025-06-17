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
import torch.nn.functional as F
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from torch import nn, optim, utils
import torch.nn as nn
import models.BiMamba4RUL 
import models.BiMamba4RULplus
import models.TBiMamba4RUL
import models.Mamba4RUL
# from tensorboardX import SummaryWriter
from sklearn import preprocessing
from torch.utils.data import DataLoader
import rulutils

class RULDataset(Dataset):
    def __init__(self,x_train,y_train):
        self.x=x_train
        self.y=y_train
        self.len = len(self.x)
    def __getitem__(self, index):
        history=np.array(self.x[index])
        return history,self.y[index]
    def __len__(self):
        return self.len
def main():  
    datasetnames=["FD001","FD001","FD001","FD002","FD002","FD002","FD003","FD003","FD003","FD004","FD004","FD004"]
    # datasetnames=["FD003","FD003","FD003","FD004","FD004","FD004"]
    model_dir=r"./checkpoint"
    for datasetname in datasetnames:
        sequence_length=32
        alpha = 0.1
        best_test_loss = float('inf')
        best_test_score = float('inf')
        threshold = 120
        sensors = ['s_2','s_3', 's_4', 's_7','s_8','s_9','s_11', 's_12','s_13','s_14','s_15','s_17','s_20','s_21']
        x_train, y_train, x_val, y_val, x_test, y_test = rulutils.get_data(datasetname, sensors, sequence_length, alpha, threshold)
        train_dataset = RULDataset(x_train,y_train)
        val_dataset = RULDataset(x_val,y_val)
        test_dataset = RULDataset(x_test, y_test)
        mselist=[]
        vallist=[]
        # test_dataset = RULDataset(x_test, y_test)
        epochs=150
        found_start = False
        bsize = 128
        if datasetname=="FD001":
            sp_d_model=256
            num_scales=2
            d_model=256
        elif datasetname=="FD002":
            sp_d_model=128
            num_scales=2
            d_model=128
        elif datasetname=="FD003":
            sp_d_model=256
            num_scales=4
            d_model=1024
            bsize = 32
        elif datasetname=="FD004":
            sp_d_model=256
            num_scales=2
            d_model=128
        train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=bsize,shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        print(f"batch size : {bsize}")
        print(f'Testing configuration: sequence_length={sequence_length}, sp_d_model={sp_d_model}, alpha={alpha}, num_scales={num_scales}, d_model={d_model}')
        # sequence_length = sequence_length  # sequence_length,48,64 3
        # sp_d_model = sp_d_model       # sp_d_model 128 256 512 4
        # alpha = alpha           # alpha 0.3 0.5 3
        # num_scales = num_scales        # num_scales 2 3 4 4 
        # d_model = d_model         # d_model 128 256 512 1024 5
        d_ff = 256            # d_ff 128 256 512 1024 5
        seq_len = sequence_length
        stride = sequence_length
        patch_len = sequence_length
        ch_ind = 1
        d_state = 8
        d_conv = 2
        e_fact = 1
        dropout = 0.1
        bi_dir = 1
        residual = 1
        e_layers = 2
        enc_in = 14
        sp_num = 14
        model=models.BiMamba4RULplus.Model(num_scales,seq_len,ch_ind,stride,patch_len,d_model,d_state,d_conv,e_fact,d_ff,dropout,bi_dir,residual,e_layers,enc_in,sp_num,sp_d_model)
        # model=models.Mamba4RUL.Model(num_scales,seq_len,ch_ind,stride,patch_len,d_model,d_state,d_conv,e_fact,d_ff,dropout,bi_dir,residual,e_layers,enc_in,sp_num,sp_d_model)
        # model=models.TBiMamba4RUL.Model(num_scales,seq_len,ch_ind,stride,patch_len,d_model,d_state,d_conv,e_fact,d_ff,dropout,bi_dir,residual,e_layers,enc_in,sp_num,sp_d_model)
        model = model.cuda()
        optimizer = optim.Adam(model.parameters(),lr=0.00001)
        minavgvaloss = float('inf')
        best_val_loss = float('inf')
        patience=20

        patience_c=0
        for epoch in range(0,epochs + 1):
            print(f"epoch: {epoch}")
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
            # mselist.append(sumtrainloss)
            model.eval()
            sumtrainloss=0
            with torch.no_grad():
                for i,(history,rul) in enumerate(train_loader):
                    prediction = model.forward(history)
                    val_loss = model.loss_function(prediction,rul)
                    sumtrainloss += val_loss.item()*rul.size(0)
                sumtrainloss = sumtrainloss / x_train.shape[0]
                mselist.append(sumtrainloss)
                for i,(history,rul) in enumerate(val_loader):
                    prediction = model.forward(history)
                    val_loss = model.loss_function(prediction,rul)
                    sumvalloss += val_loss.item()*rul.size(0)
                sumvalloss = sumvalloss / x_val.shape[0]
                print(f"train loss:{sumtrainloss:.2f} val loss:{sumvalloss:.2f}")
                vallist.append(sumvalloss)
                patience_c+=1
                if(sumvalloss<minavgvaloss):
                    patience_c=0
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        print(f"New best model found with val_loss: {sumvalloss:.5f}")
                        torch.save(model.state_dict(),osp.join(model_dir,f"temp.pt"))
                        minavgvaloss=sumvalloss
                if patience_c>patience:
                    break
                
            torch.cuda.empty_cache()
        x_a = np.linspace(0, len(mselist) - 1, len(mselist))
        x_b = np.linspace(0, len(vallist) - 1, len(vallist))
        vallist = np.interp(x_a, x_b, vallist)
        fig = plt.figure()
        plt.plot(mselist,label='train')
        plt.plot(vallist,label='valid')
        plt.legend(loc='best')
        plt.xlabel('epoch',{'fontsize':20,'color':'black'})    # 設定 x 軸標籤
        plt.ylabel('loss',{'fontsize':20,'color':'black'})  # 設定 y 軸標籤
        plt.savefig(f'{datasetname}training&validationLosses.png')
        model.eval()
        test_rmse = 0.0
        test_score = 0.0
        model.load_state_dict(torch.load(osp.join(model_dir, "temp.pt"), weights_only=True))
        with torch.no_grad():
            for i, (history, rul) in enumerate(test_loader):
                history, rul = history.cuda(), rul.cuda()  # Move both tensors to CUDA
                output = model.forward(history)  # Forward pass with GPU data
                rmse_loss = torch.sqrt(F.mse_loss(output, rul.view_as(output), reduction='mean'))
                if i == 203 and datasetname == 'FD004':
                    temp=rulutils.score(rul.cpu().detach().numpy(), output.cpu().detach().numpy())
                    print("have odd data!")
                    print(f"no.{i} socre : {temp}")
                    test_score += temp
                else:
                    temp=rulutils.score(rul.cpu().detach().numpy(), output.cpu().detach().numpy())
                    print(f"no.{i} socre : {temp}")
                    test_score += temp
                                # print(f"no.{i} socre: {temp}")
                test_rmse += rmse_loss.item()
            test_rmse /= len(test_loader)
            with open("erecord.txt", "a") as file:
                file.write(f"Test score: {test_score:.2f} Test rmse:{test_rmse:.2f} in {datasetname}\n")
            print(f"Test score: {test_score:.2f} Test rmse:{test_rmse:.2f}")
        torch.cuda.empty_cache()
if __name__ == "__main__":
    main()









    