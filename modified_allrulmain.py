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
# from tensorboardX import SummaryWriter
from sklearn import preprocessing
from torch.utils.data import DataLoader
import rulutils
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
        return history,self.y[index]
    def __len__(self):
        return self.len
def main():
    from itertools import product    
        # config = self.config
    model_dir=r"./checkpoint"
    datasetnames=["FD002"]
    # datasetnames=["FD004"]
    for datasetname in datasetnames:
        for sequence_length in [32, 48, 64]:
                alpha = 0.1
                best_test_loss = float('inf')
                best_test_score = float('inf')
                # best_test_score = 574
                # if datasetname == "FD001":
                #     best_test_loss = 8.5
                threshold = 120
                sensors = ['s_2','s_3', 's_4', 's_7','s_8','s_9','s_11', 's_12','s_13','s_14','s_15','s_17','s_20','s_21']
                x_train, y_train, x_val, y_val, x_test, y_test = rulutils.get_data(datasetname, sensors, sequence_length, alpha, threshold)
                train_dataset = RULDataset(x_train,y_train)
                val_dataset = RULDataset(x_val,y_val)
                test_dataset = RULDataset(x_test, y_test)
                        # test_dataset = RULDataset(x_test, y_test)
                hyperparameter_grid = product([64, 128, 256, 512], [1, 2, 3, 4], [64, 128, 256, 512, 1024])
                epochs=150
                found_start = False
                # 定義從哪個超參數組合開始
                start_combination = (32,256, 0.1, 2, 512)
                for  sp_d_model, num_scales, d_model in hyperparameter_grid:
                    if not found_start:
                        if (sequence_length, sp_d_model, alpha, num_scales, d_model) == start_combination:
                            found_start = True
                        else:
                            continue
                    if sp_d_model >= 256 and num_scales >= 3 and d_model >=512:
                        bsize = 8
                    elif (sp_d_model >= 512 and num_scales >= 3 ) or d_model >=512:
                        bsize = 16
                    elif sp_d_model >= 128 and num_scales >= 3 and d_model >=256:
                        bsize = 32
                    elif sp_d_model >= 128 or num_scales >= 2 and d_model >= 256:
                        bsize = 64
                    else:
                        bsize = 128
                    print(f'bsize:{bsize}')
                    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=bsize,shuffle=True)
                    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
                    print(f'Testing configuration: sequence_length={sequence_length}, sp_d_model={sp_d_model}, alpha={alpha}, num_scales={num_scales}, d_model={d_model}')
                    sequence_length = sequence_length  # sequence_length,48,64 3
                    sp_d_model = sp_d_model       # sp_d_model 128 256 512 4
                    alpha = alpha           # alpha 0.3 0.5 3
                    num_scales = num_scales        # num_scales 2 3 4 4 
                    d_model = d_model         # d_model 128 256 512 1024 5
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
                    model = model.cuda()
                    optimizer = optim.Adam(model.parameters(),lr=0.00001)
                    mselist=[]
                    vallist=[]
                    minavgvaloss = float('inf')
                    best_val_loss = float('inf')
                    patience=15
                    patience_c=0
                    for epoch in range(0,epochs + 1):
                            print(f"epoch: {epoch}")
                            model.train() 
                            sumvalloss=0
                            sumtrainloss=0
                            sumvalscore=0.
                            for i,(history,rul) in enumerate(train_loader):
                                optimizer.zero_grad()
                                output = model.forward(history)
                                train_loss = model.loss_function(output,rul)
                                train_loss.backward()
                                optimizer.step()
                                sumtrainloss+=train_loss.item()*rul.size()[0]
                            torch.cuda.empty_cache()
                            # sumtrainloss = sumtrainloss / x_train.shape[0]
                            # mselist.append(sumtrainloss)
                            model.eval()
                            with torch.no_grad():
                                for i,(history,rul) in enumerate(val_loader):
                                        prediction = model.forward(history)
                                        val_loss = model.loss_function(prediction,rul)
                                        sumvalloss += val_loss.item()*rul.size(0)
                                sumvalloss = sumvalloss / x_val.shape[0]
                                # print(f"train loss:{sumtrainloss} val loss:{sumvalloss}")
                                vallist.append(sumvalloss)
                                patience_c+=1
                                if(sumvalloss<minavgvaloss):
                                        patience_c=0
                                        if val_loss < best_val_loss:
                                            best_val_loss = val_loss
                                            print(f"New best model found with val_loss: {val_loss:.5f}")
                                            torch.save(model.state_dict(),osp.join(model_dir,f"temp.pt"))
                                        minavgvaloss=sumvalloss
                                if patience_c>patience:
                                    break
                    try:
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
                                else:
                                    temp=rulutils.score(rul.cpu().detach().numpy(), output.cpu().detach().numpy())
                                    print(f"no.{i} socre : {temp}")
                                    test_score += temp
                                    # print(f"no.{i} socre: {temp}")
                                test_rmse += rmse_loss.item()
                        torch.cuda.empty_cache()
                        test_rmse /= len(test_loader)
                        print(f"Test score: {test_score} with combination {(sequence_length, sp_d_model, alpha, num_scales, d_model, d_ff)} in {datasetname}")
                        with open("record.txt", "a") as file:
                            file.write(f'Testing configuration: sequence_length={sequence_length}, sp_d_model={sp_d_model}, alpha={alpha}, num_scales={num_scales}, d_model={d_model} in {datasetname}\n')
                            file.write(f"Test Loss: {test_score:.5f} and Test rmse: {test_rmse:.5f}\n")
                        if test_score < best_test_score:
                            best_test_score = test_score
                            best_combination = (sequence_length, sp_d_model, alpha, num_scales, d_model, d_ff)
                            torch.save(model.state_dict(), osp.join(model_dir, f"{datasetname}minvallossinGridSearch.pt"))
                            print(f"Best Test score: {best_test_score:.5f} and Test rmse: {test_rmse:.5f} with combination {best_combination} in {datasetname}")
                            
                            with open("BestTestLoss.txt", "a") as file:
                                file.write(f"Best Test Loss: {best_test_score:.5f}and Test rmse: {test_rmse:.5f}  with combination {best_combination} in {datasetname}\n")
                    except Exception as e: 
                        print(e)
                        with open("problem.txt", "a") as file:
                            file.write(f"there is {e} problem in {(datasetname,sequence_length, sp_d_model, alpha, num_scales, d_model, d_ff)} loss is {test_rmse}score is {test_score}\n")
if __name__ == "__main__":
    main()