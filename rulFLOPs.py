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
import models.Transformer4RUL
import models.BiMamba4RULplus
import models.TBiMamba4RUL
import models.Mamba4RUL
import models.VAE 
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis, parameter_count_table
# from tensorboardX import SummaryWriter
from sklearn import preprocessing
from torch.utils.data import DataLoader
import rulutils
import models.MR_LSTM
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
    model_dir=r"./checkpoint"
    datasetname="FD001"
    sequence_length = 32
    alpha = 0.1
    threshold = 120
    sensors = ['s_2','s_3', 's_4', 's_7','s_8','s_9','s_11', 's_12','s_13','s_14','s_15','s_17','s_20','s_21']
    x_train, y_train, x_val, y_val, x_test, y_test = rulutils.get_data(datasetname, sensors, sequence_length, alpha, threshold)
    train_dataset = RULDataset(x_train,y_train)
    val_dataset = RULDataset(x_val,y_val)
    test_dataset = RULDataset(x_test, y_test)
    epochs=1
    found_start = False
    bsize=128
    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bsize,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # print(f'Testing configuration: sequence_length={sequence_length}, sp_d_model={sp_d_model}, alpha={alpha}, num_scales={num_scales}, d_model={d_model}')
    sequence_length = sequence_length  # sequence_length,48,64 3
    sp_d_model = 256       # sp_d_model 128 256 512 4
    alpha = alpha           # alpha 0.3 0.5 3
    num_scales = 2        # num_scales 2 3 4 4 
    d_model = 256         # d_model 128 256 512 1024 5
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
    input_dim = 14  # Number of input features after preprocessing
    hidden_dim = 256
    output_dim = 1
    num_layers = 8
    num_resolutions = 8

# Initialize model
    # model = models.MR_LSTM.Model(input_dim, hidden_dim, output_dim, num_layers, num_resolutions)
    # model=models.BiMamba4RULplus.Model(num_scales,seq_len,ch_ind,stride,patch_len,d_model,d_state,d_conv,e_fact,d_ff,dropout,bi_dir,residual,e_layers,enc_in,sp_num,sp_d_model)
    # model=models.Mamba4RUL.Model(num_scales,seq_len,ch_ind,stride,patch_len,d_model,d_state,d_conv,e_fact,d_ff,dropout,bi_dir,residual,e_layers,enc_in,sp_num,sp_d_model)
    model=models.VAE.Model(input_dim, hidden_dim, output_dim, num_layers, num_resolutions)
    # model=models.TBiMamba4RUL.Model(num_scales,seq_len,ch_ind,stride,patch_len,d_model,d_state,d_conv,e_fact,d_ff,dropout,bi_dir,residual,e_layers,enc_in,sp_num,sp_d_model)
    # model=models.Transformer4RUL.Model()
    # epochs=60
    # optimizer = optim.Adam(model.parameters(),lr=0.00001)
    model.eval()
    summary(model, input_size=(64, 32, 14))
    input_tensor = torch.randn(64, 32, 14).cuda()
    # flop_analysis = FlopCountAnalysis(model, input_tensor)
    # print(f"FLOPs: {flop_analysis.total()}")
    with torch.no_grad():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    # for epoch in range(0,epochs + 1):
    #     model.train()
    #     model = model.cuda()
        sumvalloss=0
    #     for i,(history,rul) in enumerate(train_loader):
    #         optimizer.zero_grad()
    #         history = history.cuda()
    #         prediction = model.forward(history) # 推論
    #         train_loss = models.Transformer4RUL.Model().loss_function(prediction,rul)
    #         train_loss.backward()
    #         optimizer.step()
    #     torch.cuda.empty_cache()
        model.eval()
        with torch.no_grad():
            for i,(history,rul) in enumerate(train_loader):
                history = history.cuda()
                prediction = model.forward(history)
                val_loss = models.Transformer4RUL.Model().loss_function(prediction,rul)
                sumvalloss += val_loss.item()*rul.size(0)
            # sumvalloss = sumvalloss / x_val.shape[0]
            # print(f"val_loss: {sumvalloss:.5f}")
        end.record()
        torch.cuda.synchronize()
    print(f"Inference Time: {start.elapsed_time(end)} ms")  # 毫秒                        
if __name__ == "__main__":
    main()