import os
import argparse
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
# from tensorboardX import SummaryWriter
from sklearn import preprocessing
from torch.utils.data import DataLoader
import rulutils
import models.TSMamba
def load_part_array_merge(npz_units,path):
    sample_array_lst = []
    label_array_lst = []
    for npz_unit in npz_units:
      # loaded = np.load(npz_unit)
      loaded = np.load(os.path.join(path,npz_unit))
      sample_array_lst.append(loaded['sample'])
      label_array_lst.append(loaded['label'])
    sample_array = np.dstack(sample_array_lst)
    label_array = np.concatenate(label_array_lst)
    sample_array = sample_array.transpose(2, 0, 1)
    return sample_array, label_array
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
def main():
        # config = self.config
        sequence_length = 50  # 32,48,64 
        sp_d_model = 256       # 64 128 256 512 
        alpha = 0.1           # 0.1 0.3 0.5 
        num_scales = 3        # 1 2 3 4 
        d_model = 256         # 64 128 256 512 1024 
        d_ff = 64            # 64 128 256 512 1024 
	    # smoothing intensity
	        # max RUL
        # input_dim=len(sensors)
        # input_dim = 20
        # threshold = 120
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
        enc_in = 20
        sp_num = 20
        # model=models.BiMamba4RUL.Model()
        # model=models.BiMamba4RULplus.Model(num_scales,seq_len,ch_ind,stride,patch_len,d_model,d_state,d_conv,e_fact,d_ff,dropout,bi_dir,residual,e_layers,enc_in,sp_num,sp_d_model)
        model=models.TSMamba.Model(num_scales,seq_len,ch_ind,stride,patch_len,d_model,d_state,d_conv,e_fact,d_ff,dropout,bi_dir,residual,e_layers,enc_in,sp_num,sp_d_model)
        model = model.cuda()
        trainlist=os.listdir(r'./N-CMPASS/train')
        path=r'./N-CMPASS/train'
        x_train,y_train=load_part_array_merge(trainlist,path)
        train_dataset = RULDataset(x_train,y_train)
        path=r'./N-CMPASS/test'
        testlist=os.listdir(r'./N-CMPASS/test')
        x_test,y_test=load_part_array_merge(testlist,path)
        test_dataset = RULDataset(x_test,y_test)
        # test_dataset = RULDataset(x_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=256,shuffle=True)
        context_dim=sequence_length
        print("context_dim",context_dim)
        epochs=150
        optimizer = optim.Adam(model.parameters(),lr=0.00001)
        # optimizer = optim.Adam(model.parameters(),lr=0.00001)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.95)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        mselist=[]
        vallist=[]
        minavgvaloss=9999
        patience=20
        patience_c=0
        model_dir=r'./N-CMPASS'
        datasetname='1'
        for epoch in range(0, epochs + 1):
            print(f"epoch: {epoch}")
            sumvalloss = 0
            sumtrainloss = 0
            sumvalscore = 0.
            
            # 訓練階段
            for i, (history, rul) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model.forward(history)
                train_loss = model.loss_function(output, rul)
                train_loss.backward()
                optimizer.step()
                sumtrainloss += train_loss.item() * rul.size(0)
                
            sumtrainloss = sumtrainloss / x_train.shape[0]
            mselist.append(sumtrainloss)
            
            # 驗證階段
            model.eval()
            with torch.no_grad():  # 禁用計算圖，減少記憶體使用
                for i, (history, rul) in enumerate(test_loader):
                    prediction = model.forward(history)
                    val_loss = model.loss_function(prediction, rul)
                    sumvalloss += val_loss.item() * rul.size(0)
            
            sumvalloss = sumvalloss / x_test.shape[0]
            print(f"train loss:{sumtrainloss} val loss:{sumvalloss}")
            vallist.append(sumvalloss)
            
            patience_c += 1
            if sumvalloss < minavgvaloss:
                patience_c = 0
                print("saving new pt file")
                torch.save(model.state_dict(), osp.join(model_dir, f"{datasetname}minvalloss_epoch.pt"))
                minavgvaloss = sumvalloss
            
            if patience_c > patience:
                break
            
            model.train()  # 切回訓練模式
            torch.cuda.empty_cache()

        x_a = np.linspace(0, len(mselist) - 1, len(mselist))
        x_b = np.linspace(0, len(vallist) - 1, len(vallist))
        vallist = np.interp(x_a, x_b, vallist)
        fig = plt.figure()
        plt.plot(mselist,label='train')
        plt.plot(vallist,label='valid')
        plt.xlabel('epoch num',{'fontsize':20,'color':'black'})    # 設定 x 軸標籤
        plt.ylabel('loss vlaue',{'fontsize':20,'color':'black'})  # 設定 y 軸標籤
        plt.savefig('training&validationLosses.png')
if __name__ == "__main__":
    main()