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
        model_dir=r"./checkpoint"
        datasetname="FD002"
        epoch=100
        # logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')
        # if  not os.path.exists(traindataname) or not os.path.exists(testdataname):
        if True:
            # sensors = ['s_2','s_3', 's_4', 's_7','s_8','s_9','s_11', 's_12','s_13','s_14','s_15','s_17','s_20','s_21']
            sensors = ['s_1','s_2','s_3','s_4','s_5','s_6','s_7','s_8','s_9','s_10','s_11','s_12','s_13','s_14','s_15','s_16','s_17','s_18','s_19','s_20','s_21']
            # sensors = ['s_3', 's_4', 's_7', 's_11', 's_12']
            sequence_length = 32  # 32,48,64 
            sp_d_model = 128       # 64 128 256 512 
            alpha = 0.1           # 0.1 0.3 0.5 
            num_scales = 1        # 1 2 3 4 
            d_model = 256         # 64 128 256 512 1024 
            d_ff = 64            # 64 128 256 512 1024 
	        # smoothing intensity
	        # max RUL
            input_dim=len(sensors)
            threshold = 120
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
            # enc_in = 14
            # sp_num = 14
            enc_in = 21
            sp_num = 21
            # model=models.BiMamba4RUL.Model()
            model=models.BiMamba4RULplus.Model(num_scales,seq_len,ch_ind,stride,patch_len,d_model,d_state,d_conv,e_fact,d_ff,dropout,bi_dir,residual,e_layers,enc_in,sp_num,sp_d_model)
            model = model.cuda()
            # model.load_state_dict(torch.load('.//checkpoint//FD001_epoch'+str(epoch)+'.pt'))
            x_train, y_train, x_val, y_val, x_test, y_test = rulutils.get_data(datasetname, sensors, sequence_length, alpha, threshold)
            train_dataset = RULDataset(x_train,y_train)
            test_dataset = RULDataset(x_val,y_val)
            # test_dataset = RULDataset(x_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=128,shuffle=True)
        context_dim=sequence_length
        print("context_dim",context_dim)
        epochs=150
        train_folder=build_dir(datasetname+"test1")
        optimizer = optim.Adam(model.parameters(),lr=0.00001)
        # optimizer = optim.Adam(model.parameters(),lr=0.00001)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.95)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        mselist=[]
        vallist=[]
        minavgvaloss=9999
        patience=20
        patience_c=0
        for epoch in range(0,epochs + 1):
                print(f"epoch: {epoch}")
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
                sumtrainloss = sumtrainloss / x_train.shape[0]
                mselist.append(sumtrainloss)
                model.eval()
                for i,(history,rul) in enumerate(test_loader):
                        prediction = model.forward(history)
                        val_loss = model.loss_function(prediction,rul)
                        sumvalloss += val_loss.item()*rul.size(0)
                        # rulutils.score_in_train(rul.cpu().detach().numpy(),prediction.cpu().detach().numpy())
                sumvalloss = sumvalloss / x_val.shape[0]
                # print(f"epoch:{epoch} train loss{sumtrainloss} val loss:{sumvalloss}")
                print(f"train loss:{sumtrainloss} val loss:{sumvalloss}")
                vallist.append(sumvalloss)
                patience_c+=1
                # if True:
                if(sumvalloss<minavgvaloss):
                        patience_c=0
                        print("saveing new pt file")
                        torch.save(model.state_dict(),osp.join(model_dir,f"{datasetname}minvalloss_epoch.pt"))
                        minavgvaloss=sumvalloss
                        # torch.save(model.state_dict(),osp.join(model_dir,f"{datasetname}_epoch{epoch}.pt"))
                # rulutils.score(minrul,minprediction)
                if patience_c>patience:
                    
                    break
                model.train() 
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