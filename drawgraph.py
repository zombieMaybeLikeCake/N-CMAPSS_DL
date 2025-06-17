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
        # logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')
        # if  not os.path.exists(traindataname) or not os.path.exists(testdataname):
        sensors = ['s_2','s_3', 's_4', 's_7','s_8','s_9','s_11', 's_12','s_13','s_14','s_15','s_17','s_20','s_21']
        # sensors = ['s_3', 's_4', 's_7', 's_11', 's_12']
        sequence_length = 32
	    # smoothing intensity
        alpha = 0.1
	    # max RUL
        input_dim=len(sensors)
        threshold = 120
        specifyno= 28
        # model=models.BiMamba4RUL.Model()
        model=models.BiMamba4RULplus.Model()
        model = model.cuda()
        model.load_state_dict(torch.load('.//checkpoint//'+datasetname+'minvalloss_epoch'+'.pt'))
        # model.load_state_dict(torch.load('.//checkpoint//FD001_epoch'+str(epoch)+'.pt'))
        x_train, y_train,x_test, y_test = rulutils.get_certain_test_data(datasetname, sensors, sequence_length, alpha, threshold,specifyno)
        train_dataset = RULDataset(x_train,y_train)
        test_dataset = RULDataset(x_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1,shuffle=False)
        mselist=[]
        vallist=[]
        pruls=[]
        rruls=[]
        minavgvaloss=9999
        sumvalloss=0.
        test_score=0.
        scores=[]
        model.eval()
        # for i,(history,rul) in enumerate(train_loader):
        for i,(history,rul) in enumerate(test_loader):
            prediction = model.forward(history)
            val_loss = model.loss_function(prediction,rul)
            sumvalloss += val_loss.item()*rul.size(0)
            score = rulutils.score_in_train(rul.cpu().detach().numpy(),prediction.cpu().detach().numpy())
            test_score += score
            scores.append(score)
            # print(prediction.cpu().detach().numpy()[0][0])
            pruls.append(prediction.cpu().detach().numpy()[0][0])
            rruls.append(rul.cpu().detach().numpy()[0][0])
            vallist.append(sumvalloss)
        # print(f"epoch:{epoch} train loss{sumtrainloss} val loss:{sumvalloss}")
        # print(f"train loss:{sumvalloss/x_train.shape[0]}")
        print(f"test loss:{sumvalloss/x_test.shape[0]}")
        # x_a = np.linspace(0, len(mselist) - 1, len(mselist))
        # x_b = np.linspace(0, len(vallist) - 1, len(vallist))
        # vallist = np.interp(x_a, x_b, vallist)
        for p,r,res in zip(pruls,rruls,scores):
            print(f"real: {r:.2f} prediction: {p:.2f} score: {res:.2f}")
        fig = plt.figure()
        plt.title(f'{datasetname} No.{specifyno} engine RUL')
        plt.plot(pruls, label='prediction')
        plt.plot(rruls, label='real')
        plt.xlabel('time cycle',{'fontsize':20,'color':'black'})    # 設定 x 軸標籤
        plt.ylabel('RUL',{'fontsize':20,'color':'black'})  # 設定 y 軸標籤
        plt.savefig(osp.join(r"./"+datasetname,f'{datasetname}_No.{specifyno}_engineRUL.png'))
if __name__ == "__main__":
    main()