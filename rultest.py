import os
import argparse
import models.BiMamba4RULplus
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
# from tensorboardX import SummaryWriter
from sklearn import preprocessing
from torch.utils.data import DataLoader
import rulutils
import os
import warnings
import tensorflow as tf
import logging

# 忽略 Python 警告
warnings.filterwarnings('ignore')

# 设置环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示警告和错误信息
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用 oneDNN 自定义操作
# 设置 TensorFlow 日志级别
tf.get_logger().setLevel(logging.ERROR)

# 设置所有处理器的日志级别为 ERROR
for handler in tf.get_logger().handlers:
    handler.setLevel(logging.ERROR)
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
        # config = self.config
        model_dir=r"./checkpoint"
        datasetname="FD003"
        # logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')
        # if  not os.path.exists(traindataname) or not os.path.exists(testdataname):
        if True:
            sensors = ['s_2','s_3', 's_4', 's_7','s_8','s_9','s_11', 's_12','s_13','s_14','s_15','s_17','s_20','s_21']
            # sensors = ['s_3', 's_4', 's_7', 's_11', 's_12']
            sequence_length = 32
	        # smoothing intensity
            alpha = 0.1
	        # max RUL
            input_dim=len(sensors)
            threshold = 120
            # model=models.BiMamba4RUL.Model()
            num_scales = 3
            seq_len = sequence_length
            ch_ind = 1
            stride = sequence_length
            patch_len = sequence_length
            d_model = 256
            d_state = 8
            d_conv = 2
            e_fact = 1
            d_ff = 256
            dropout = 0.15
            bi_dir = 1
            residual = 1
            e_layers = 2
            enc_in = 14
            sp_num = 14
            sp_d_model = 64
            # model=models.BiMamba4RUL.Model()
            model=models.BiMamba4RULplus.Model()
            model = model.cuda()
            # model.load_state_dict(torch.load('.//checkpoint//'+datasetname+'_epoch440'+'.pt'))
            model.load_state_dict(torch.load('.//checkpoint//'+datasetname+'minvalloss_epoch'+'.pt'))
            x_train, y_train, x_val, y_val, x_test, y_test = rulutils.get_data(datasetname, sensors, sequence_length, alpha, threshold)
            # test_dataset = RULDataset(x_val, y_val)
            test_dataset = RULDataset(x_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=1,shuffle=False)
        model.eval()
        test_rmse = 0.
        test_score = 0.
        predictions=[]
        test_rmse_values = []
        for i, (history,rul) in enumerate(test_loader):
            output = model.forward(history) 
            rmse_loss=torch.sqrt(F.mse_loss(output.cuda(),rul.view_as(output).cuda(), reduction='mean'))
            num=rul.size(0)
            output=output.cpu().detach().numpy()
            rul=rul.cpu().detach().numpy()
            predictions.append(output[0][0])
            # rmse_loss=rulutils.evaluate(rul,output, label='test')
            test_rmse += rmse_loss.item()*num
            test_rmse_values.append(rmse_loss.item())
            # test_score += rulutils.score(rul,output)
            if i!=203:
                test_score += rulutils.score(rul,output)
                continue
            else:
                # test_score += rulutils.score(rul,output)
                print(f"engine NO:{i+1} y_true:{rul}, y_hat:{output}")
            # rulutils.evaluate(rul,output,'test')
            # test_rmse += rmse_loss.item() / len(test_loader)
            # test_rmse += rulutils.evaluate(rul,output,'test')
        test_score = test_score
        test_rmse = test_rmse/x_test.shape[0]
        print('Final Result : test_rmse %.2f test_score %.2f' %(test_rmse,test_score))
        plt.figure()
        plt.hist(test_rmse_values, bins=20, edgecolor='black')
        plt.xlabel('RMSE', fontsize=20, color='black')
        plt.ylabel('Frequency', fontsize=20, color='black')
        plt.title('RMSE Distribution', fontsize=24, color='black')
        # plt.scatter(range(len(predictions)), predictions, label='prediction', marker='o')
        # plt.scatter(range(len(y_test)), y_test, label='real', marker='o')
        # plt.xlabel('time cycle', {'fontsize': 20, 'color': 'black'})  # 設定 x 軸標籤
        # plt.ylabel('RUL', {'fontsize': 20, 'color': 'black'})  # 設定 y 軸標籤
        # plt.legend() 
        plt.savefig(f'{datasetname}RMSE.png')
if __name__ == "__main__":
    main()