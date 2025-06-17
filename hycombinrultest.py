import os
import argparse
import models.BiMamba4RULbackup
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
import models.BiMamba4RUL 
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
        # print(self.x[index].shape)
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
    model_dir = r"./checkpoint"
    datasetname = "FD001"
    threshold=120
    # 超參數組合範圍
    sequence_lengths = [32, 48, 64]
    sp_d_models = [64, 128, 256, 512]
    alphas = [0.1, 0.3, 0.5]
    num_scales_list = [1, 2, 3, 4]
    d_models = [64, 128, 256, 512, 1024]
    d_ffs = [64, 128, 256, 512, 1024]
    sensors = ['s_2','s_3', 's_4', 's_7','s_8','s_9','s_11', 's_12','s_13','s_14','s_15','s_17','s_20','s_21']
    best_test_loss = float('inf')
    best_combination = None

    # 遍歷所有超參數組合
    for sequence_length in sequence_lengths:
        for sp_d_model in sp_d_models:
            for alpha in alphas:
                for num_scales in num_scales_list:
                    for d_model in d_models:
                        for d_ff in d_ffs:
                            # 模型檔案名稱
                            model_path = osp.join(model_dir, f"{datasetname}minvalloss_in{str(sequence_length)}_{str(sp_d_model)}_{str(alpha)}_{str(num_scales)}_{str(d_model)}_{str(d_ff)}.pt")
                            
                            if not osp.exists(model_path):
                                continue

                            # 加載模型
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
            # model=models.BiMamba4RUL.Model()
                            model=models.BiMamba4RULplus.Model(num_scales,seq_len,ch_ind,stride,patch_len,d_model,d_state,d_conv,e_fact,d_ff,dropout,bi_dir,residual,e_layers,enc_in,sp_num,sp_d_model)
                            model.load_state_dict(torch.load(model_path))
                            model = model.cuda()
                            
                            # 獲取數據
                            x_train, y_train, x_val, y_val, x_test, y_test = rulutils.get_data(datasetname, sensors, sequence_length, alpha, threshold)
                            test_dataset = RULDataset(x_test, y_test)
                            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
                            
                            # 測試模型
                            model.eval()
                            test_rmse = 0.
                            test_score = 0.
                            for i, (history, rul) in enumerate(test_loader):
                                output = model.forward(history.cuda()) 
                                rmse_loss = torch.sqrt(F.mse_loss(output.cuda(), rul.view_as(output).cuda(), reduction='mean'))
                                test_rmse += rmse_loss.item()
                                output=output.cpu().detach().numpy()
                                rul=rul.cpu().detach().numpy()
                                test_score += rulutils.score(rul,output)
                            test_rmse /= len(test_loader)
                            print('Final Result : test_rmse %.2f test_score %.2f' %(test_rmse,test_score))
                            # 比較是否為最佳test_loss
                            if test_rmse < best_test_loss:
                                best_test_loss = test_rmse
                                best_combination = (sequence_length, sp_d_model, alpha, num_scales, d_model, d_ff)
                                print(f"Best Test Loss: {best_test_loss} with combination {best_combination}")
                            print(f"combination end{(sequence_length, sp_d_model, alpha, num_scales, d_model, d_ff)}")
    
    print(f"Best Test Loss: {best_test_loss} with combination {best_combination}")
    
if __name__ == "__main__":
    main()

