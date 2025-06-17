import os
import argparse
import optuna
import torch
import numpy as np
import random
import os.path as osp
import logging
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import models.BiMamba4RULplus
from sklearn import preprocessing
from utils import get_data_fixed_gss  # 使用我們處理 `dailysemtemp.csv` 的函式
import matplotlib.pyplot as plt
# 設定 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SeaTempDataset(Dataset):
    def __init__(self, x_data, y_data):
        """
        x_data: (樣本數, 32, 10) -> 32 天的歷史數據
        y_data: (樣本數, 7) -> 未來 7 天的 sea_temp
        """
        self.x = torch.tensor(x_data, dtype=torch.float32)  # 轉成 PyTorch Tensor
        self.y = torch.tensor(y_data, dtype=torch.float32)  # 轉成 PyTorch Tensor
        self.len = len(self.x)

    def __getitem__(self, index):
        history = self.x[index].to(device)  # 將數據移到 GPU
        future_temp = self.y[index].to(device)
        return history, future_temp

    def __len__(self):
        return self.len

def main():
    # 設定超參數
    sequence_length = 32  # 時間窗口長度
    alpha = 0.1  # 指數平滑參數
    test_size = 0.2  # 測試集比例
    batch_size = 128  # 訓練批次大小
    learning_rate = 1e-5  # Adam 學習率
    epochs = 150  # 訓練輪數
    patience = 20  # 早停標準
    model_dir = "./checkpoint"  # 存放模型權重
    os.makedirs(model_dir, exist_ok=True)

    # 讀取 `dailysemtemp.csv`
    file_path = "newseatempclasssegment.csv"
    df = pd.read_csv(file_path)

    # 取得符合 `time window = 32` 的數據
    x_train, y_train, x_test, y_test = get_data_fixed_gss(df, sequence_length=sequence_length, alpha=alpha, test_size=test_size)

    # 建立 PyTorch DataLoader
    train_dataset = SeaTempDataset(x_train, y_train)
    test_dataset = SeaTempDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 測試時 batch_size = 1

    print(f"Training Data: {x_train.shape}, Testing Data: {x_test.shape}")
    
    # 設定 Mamba 模型超參數
    num_scales = 4
    d_model = 256
    sp_d_model = 128
    d_ff = 256
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
    enc_in = 9  # **dailysemtemp.csv 有 10 個特徵**
    sp_num = 9  # 依照特徵數修改

    # 選擇 BiMamba4RULplus 模型
    model = models.BiMamba4RULplus.Model(num_scales, seq_len, ch_ind, stride, patch_len, d_model, d_state,
                                         d_conv, e_fact, d_ff, dropout, bi_dir, residual, e_layers, enc_in, sp_num, sp_d_model)
    model = model.to(device)  # 移動到 GPU

    # 設定 Adam 優化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 用來儲存訓練與驗證的損失
    train_losses = []
    val_losses = []

    # 訓練模型
    min_val_loss = float("inf")
    patience_count = 0
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        model.train()
        train_loss = 0
        for history, future_temp in train_loader:
            optimizer.zero_grad()
            output = model(history)  # 前向傳播
            loss = F.mse_loss(output, future_temp)  # 使用 MSE 損失
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * history.size(0)
        # 驗證模型
        model.eval()
        train_loss=0
        with torch.no_grad():
            for history, future_temp in train_loader:
                output = model(history)
                loss = F.mse_loss(output, future_temp)
                train_loss += loss.item() * history.size(0)
        train_loss /= len(train_loader.dataset)
        print(f"Train Loss: {train_loss:.4f}")
        val_loss = 0
        with torch.no_grad():
            for history, future_temp in test_loader:
                output = model(history)
                loss = F.mse_loss(output, future_temp)
                val_loss += loss.item() * history.size(0)

        val_loss /= len(test_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}")

        # 記錄每個 epoch 的訓練與驗證損失
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 早停判斷
        if val_loss < min_val_loss:
            patience_count = 0
            min_val_loss = val_loss
            print(f"Saving best model with val_loss: {val_loss:.5f}")
            torch.save(model.state_dict(), osp.join(model_dir, "best_model.pt"))
        else:
            patience_count += 1

        if patience_count > patience:
            print("Early stopping...")
            break

    # # 測試模型
    model.load_state_dict(torch.load(osp.join(model_dir, "best_model.pt")))
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for history, future_temp in test_loader:
            output = model(history)
            print(f"predit:{output.cpu().detach().numpy()}true:{future_temp.cpu().detach().numpy()}\n")
            loss = F.mse_loss(output, future_temp)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    # 繪製訓練與驗證損失
    # 當訓練提前終止時，會有不同的epoch數，因此繪圖時需要考慮實際訓練的epoch數
    num_epochs_to_plot = len(train_losses)  # 使用實際訓練的epoch數量
    plt.figure(figsize=(10, 6))
    # plt.plot(range(num_epochs_to_plot), train_losses, label='Train Loss', color='blue', linestyle='-', marker='o')
    # plt.plot(range(num_epochs_to_plot), val_losses, label='Validation Loss', color='red', linestyle='-', marker='o')
    plt.plot(range(num_epochs_to_plot), train_losses, label='Train Loss')
    plt.plot(range(num_epochs_to_plot), val_losses, label='Validation Loss')
    # #         plt.plot(mselist,label='train')
    #     plt.plot(vallist,label='valid')
    #     plt.legend(loc='best')
    #     plt.xlabel('epoch',{'fontsize':20,'color':'black'})    # 設定 x 軸標籤
    #     plt.ylabel('loss',{'fontsize':20,'color':'black'})  # 設定 y 軸標籤
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'training&validationLosses.png')

if __name__ == "__main__":
    main()