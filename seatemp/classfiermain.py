import os
import torch
import numpy as np
import random
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import logging
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import models.BiMamba4tempclass
from sklearn import preprocessing
from utils import get_data_for_classification_with_segments

# 設定 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SeaTempClassificationDataset(Dataset):
    def __init__(self, x_data, y_data, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.x = torch.tensor(np.array(x_data), dtype=torch.float32).to(self.device)
        self.y = torch.tensor(np.array(y_data), dtype=torch.long).to(self.device)
        self.len = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

def main():
    # 設定超參數
    sequence_length = 32  # 時間窗口長度
    test_size = 0.2  # 測試集比例
    batch_size = 64  # 訓練批次大小
    learning_rate = 1e-5  # Adam 學習率
    epochs = 150  # 訓練輪數
    patience = 20  # 早停標準
    model_dir = "./checkpoint"  # 存放模型權重
    os.makedirs(model_dir, exist_ok=True)

    # 讀取分類數據
    file_path = "./newseatempclasssegmentv2.csv"
    df = pd.read_csv(file_path)

    # 取得符合 `time window = 32` 的數據
    x_train, y_train, x_test, y_test, num_classes = get_data_for_classification_with_segments(df, sequence_length=sequence_length, test_size=test_size)
    print(f"y_test min: {y_test.min()}, max: {y_test.max()}")
    print(f"y_train min: {y_train.min()}, max: {y_train.max()}")
    print(f"num_classes: {num_classes}")

    # 建立 PyTorch DataLoader
    train_dataset = SeaTempClassificationDataset(x_train, y_train, device=device)
    test_dataset = SeaTempClassificationDataset(x_test, y_test, device=device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 定義分類模型
    model = models.BiMamba4tempclass.Model(num_scales=4, seq_len=sequence_length, ch_ind=1, stride=sequence_length,
                                           patch_len=sequence_length, d_model=256, d_state=8, d_conv=2, e_fact=1, 
                                           d_ff=256, dropout=0.1, bi_dir=1, residual=1, e_layers=2, 
                                           enc_in=9, sp_num=9, sp_d_model=128, num_classes=num_classes)
    model = model.to(device)  # 移動到 GPU

    # 設定 Adam 優化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)

    # 用來儲存損失與準確度
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # 訓練模型
    min_val_loss = float("inf")
    patience_count = 0
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for history, labels in train_loader:
            history, labels = history.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(history)  # 前向傳播
            loss = criterion(outputs, labels)  # 使用 CrossEntropyLoss
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * history.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = correct / total
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        
        # 保存訓練損失與準確度
        # 驗證模型
        model.eval()
        train_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for history, labels in train_loader:
                outputs = model(history)
                loss = criterion(outputs, labels)
                train_loss += loss.item() * history.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        train_loss /= len(train_loader.dataset)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for history, labels in test_loader:
                outputs = model(history)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * history.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(test_loader.dataset)
        val_acc = correct / total
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        
        # 保存驗證損失與準確度
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

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

    # 繪製訓練與驗證損失與準確度圖表
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('train_validation_loss.png')

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_accuracies)), train_accuracies, label='Train Accuracy')
    plt.plot(range(len(val_accuracies)), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('train_validation_accuracy.png')

    # 輸出最終的測試結果
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    model.load_state_dict(torch.load(osp.join(model_dir, "best_model.pt")))
    with torch.no_grad():
        for history, labels in test_loader:
            outputs = model(history)
            loss = criterion(outputs, labels)
            with open("classrecord.txt", "a") as file:
                file.write(f"outputs: {outputs.cpu().detach().numpy()},GT:{labels.cpu().detach().numpy()}\n")
            val_loss += loss.item() * history.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(test_loader.dataset)
    val_acc = correct / total
    print(f"Final Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    main()
