#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# In[2]:


import torch
print(torch.cuda.is_available())  # 如果是 True，表示可以使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)  # 輸出 "cuda" 或 "cpu"
print(torch.cuda.device_count())  # 會回傳可用的 GPU 數量
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))  # 顯示 GPU 名稱


# In[3]:


import torch
print(torch.__version__)  # 確認 PyTorch 版本
print(torch.version.cuda)  # 確認 PyTorch 是否支援 CUDA


# In[4]:


# 設定設備 (GPU 優先，否則使用 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:





# In[5]:


# 1. 載入 MNIST 數據集
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# In[6]:


dataiter = iter(trainloader)
images, labels = next(dataiter)


# In[7]:


labels


# In[9]:


import matplotlib.pyplot as plt
import numpy as np
# 3. 取消 Normalize，讓圖片回到 0~1 範圍
images = images * 0.5 + 0.5  # 反轉 Normalize 過的數據

# 4. 定義顯示函數
def imshow(img, ax):
    img = img.numpy()  # 轉換為 NumPy 陣列
    ax.imshow(np.squeeze(img), cmap="gray")  # 顯示灰階圖片
    # ax.axis("off")  # 隱藏座標軸

# 5. 顯示前 10 張圖片
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):  # 使用 ax 來分配不同子圖
    imshow(images[i], ax)  # 顯示圖片
    # print(images[i].flatten())
    ax.set_title(f"Label: {labels[i].item()}")  # 設置標題

# plt.tight_layout()
# plt.show()


# In[10]:


# 2. 定義 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = x.view(-1, 64 * 7 * 7)  # 攤平成 1D 向量
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # 不用 activation (會接 cross-entropy)
        return x

# 初始化模型
model = CNN().to(device)


# In[11]:


# 3. 設定損失函數 & 優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 訓練模型
EPOCHS = 5


# In[12]:


import torch

# 檢查 PyTorch 是否偵測到 GPU
print("GPU 可用:", torch.cuda.is_available())

# 獲取 GPU 名稱
if torch.cuda.is_available():
    print("GPU 名稱:", torch.cuda.get_device_name(0))

# 檢查 GPU 記憶體使用狀況
if torch.cuda.is_available():
    gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
    print(f"已使用 GPU 記憶體: {gpu_memory:.2f} MB")

print(torch.__version__)


# In[13]:


# import torch
# import time
# from tqdm import tqdm  # 載入 tqdm

# # 訓練迴圈
# for epoch in range(EPOCHS):
#     running_loss = 0.0
#     start_time = time.time()  # 記錄開始時間
    
#     # 在每個 Epoch 開始時創建進度條
#     pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")

#     # 手動更新進度條
#     for images, labels in pbar:
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()  # 清除梯度
#         outputs = model(images)  # 前向傳播
#         loss = criterion(outputs, labels)  # 計算 loss
#         loss.backward()  # 反向傳播
#         optimizer.step()  # 更新權重

#         running_loss += loss.item()
        
#         # 更新 tqdm 進度條顯示 Loss
#         pbar.set_postfix(loss=f"{loss.item():.4f}")
        
#     end_time = time.time()  # 記錄結束時間
#     epoch_time = end_time - start_time  # 計算訓練時間
#     avg_loss = running_loss / len(trainloader)  # 計算平均 Loss


#         # 每10個 epoch 儲存模型
#     if (epoch + 1) % 2 == 0 or epoch + 1 == EPOCHS :
#         torch.save(model.state_dict(), f"trained_model_epoch_{epoch+1}.pth")
#         print(f"模型已儲存：trained_model_epoch_{epoch+1}.pth")
    
#     print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")


# In[15]:


import torch
import time
from tqdm import tqdm  # 載入 tqdm

class ModelTrainer:
    def __init__(self, model, trainloader, optimizer, criterion, epochs, device, save_interval=10):
        """
        初始化 ModelTrainer 類別。
        
        :param model: 訓練的模型
        :param trainloader: 訓練資料加載器
        :param optimizer: 優化器
        :param criterion: 損失函數
        :param epochs: 訓練的輪數
        :param device: 訓練設備（CPU 或 GPU）
        :param save_interval: 每多少個 epoch 儲存一次模型（預設每10個 epoch 儲存一次）
        """
        self.model = model
        self.trainloader = trainloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.device = device
        self.save_interval = save_interval

    def train(self):
        """執行模型訓練"""
        for epoch in range(self.epochs):
            running_loss = 0.0
            start_time = time.time()  # 記錄開始時間

            # 創建進度條
            pbar = tqdm(self.trainloader, desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch")

            # 訓練每個 batch
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()  # 清除梯度
                outputs = self.model(images)  # 前向傳播
                loss = self.criterion(outputs, labels)  # 計算 loss
                loss.backward()  # 反向傳播
                self.optimizer.step()  # 更新權重

                running_loss += loss.item()

                # 更新進度條顯示當前 loss
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            end_time = time.time()  # 記錄結束時間
            epoch_time = end_time - start_time  # 計算訓練時間
            avg_loss = running_loss / len(self.trainloader)  # 計算平均 Loss

            # 每 self.save_interval 個 epoch 儲存一次模型
            if (epoch + 1) % self.save_interval == 0 or epoch + 1 == self.epochs:
                torch.save(self.model.state_dict(), f"trained_model_epoch_{epoch+1}.pth")
                print(f"模型已儲存：trained_model_epoch_{epoch+1}.pth")

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

# 使用範例
# 假設你已經有以下變數：model, trainloader, optimizer, criterion, EPOCHS, device
trainer = ModelTrainer(model=model, 
                       trainloader=trainloader, 
                       optimizer=optimizer, 
                       criterion=criterion, 
                       epochs=EPOCHS, 
                       device=device, 
                       save_interval=10)

trainer.train()  # 開始訓練


# In[16]:


# 重新載入模型
# model = YourModelClass()  # 用您定義的模型類別初始化模型
model.load_state_dict(torch.load(f"trained_model_epoch_{EPOCHS}.pth"))
model.eval()  # 評估模式


# In[ ]:





# In[17]:


# # 5. 測試模型
# correct = 0
# total = 0

# # 使用 tqdm 來顯示測試過程中的進度
# with torch.no_grad():
#     pbar = tqdm(testloader, desc="Testing", unit="batch")
#     for images, labels in pbar:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#         # 更新 tqdm 進度條顯示準確度
#         # pbar.update(labels.size(0))
#         pbar.set_postfix(accuracy=f"{100 * correct / total:.2f}%")

# accuracy = 100 * correct / total
# print(f"測試準確度：{accuracy:.2f}%")


# In[18]:


import torch
import time
from tqdm import tqdm

# 假設你的 model 和 testloader 已經設定好了
start_time = time.time()  # 記錄測試開始時間
correct = 0
total = 0

# 使用 tqdm 來顯示測試過程中的進度
with torch.no_grad():
    pbar = tqdm(testloader, desc="Testing", unit="batch")  

    for images, labels in pbar:
        batch_start_time = time.time()  # 記錄這個 batch 的開始時間

        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
   
        # 計算 FPS
        batch_time = time.time() - batch_start_time  # 這個 batch 花費的時間
        fps = labels.size(0) / batch_time if batch_time > 0 else 0  # FPS = 圖片數 / 時間
        
        elapsed_time = time.time() - start_time  # 總時間
        fps = total / elapsed_time  # 以整體時間來計算 FPS
        
        # 使用 set_postfix 更新進度條旁邊顯示的準確度與 FPS
        pbar.set_postfix(accuracy=f"{100 * correct / total:.2f}%", fps=f"{fps:.2f}")

accuracy = 100 * correct / total
print(f"測試準確度：{accuracy:.2f}%")


# In[21]:


@torch.no_grad()
def predict(image, model):
    # 在這裡，所有操作都不會計算梯度
    output = model(image)
    return output


# In[ ]:





# In[25]:


def get_pdn_medium(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=1)
    )


# In[27]:


# 重新載入模型
# model = YourModelClass()  # 用您定義的模型類別初始化模型
teacher = get_pdn_medium(out_channels=384)

# for GPU because the original neural network is trained on a GPU
# teacher.load_state_dict(torch.load(f"teacher_medium.pth"))

# for CPU
teacher.load_state_dict(torch.load(f"teacher_medium.pth", map_location=torch.device('cpu')))

teacher.eval()  # 評估模式

# torch.load with map_location=torch.device('cpu') to map your storages to the CPU.


# In[ ]:





# In[ ]:





# In[ ]:




