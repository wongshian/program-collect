#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# In[2]:


# 設定 COCO 數據集路徑
coco_root = "data/coco128"  # 根目錄
annotation_file = f"{coco_root}/annotations.json"  # 標註檔
image_dir = f"{coco_root}/images"  # 圖片資料夾

# 定義 COCO 數據集
coco_dataset = torchvision.datasets.CocoDetection(root=image_dir, annFile=annotation_file,
                                                  transform=transforms.ToTensor())


# In[3]:


# 測試讀取一筆資料
def visualize_sample(index=0):
    img, target = coco_dataset[index]  # 取得圖片與標註
    
    # 顯示圖片
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(img.permute(1, 2, 0))  # 轉換通道順序
    
    # 繪製 bounding boxes
    for obj in target:
        bbox = obj["bbox"]  # COCO 格式 [x_min, y_min, width, height]
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, f"{obj['category_id']}", color='red', fontsize=10)
    
    plt.show()

# 可視化一筆數據
visualize_sample(2)


# In[ ]:





# In[153]:


def collate_fn(batch):
    images, targets = zip(*batch)  # 拆開圖片和標註

    # 從 targets 讀取對應的 image_id
    image_ids = [target[0]["image_id"] for target in targets if len(target) > 0]
    print(image_ids)
    # 根據 image_id 查找對應的檔案名稱
    file_names = [train_dataset.coco.loadImgs(img_id)[0]["file_name"] for img_id in image_ids]

    return images, targets, file_names

# 設定 DataLoader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn  # 使用常規的函數代替 lambda
)

for images, targets, file_names in train_dataloader:
    print(f"本批次圖片檔名: {file_names}")
    break  # 先測試第一批


# In[133]:


def collate_fn(batch):
    images, targets = zip(*batch)  # 拆開圖片和標註

    # 從 batch 中獲取對應的 image_id
    image_ids = [train_dataset.ids[i] for i in range(len(batch))]

    # 根據 image_id 查找對應的檔案名稱
    file_names = [train_dataset.coco.loadImgs(img_id)[0]["file_name"] for img_id in image_ids]

    return images, targets, file_names
# 設定 DataLoader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn  # 使用常規的函數代替 lambda
)
    
for images, targets, file_names in train_dataloader:
    print(f"本批次圖片檔名: {file_names}")
    break  # 先測試第一批


# In[114]:


batch


# In[154]:


import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection

# 設定轉換（這裡我們會進行圖片增強）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(0.5),  # 隨機水平翻轉
    transforms.Resize((256, 256))  # 設定輸入大小
])

# 載入 COCO 數據集
train_dataset = CocoDetection(
    root=image_dir, 
    annFile=annotation_file, 
    transform=transform
)

# 設定 DataLoader
batch_size = 2  # 根據你的硬體可以調整

# 自定義 collate_fn 函數
# def collate_fn(batch):
#     return tuple(zip(*batch))
# def collate_fn(batch):
#     images, targets = zip(*batch)  # 拆開圖片和標註
#     file_names = [train_dataset.coco.loadImgs(train_dataset.ids[i])[0]["file_name"] for i in range(len(batch))]
#     return images, targets, file_names

def collate_fn(batch):
    images, targets = zip(*batch)  # 拆開圖片和標註

    # 從 targets 讀取對應的 image_id
    image_ids = [target[0]["image_id"] for target in targets if len(target) > 0]
    print(image_ids)
    # 根據 image_id 查找對應的檔案名稱
    file_names = [train_dataset.coco.loadImgs(img_id)[0]["file_name"] for img_id in image_ids]

    return images, targets, file_names    
# 設定 DataLoader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn  # 使用常規的函數代替 lambda
)


# 測試 DataLoader 是否正常運行
# for images, targets in train_dataloader:
#     print(f"Batch size: {len(images)}")
#     print(f"Images shape: {images[0].shape}")
#     print(f"Targets: {targets[0]}")
#     break  # 顯示第一個批次，檢查資料是否正確

i = 0
for images, targets, file_name in train_dataloader:
    print(f"Batch size: {len(images)}")
    print(f"Images shape: {images[0].shape}")
    print(f"Targets: {targets[0]}")
    print(f"file_name: {file_name}")

    i = i+1
    if i == 5:
        break  # 顯示第一個批次，檢查資料是否正確


# In[155]:


def convert(targets):
    processed_targets = []

    for target in targets:
        processed_target = []
        for a in target:
            tmp = {}
            tmp['class_labels'] = torch.tensor([int(a['category_id'])]).to(device)
            tmp['boxes'] = torch.tensor(a['bbox']).to(device)
            processed_target.append(tmp)
        processed_targets.append(processed_target)
    return processed_targets


# In[158]:


from transformers import DetrForObjectDetection
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# 載入預訓練模型
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# 設定訓練參數
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 優化器設定
optimizer = AdamW(model.parameters(), lr=1e-5)

# 訓練步驟
num_epochs = 3  # 訓練輪次
# 訓練循環
import torch
from tqdm import tqdm

# 訓練循環
import torch
from tqdm import tqdm

# 訓練循環
for epoch in range(num_epochs):
    model.train()

    # 逐批次訓練
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):

        print(batch[2])
        # 處理 pixel_values
        pixel_values = torch.stack(list(batch[0]), dim=0)  # 合併圖片數據為一個 tensor
        pixel_values = pixel_values.to(device)  # 將 pixel_values 移動到指定設備

        # 處理 targets，將其轉換為 list 格式
        targets = list(batch[1])  # 確保 targets 是 list 格式
        targets = convert(targets)
        # 將 targets 轉換為每個字典包含所需的結構
        # targets = [{
        #     "class_labels": torch.tensor([item["class_labels"]]).to(device),
        #     "boxes": item["boxes"].to(device)
        # } for batch in targets for item in batch]
        # targets
        # 將 targets 中的所有字典值轉換為 tensor 並移動到設備
        # targets = [{
        #     key: torch.tensor(value).to(device) if isinstance(value, list) else value
        #     for key, value in target.items()
        # } for target in targets]
        formatted_targets = []
        for t in targets:
            if len(t)>0:
                formatted_targets.append({
                    "class_labels": torch.tensor([item["class_labels"].item() for item in t], dtype=torch.long).to(device),
                    "boxes": torch.stack([item["boxes"] for item in t]).to(device) 
                })
        formatted_targets
        targets = formatted_targets
        
        # 模型的正向傳播，並傳入 labels（即 targets）
        outputs = model(pixel_values, labels=targets)
        # outputs = model(pixel_values)
        
        # print(outputs)
        # 提取損失值
        loss = outputs.loss
        
        # 優化步驟
        optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()  # 計算梯度
        optimizer.step()  # 更新權重

    print(f"Epoch {epoch+1} loss: {loss.item()}")


# 儲存訓練好的模型
model.save_pretrained("./detr_trained")


# In[88]:


targets = list(batch[1]) 
targets


# In[87]:


targets


# In[84]:


targets = list(batch[1])  # 確保 targets 是 list 格式
targets = convert(targets)
targets


# In[85]:


formatted_targets = []
for batch in targets:
    formatted_targets.append({
        "class_labels": torch.tensor([item["class_labels"].item() for item in batch], dtype=torch.long).to(device),
        "boxes": torch.stack([item["boxes"] for item in batch]).to(device)
    })
formatted_targets


# In[76]:


# 將 targets 轉換為每個字典包含所需的結構
targets = [{
    "class_labels": torch.tensor([item["class_labels"]]).to(device),
    "boxes": item["boxes"].to(device)
} for batch in targets for item in batch]
targets


# In[77]:


len(targets)


# In[78]:


targets


# In[67]:


targets


# In[46]:


batch[0][0].shape


# In[48]:


len(batch)


# In[60]:


list(pixel_values)


# In[62]:


torch.stack(list(pixel_values), dim=0)


# In[69]:


list(targets)


# In[72]:


pixel_values, targets = batch

# 假設 pixel_values 是 (batch_size, 3, height, width)
# 這是 RGB 圖像，需移動到 GPU 或 CPU
pixel_values = torch.stack(list(pixel_values), dim=0)
pixel_values = pixel_values.to(device)
pixel_values.shape


# In[ ]:


# 我們先初始化一個空的列表，用來存儲處理過後的 targets

def convert(targets):
    processed_targets = []
    
    # 遍歷 targets 中的每一個字典（即每個 target）
    for target in targets:
        # 初始化一個空字典，來存儲處理過後的 (key, value)
        processed_target = []
        
        # 遍歷 target 字典中的每一對 (key, value)
        for a in target:
            tmp = {}
            tmp['class_labels'] = torch.tensor(a['category_id']).to(device)
            tmp['bbox'] = torch.tensor(a['bbox']).to(device)
            processed_target.append(tmp)
        
        # 把處理過的 target 加入 processed_targets 列表中
        processed_targets.append(processed_target)
    
    # # 現在 processed_targets 就是每個 target 都被處理過的結果
    # targets = processed_targets
    return processed_targets


# In[98]:


# 我們先初始化一個空的列表，用來存儲處理過後的 targets
processed_targets = []

# 遍歷 targets 中的每一個字典（即每個 target）
for target in targets:
    # 初始化一個空字典，來存儲處理過後的 (key, value)
    processed_target = []
    
    # 遍歷 target 字典中的每一對 (key, value)
    for a in target:
        tmp = {}
        tmp['class_labels'] = torch.tensor(a['category_id']).to(device)
        tmp['bbox'] = torch.tensor(a['bbox']).to(device)
        processed_target.append(tmp)
    
    # 把處理過的 target 加入 processed_targets 列表中
    processed_targets.append(processed_target)

# # 現在 processed_targets 就是每個 target 都被處理過的結果
# targets = processed_targets
processed_targets


# In[84]:


a = [{
    key: torch.tensor(value).to(device) if isinstance(value, list) else value
    for key, value in target[0]
} for target in targets]


# In[77]:


for target in targets:
    print(target)


# In[83]:


len(target)


# In[90]:


for a in target:
    print(type(a))
    print(a)
    for key,value in a:
        print(key)
        print(value)


# In[86]:


for k,v in target:
    print(k)
    print(v)


# In[81]:


for k in target:
    print(k)


# In[ ]:




