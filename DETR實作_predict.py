#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import glob
from PIL import Image

# 目錄設定
base_dir = os.path.abspath("./")
image_dir = "data\\coco128\\images\\train2017"   # 圖片資料夾
label_dir = "data\\coco128\\labels\\train2017"   # YOLO 標註資料夾
output_json = "data\\coco128\\annotations.json"  # 輸出 COCO JSON


# In[2]:


os.path.join(base_dir, image_dir)


# In[3]:


data_path = os.path.join(base_dir, image_dir)
label_path = os.path.join(base_dir, label_dir)

print(data_path)
print(label_path)


# In[4]:


files = glob.glob(data_path + '\\*.jpg')
files[0:5]


# In[5]:


# 類別對應 (這裡假設是 COCO 的 80 類)
categories = [{"id": i, "name": f"class_{i}"} for i in range(80)]

# 初始化 COCO 格式
coco = {
    "images": [],
    "annotations": [],
    "categories": categories
}


# In[21]:


# 讀取圖片與標註
annotation_id = 0
for img_id, img_file in enumerate(files):
    print(img_id)
    print(img_file)
    if not img_file.endswith((".jpg", ".png")):
        continue

    # 讀取圖片資訊
    img_path = img_file
    image = Image.open(img_path)
    width, height = image.size

    # 加入圖片 metadata
    coco["images"].append({
        "id": img_id,
        "file_name": img_file,
        "width": width,
        "height": height
    })

    # 找到對應的標註檔
    label_file = img_file.replace('images','labels').replace(".jpg", ".txt").replace(".png", ".txt")
    if not os.path.exists(label_file):
        continue

    # 讀取 YOLO 標註
    with open(label_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])  # 物件類別
        x_center, y_center, w, h = map(float, parts[1:])  # 讀取 YOLO 格式

        # 轉換為 COCO 格式 (絕對座標)
        x_min = (x_center - w / 2) * width
        y_min = (y_center - h / 2) * height
        bbox_width = w * width
        bbox_height = h * height

        # 儲存標註
        coco["annotations"].append({
            "id": annotation_id,
            "image_id": img_id,
            "category_id": class_id,
            "bbox": [x_min, y_min, bbox_width, bbox_height],
            "area": bbox_width * bbox_height,
            "iscrowd": 0
        })
        annotation_id += 1

# 輸出 JSON
with open(output_json, "w") as json_file:
    json.dump(coco, json_file, indent=4)

print(f"轉換完成！COCO JSON 已儲存為 {output_json}")


# In[6]:


import json
with open("./data/coco128/annotations.json", encoding="utf-8") as f:
    data = json.load(f)
print(json.dumps(data, indent=4))


# In[ ]:





# In[105]:


# COCO_class = {1: 'person',
#  2: 'bicycle',
#  3: 'car',
#  4: 'motorcycle',
#  5: 'airplane',
#  6: 'bs',
#  7: 'train',
#  8: 'trck',
#  9: 'boat',
#  10: 'traffic light',
#  11: 'fire hydrant',
#  12: 'stop sign',
#  13: 'parking meter',
#  14: 'bench',
#  15: 'bird',
#  16: 'cat',
#  17: 'dog',
#  18: 'horse',
#  19: 'sheep',
#  20: 'cow',
#  21: 'elephant',
#  22: 'bear',
#  23: 'zebra',
#  24: 'giraffe',
#  25: 'backpack',
#  26: 'mbrella',
#  27: 'handbag',
#  28: 'tie',
#  29: 'sitcase',
#  30: 'frisbee',
#  31: 'skis',
#  32: 'snowboard',
#  33: 'sports ball',
#  34: 'kite',
#  35: 'baseball bat',
#  36: 'baseball glove',
#  37: 'skateboard',
#  38: 'srfboard',
#  39: 'tennis racket',
#  40: 'bottle',
#  41: 'wine glass',
#  42: 'cp',
#  43: 'fork',
#  44: 'knife',
#  45: 'spoon',
#  46: 'bowl',
#  47: 'banana',
#  48: 'apple',
#  49: 'sandwich',
#  50: 'orange',
#  51: 'broccoli',
#  52: 'carrot',
#  53: 'hot dog',
#  54: 'pizza',
#  55: 'dont',
#  56: 'cake',
#  57: 'chair',
#  58: 'coch',
#  59: 'potted plant',
#  60: 'bed',
#  61: 'dining table',
#  62: 'toilet',
#  63: 'tv',
#  64: 'laptop',
#  65: 'mose',
#  66: 'remote',
#  67: 'keyboard',
#  68: 'cell phone',
#  69: 'microwave',
#  70: 'oven',
#  71: 'toaster',
#  72: 'sink',
#  73: 'refrigerator',
#  74: 'book',
#  75: 'clock',
#  76: 'vase',
#  77: 'scissors',
#  78: 'teddy bear',
#  79: 'hair drier',
#  80: 'toothbrsh'}
# COCO_CLASSES = list(COCO_class.values())


# In[106]:


# COCO_CLASSES = [
#     "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
#     "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog", 
#     "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", 
#     "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", 
#     "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
#     "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", 
#     "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", 
#     "bed", "N/A", "dining table", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", 
#     "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
#     "scissors", "teddy bear", "hair drier", "toothbrush"
# ]


# In[196]:


# COCO_CLASSES = [
#     "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
#     "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog", 
#     "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", 
#     "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", 
#     "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
#     "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", 
#     "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", 
#     "bed", "N/A", "dining table", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", 
#     "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
#     "scissors", "teddy bear", "hair drier", "toothbrush", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
# ]


# In[209]:


# https://github.com/facebookresearch/detr/issues/23#issuecomment-636322576
COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


# In[210]:


COCO_CLASSES.index('dining table')


# In[211]:


len(COCO_CLASSES)


# In[212]:


# 類別顏色映射
num_classes = len(COCO_CLASSES)
colors = np.random.rand(num_classes, 3)  # 隨機顏色生成 (0-1)


# In[213]:


import torch
import torchvision.transforms as T
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor


# In[214]:


# 設定裝置（如果沒有 GPU，則使用 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[110]:


# 載入 DETR 預訓練模型
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
model.eval()


# In[215]:


# 載入 COCO Tiny 數據集標註
with open("./data/coco128/annotations.json", "r", encoding="utf-8") as f:
    coco_data = json.load(f)


# In[216]:


def pred(image, confidence_score = 0.9):
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # 取得預測結果
    target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=confidence_score)[0]
    return results


# In[217]:


def plot_(image, results):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    
    # 繪製圖片 & 預測結果
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(image)
    
    # 確保這些 tensor 是在 CPU 上的
    for score, label, box in zip(results["scores"].cpu().numpy(), 
                                  results["labels"].cpu().numpy(), 
                                  results["boxes"].cpu().numpy()):
        x0, y0, x1, y1 = box
        w = x1-x0
        h = y1-y0
        color = colors[label]  # 根據 label 來選擇顏色
        rect = patches.Rectangle((x0, y0), w, h, linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        
        # 顯示類別名稱和分數
        class_name = COCO_CLASSES[label]  # 使用對應的類別名稱
        ax.text(x0, y0, f"{class_name} ({score:.2f})", fontsize=10, color=color)
    
    plt.show()


# In[218]:


# 轉換圖片格式（符合 DETR 輸入要求）
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
for idx in range(30,35):
    image_info = coco_data["images"][idx]  # 取第一張圖
    image_path =image_info['file_name']

    # 讀取圖片
    image = Image.open(image_path).convert("RGB")
    
    print(idx, image_path)
    results = pred(image, confidence_score = 0.9)
    print(results)
    plot_(image,results)


# In[174]:


idx = 1
coco_data["images"][idx]


# In[175]:


# 隨機選擇一張圖片
image_info = coco_data["images"][idx]  # 取第一張圖
image_path =image_info['file_name']
print(image_path)


# In[176]:


# 讀取圖片
image = Image.open(image_path).convert("RGB")
# 轉換圖片格式（符合 DETR 輸入要求）
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
inputs = processor(images=image, return_tensors="pt").to(device)


# In[177]:


# Forward pass
with torch.no_grad():
    outputs = model(**inputs)


# In[178]:


inputs.keys()


# In[179]:


outputs.keys()


# In[180]:


outputs['logits']


# In[181]:


# 取得預測結果
target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]


# In[182]:


results


# In[183]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# 繪製圖片 & 預測結果
fig, ax = plt.subplots(1, figsize=(8, 6))
ax.imshow(image)

# 確保這些 tensor 是在 CPU 上的
for score, label, box in zip(results["scores"].cpu().numpy(), 
                              results["labels"].cpu().numpy(), 
                              results["boxes"].cpu().numpy()):
    print(label)
    x0, y0, x1, y1 = box
    w = x1-x0
    h = y1-y0
    color = colors[label]  # 根據 label 來選擇顏色
    rect = patches.Rectangle((x0, y0), w, h, linewidth=2, edgecolor=color, facecolor="none")
    ax.add_patch(rect)
    
    # 顯示類別名稱和分數
    # label = label - 2
    print(label)
    class_name = COCO_CLASSES[label]  # 使用對應的類別名稱
    ax.text(x0, y0, f"{class_name} ({score:.2f})", fontsize=10, color=color)

plt.show()


# In[1]:


# ## 以下是試跑能不能模型跑個結果
# import torch
# import torchvision.models as models
# import torchvision.transforms as T
# from PIL import Image

# # 設定裝置
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device

# # 載入 ResNet-18 預訓練模型
# model = models.resnet18(pretrained=True).to(device)
# model.eval()

# # 圖片前處理（轉換成 ResNet-18 可接受的格式）
# transform = T.Compose([
#     T.Resize((224, 224)),  # ResNet-18 輸入大小固定為 224x224
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 均值 & 標準差
# ])

# # 載入 COCO Tiny 的其中一張圖片
# image_path = "./data/coco128/images/train2017/000000000009.jpg"  # 請換成你資料夾內的圖片
# image = Image.open(image_path).convert("RGB")
# input_tensor = transform(image).unsqueeze(0).to(device)  # 增加 batch 維度
# # Forward pass
# with torch.no_grad():
#     features = model(input_tensor)

# # 印出 ResNet-18 的輸出
# print("ResNet-18 輸出特徵向量大小:", features.shape)


# In[ ]:





# In[ ]:





# In[ ]:




