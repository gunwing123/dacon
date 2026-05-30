#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss


# In[2]:


train = pd.read_csv('physical/train.csv')
dev = pd.read_csv('physical/dev.csv')
test = pd.read_csv('physical/sample_submission.csv')
train['label'] = (train['label'] != 'unstable').astype(int)
dev['label'] = (dev['label'] != 'unstable').astype(int)


# In[4]:


def video_frames(video_path, num_frames = 4):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    indices = [int(frame_count * i/(num_frames+1)) for i in range(1, num_frames+1)]

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()
    return frames


# In[ ]:


class PhysicalvideoDataset(Dataset):
    def __init__(self, df, targets=None, transform=None, root_dir="physical"):
        self.df = df
        self.targets = targets  # list/np.array/torch tensor or None
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_id = str(self.df.iloc[idx]["id"])  # 예: "0001"
        folder = os.path.join(self.root_dir, sample_id)

        #path 설정
        img_front = Image.open(os.path.join(folder, "front.png")).convert("RGB")
        img_top   = Image.open(os.path.join(folder, "top.png")).convert("RGB")
        video_path = f"{folder}/simulation.mp4"

        #video frame
        frames = video_frames(video_path)
        frames = [Image.fromarray(f) for  f in frames]

        # transform이 "두 장을 같이" 처리할 수 있으면 그게 베스트
        if self.transform:
            img_top = self.transform(img_top)  # (top, front) 반환하도록
            img_front = self.transform(img_front)
            frames = [self.transform(f) for  f in frames]
        frames = torch.stack(frames)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return img_top, img_front, frames, target


# In[6]:


train_transformer = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomPerspective(distortion_scale=1, p = 0.95),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees= (-10,10), scale=(0.8,1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dev_transformer = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# In[ ]:


train_dataset = PhysicalvideoDataset(
    train,
    train['label'].values,
    transform=train_transformer
)


# In[25]:


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,#train은 무조건 섞어줘야한다
    num_workers=4,
    pin_memory=True
)
"""
dev_loader = DataLoader(
    dev_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)"""


# In[26]:


class VideoEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, frames):
        B, T, C, H, W = frames.shape

        frames = frames.view(B*T, C, H, W)

        feat = self.backbone(frames)  # (B*T,512)

        feat = feat.view(B, T, -1)
        feat = feat.mean(dim=1)       # (B,512)

        return feat


# In[27]:


#pretrained model
#resnet 18
from torchvision.models import resnet18, ResNet18_Weights

def make_resnet18_encoder():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Identity()   # 마지막 fc 제거 -> feature vector만 뽑음
    return model, in_features

class TwoImageTwoModel(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()

        self.model1, feat_dim1 = make_resnet18_encoder()
        self.model2, feat_dim2 = make_resnet18_encoder()
        self.video = VideoEncoder(self.model1)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim1 + feat_dim2 + feat_dim1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )

    def forward(self, x1, x2, x3):
        feat1 = self.model1(x1)   # (B, 512)
        feat2 = self.model2(x2)   # (B, 512)
        feat3 = self.video(x3)

        feat = torch.cat([feat1, feat2, feat3], dim=1)  # (B, 1024)

        logit = self.classifier(feat)            # (B, 1)

        return logit.squeeze(1)


# In[28]:


def freeze_all_but_fc(model):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

def unfreeze_layer4_and_fc(model):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.model1.layer4.parameters():
        param.requires_grad = True
    for param in model.model2.layer4.parameters():
        param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True


# In[29]:


model = TwoImageTwoModel().to('cuda')
freeze_all_but_fc(model)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4
        )
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer= optimizer, lr_lambda=lambda epoch : 0.95**epoch, last_epoch= -1)


# In[30]:


for epoch in range(10):
    model.train()
    total_loss = 0
    for top, front, frames, target in train_loader:
        top = top.to('cuda')
        front = front.to('cuda')
        frames = frames.to('cuda')
        target = target.to('cuda')

        optimizer.zero_grad()

        logit = model(top, front, frames)

        loss = criterion(logit, target)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')#devloss: {dev_loss:.4f


# In[ ]:





# In[ ]:




