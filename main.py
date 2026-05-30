#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from zipfile import ZipFile
from PIL import Image
import warnings
import os
import pandas as pd
import numpy as np


# In[ ]:


"""with ZipFile('open (1).zip') as zip_ref:
    zip_ref.extractall('physical')"""


# In[ ]:


train = pd.read_csv('physical/train.csv')
dev = pd.read_csv('physical/dev.csv')
test = pd.read_csv('physical/sample_submission.csv')


# In[ ]:


train.head()


# In[ ]:


train['label'].value_counts()


# In[ ]:


dev['label'].value_counts()


# In[ ]:


train['label'] = (train['label'] != 'unstable').astype(int)
dev['label'] = (dev['label'] != 'unstable').astype(int)


# In[ ]:


# train데이터를 기본으로 학습을 시키고, dev데이터를 val 느낌으로 사용 - test가 dev와 구도가 동일하기 때문
# rotate는 중력의 방향이 변형되니 제외
# 이진데이터에서 train 비율이 동일 - 그냥 바로 가능
# 속도를 위해 glob으로 먼저 빼줌


# In[ ]:


class PhysicalTrainDataset(Dataset):
    def __init__(self, df, targets=None, transform=None, root_dir="physical/train"):
        self.df = df
        self.targets = targets  # list/np.array/torch tensor or None
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_id = str(self.df.iloc[idx]["id"])  # 예: "0001"
        folder = os.path.join(self.root_dir, sample_id)

        img_front = Image.open(os.path.join(folder, "front.png")).convert("RGB")
        img_top   = Image.open(os.path.join(folder, "top.png")).convert("RGB")
        img = np.array(img_top)
        img2 = np.array(img_front)

        # transform이 "두 장을 같이" 처리할 수 있으면 그게 베스트
        if self.transform:
            augmented = self.transform(image=img, image_front=img2)
            top = augmented["image"]
            front = augmented["image_front"]

        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return top, front, y


# In[ ]:


class PhysicaldevDataset(Dataset):
    def __init__(self, df, targets=None, transform=None, root_dir="physical/dev"):
        self.df = df.reset_index(drop=True)
        self.targets = targets  # list/np.array/torch tensor or None
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_id = str(self.df.iloc[idx]["id"])  # 예: "0001"
        folder = os.path.join(self.root_dir, sample_id)

        img_front = Image.open(os.path.join(folder, "front.png")).convert("RGB")
        img_top   = Image.open(os.path.join(folder, "top.png")).convert("RGB")
        img = np.array(img_top)
        img2 = np.array(img_front)
        if self.transform:
            front = self.transform(image=img2)['image']
            top = self.transform(image=img)['image']

        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return top, front, y


# In[ ]:


class PhysicaltestDataset(Dataset):
    def __init__(self, df, targets=None, transform=None, root_dir="physical/test"):
        self.df = df.reset_index(drop=True)
        self.targets = targets  # list/np.array/torch tensor or None
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_id = str(self.df.iloc[idx]["id"])  # 예: "0001"
        folder = os.path.join(self.root_dir, sample_id)

        img_front = Image.open(os.path.join(folder, "front.png")).convert("RGB")
        img_top   = Image.open(os.path.join(folder, "top.png")).convert("RGB")
        img = np.array(img_top)
        img2 = np.array(img_front)
        # transform이 "두 장을 같이" 처리할 수 있으면 그게 베스트
        if self.transform:
            front = self.transform(image=img2)['image']
            top = self.transform(image=img)['image']

        return top, front, sample_id


# In[ ]:


import albumentations as A
from albumentations.pytorch import ToTensorV2
train_transformer = A.Compose(
    [
        A.Resize(384, 384),
        A.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
            p=0.8
        ),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            border_mode=0,
            p=0.6
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ],
    additional_targets={"image_front": "image"}
)
dev_transformer = A.Compose([
    A.Resize(384, 384),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.ToTensorV2()
])


# In[ ]:


train_dataset = PhysicalTrainDataset(
    train,
    train['label'].values,
    transform=train_transformer
)
dev_dataset = PhysicaldevDataset(
    dev,
    dev['label'].values,
    transform=dev_transformer
)
test_dataset = PhysicaltestDataset(
    test,
    test['id'].values,
    transform=dev_transformer
)


# In[ ]:


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,#train은 무조건 섞어줘야한다
    num_workers=4,
    pin_memory=True
)
dev_loader = DataLoader(
    dev_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# In[ ]:


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

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim1 + feat_dim2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )
        """
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim1 + feat_dim2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(256,1)"""

    def forward(self, x1, x2):
        feat1 = self.model1(x1)   # (B, 512)
        feat2 = self.model2(x2)   # (B, 512)

        feat = torch.cat([feat1, feat2], dim=1)  # (B, 1024)# devlos 0.38, epoch 11
        #feat = self.fusion(feat)

        logit = self.classifier(feat)            # (B, 1)

        return logit.squeeze(1)


# In[ ]:


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


# In[ ]:


model = TwoImageTwoModel().to('cuda')
freeze_all_but_fc(model)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4
        )
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer= optimizer, lr_lambda=lambda epoch : 0.95**epoch, last_epoch= -1)


# In[ ]:


class EarlyStopping:
    def __init__(self, patience=3, delta=0.0, path='best_model.pt', verbose=True):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss  # loss는 낮을수록 좋으니까 음수로 바꿔서 비교

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss improved ({self.best_loss:.6f} -> {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.path)
        self.best_loss = val_loss


# In[ ]:


#find Temperature
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        # temperature가 너무 작아지는 것 방지
        temp = torch.clamp(self.temperature, min=1e-3)
        return logits / temp


def fit_temperature(logits, labels, max_iter=100):
    """
    logits: torch.Tensor of shape (N,)
    labels: torch.Tensor of shape (N,)
    """
    device = logits.to('cuda')
    scaler = TemperatureScaler().to('cuda')

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.LBFGS(scaler.parameters(), lr=0.01, max_iter=max_iter)

    logits = logits.detach()
    labels = labels.float().detach()

    def closure():
        optimizer.zero_grad()
        scaled_logits = scaler(logits)
        loss = criterion(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    T = torch.clamp(scaler.temperature.detach(), min=1e-3).item()
    return T


# In[ ]:


def collect_logits_and_labels(model, loader):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x1, x2, y in loader:
            x1 = x1.to('cuda')
            x2 = x2.to('cuda')
            y = y.to('cuda').float()

            logits = model(x1, x2)   # shape: (B,)
            all_logits.append(logits)
            all_labels.append(y)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_logits, all_labels


# In[ ]:


dev_logits, dev_labels = collect_logits_and_labels(model, dev_loader)
best_T = fit_temperature(dev_logits, dev_labels)

print("Best temperature:", best_T)


# In[ ]:


# 모델 val 데이터 평가
from sklearn.metrics import log_loss

def devloss(model, loader, criterion):
    model.eval()
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for images1, images2, targets in dev_loader:
            images1, images2 = images1.to('cuda'), images2.to('cuda')#gpu이동
            targets = targets.float().to('cuda')

            outputs = model(images1,images2)#출력값
            probs = torch.sigmoid(outputs) # 시그모이드로 확률값으로 변환


            all_targets.extend(targets.cpu().numpy())#입력 차원이 n차원이면 자동으로 변경
            all_probs.extend(probs.cpu().numpy())
    all_targets = np.vstack(all_targets)#입력 차원이 n차원이면 자동으로 변경    실제 답
    all_probs = np.vstack(all_probs)#입력 차원이 n차원이면 자동으로 변경        예측값

    # 안정성 위해 clip
    y_pred = np.clip(all_probs, 1e-7, 1 - 1e-7)
    val_logloss = log_loss(all_targets, y_pred)
    return val_logloss


# In[ ]:


early = EarlyStopping(
    patience=4,
    delta = 0.0,
    path = 'best_model.pt',
    verbose=True
)
for epoch in range(50):
    model.train()
    total_loss = 0
    for images1, images2, targets in train_loader:
        images1, images2 = images1.to('cuda'), images2.to('cuda')
        targets = targets.float().view(-1,1).to('cuda')

        logit = model(images1,images2).unsqueeze(1)
        loss = criterion(logit, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    dev_loss = devloss(model, dev_loader, criterion)
    if epoch == 4:
        unfreeze_layer4_and_fc(model)
        optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-4,
                weight_decay=5e-4
                )

    early(dev_loss, model)
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, devloss: {dev_loss:.4f}')
    if early.early_stop:
        break
#epoch 11이 top


# In[ ]:


model.load_state_dict(torch.load('best_model(loss0.1).pt'))
model.eval()


# In[ ]:


class PhysicalfullDataset(Dataset):
    def __init__(self, df, targets=None, transform=None, root_dir="physical/"):
        self.df = df
        self.targets = targets  # list/np.array/torch tensor or None
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_id = str(self.df.iloc[idx]["id"])  # 예: "0001"
        species = sample_id.split('_')[0]
        if species == 'TRAIN':
            root = self.root_dir + str('train')
        else:
            root = self.root_dir + str('dev')
        folder = os.path.join(root, sample_id)

        img_front = Image.open(os.path.join(folder, "front.png")).convert("RGB")
        img_top   = Image.open(os.path.join(folder, "top.png")).convert("RGB")

        # transform이 "두 장을 같이" 처리할 수 있으면 그게 베스트
        if self.transform:
            img_top = self.transform(img_top)  # (top, front) 반환하도록
            img_front = self.transform(img_front)
        # targets가 없으면 추론용
        if self.targets is None:
            return img_top, img_front, sample_id

        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return img_top, img_front, y


# In[ ]:


#full train
full_df = pd.concat([train, dev], axis = 0).reset_index(drop = True)
full_dataset = PhysicalfullDataset(
    full_df,
    full_df['label'].values,
    transform = dev_transformer,
)
full_loader = DataLoader(
    full_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)


# In[ ]:


num_epochs = 16   # 이전에 best였던 epoch 수

model = TwoImageTwoModel().to('cuda')
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images1, images2, targets in full_loader:
        images1 = images1.to('cuda')
        images2 = images2.to('cuda')
        targets = targets.to('cuda')

        optimizer.zero_grad()

        logits = model(images1, images2)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images1.size(0)

    train_loss /= len(full_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}] | Full Train Loss: {train_loss:.4f}")


# In[ ]:


def inference(model, loader):
    model.eval()
    all_ids = []
    all_probs = []

    with torch.no_grad():
        for images1, images2, ids in loader:
            images1 = images1.to('cuda')
            images2 = images2.to('cuda')

            logits = model(images1, images2)              # (B,)
            probs = torch.sigmoid(logits/0.9995648860931396).cpu().numpy()   # 확률값

            all_probs.extend(probs.tolist())
            all_ids.extend(ids)
    return all_ids, all_probs


# In[ ]:


#test 제작
test_ids, test_probs = inference(model, test_loader)


# In[ ]:


test['id'] = test_ids
test['stable_prob'] = test_probs
test['unstable_prob'] = 1 - test['stable_prob']


# In[ ]:


test.to_csv('submisson2.csv', index = False)


# In[ ]:




