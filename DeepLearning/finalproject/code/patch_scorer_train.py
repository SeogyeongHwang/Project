
import os
import zipfile
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import time
import datetime

from google.colab import drive

drive.mount('/content/drive')

# 압축 해제
zip_path = '/content/drive/MyDrive/Deep learning/Final Project/COCO.zip'
unzip_dir = '/content/COCO'
if not os.path.exists(unzip_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)
    print("압축 해제 완료")
else:
    print("이미 압축 해제됨")

# 디바이스 설정 및 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 중인 디바이스:", device)
if device.type == 'cuda':
    print("GPU(CUDA)가 성공적으로 연결되었습니다.")
    print("CUDA 디바이스 이름:", torch.cuda.get_device_name(0))
else:
    print("GPU(CUDA)를 사용할 수 없습니다. CPU로 실행됩니다.")

# 이미지 → 텐서 직접 변환 함수
def img_to_tensor(img):
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    return torch.tensor(img_np, dtype=torch.float32)

# PatchScorer 모델 정의
class PatchScorer(nn.Module):
    def __init__(self):
        super(PatchScorer, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(1)

# PatchDataset 정의
class PatchDataset(Dataset):
    def __init__(self, root_dir, patch_size=8):
        self.root_dir = root_dir
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.png')])
        self.patch_size = patch_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        patches, scores = [], []
        for i in range(0, h, self.patch_size):
            for j in range(0, w, self.patch_size):
                patch = img.crop((j, i, j+self.patch_size, i+self.patch_size))
                patch_tensor = img_to_tensor(patch)
                gray = cv2.cvtColor((patch_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                magnitude = np.sqrt(gx**2 + gy**2)
                score = np.sum(magnitude) / (self.patch_size**2)
                patches.append(patch_tensor)
                scores.append(score)
        return torch.stack(patches), torch.tensor(scores, dtype=torch.float32), self.image_files[idx]

# 학습 설정
train_path = os.path.join(unzip_dir, 'Train')
dataset = PatchDataset(train_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model = PatchScorer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# 학습 시작
start_time = datetime.datetime.now()
print(f"학습 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

epochs = 100
loss_history = []  # 에폭별 loss 저장용
model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    sample_count = 0
    for patches, targets, _ in dataloader:
        patches = patches.squeeze(0).to(device)
        targets = targets.squeeze(0).to(device)
        preds = model(patches)
        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        sample_count += 1
    avg_epoch_loss = epoch_loss / sample_count
    loss_history.append(avg_epoch_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_epoch_loss:.4f}")

    # 모델 저장 (5번마다 + 마지막 에폭)
    if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
        save_path = f"/content/drive/MyDrive/Deep learning/Final Project/k_8/patch_scorer_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        elapsed = datetime.datetime.now() - start_time
        minutes = elapsed.total_seconds() / 60
        print(f"모델 저장됨: {save_path} (경과 시간: {minutes:.2f}분)")
