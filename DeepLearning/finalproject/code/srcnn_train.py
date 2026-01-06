# SRCNN 구조 구현 - Lab9 내용 기반
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import zipfile
import random

from google.colab import drive
drive.mount('/content/drive')

# COCO.zip 압축 해제
zip_path = '/content/drive/MyDrive/COCO.zip'
unzip_dir = '/content/COCO'
if not os.path.exists(unzip_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)
    print("COCO.zip 압축 해제 완료")
else:
    print("COCO.zip 이미 압축 해제됨")

# SRCNN 모델 정의
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# 혼합 해상도 입력을 위한 사용자 정의 데이터셋
class MixedSRDataset(Dataset):
    def __init__(self, image_dir):
        self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        hr = self.to_tensor(img)
        base_size = img.height

        # 0.25 혹은 0.5로 scale 조정해 이미지 사이즈 128x128, 256x256에 따른 각각 train model 만들기
        scale = 0.25
        down_h = int(base_size * scale)
        down_w = int(img.width * scale)

        lr_img = transforms.Resize((down_h, down_w), interpolation=Image.BICUBIC)(img)
        lr_img = transforms.Resize((base_size, base_size), interpolation=Image.BICUBIC)(lr_img)
        lr = self.to_tensor(lr_img)

        return lr, hr

# 학습 함수 정의
def train_srcnn(model, dataloader, optimizer, criterion, device, epochs=10, save_path_base='/content/drive/MyDrive/Deep learning/Final Project/srcnn_model'):
    model.train()
    start_time = datetime.datetime.now()
    print(f"학습 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0
        for lr, hr in dataloader:
            lr, hr = lr.to(device), hr.to(device)
            pred = model(lr)
            loss = criterion(pred, hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

        # 모델 저장 (5에폭마다 저장)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            elapsed = datetime.datetime.now() - start_time
            minutes = elapsed.total_seconds() / 60
            save_path = f"{save_path_base}_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"모델 저장됨: {save_path} (경과 시간: {minutes:.2f}분)")

    end_time = datetime.datetime.now()
    print(f"학습 종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    total_duration = end_time - start_time
    print(f"총 학습 시간: {str(total_duration)}")

    # Loss 시각화
    plt.figure()
    plt.plot(range(1, epochs + 1), loss_history, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('/content/drive/MyDrive/srcnn_loss_plot.png')
    plt.show()

    # 테스트 이미지 4개 복원
    model.eval()
    test_dir = '/content/COCO/Test'
    test_files = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.png')])[:4]
    for i, path in enumerate(test_files):
        img = Image.open(path).convert('L')
        base_size = img.height
        scale = random.choice([0.25, 0.5])
        down = transforms.Resize((int(base_size * scale), int(base_size * scale)), interpolation=Image.BICUBIC)(img)
        up = transforms.Resize((base_size, base_size), interpolation=Image.BICUBIC)(down)
        input_tensor = transforms.ToTensor()(up).unsqueeze(0).to(device)
        with torch.no_grad():
            output_tensor = model(input_tensor)
        output_image = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)

        plt.figure()
        plt.title(f'Reconstructed Test Image {i+1}')
        plt.imshow(output_image.squeeze(), cmap='gray')
        plt.axis('off')
        plt.savefig(f'/content/drive/MyDrive/reconstructed_test_{i+1}.png')
        plt.show()

# 예시 실행 코드 (사용 시 주석 해제)
""
# 하이퍼파라미터 설정
image_dir = '/content/COCO/Train'  # COCO 데이터셋 경로
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 혼합 해상도 데이터셋 및 데이터로더 준비
dataset = MixedSRDataset(image_dir)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 모델 초기화 및 학습
model = SRCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

train_srcnn(model, dataloader, optimizer, criterion, device, epochs=100)

