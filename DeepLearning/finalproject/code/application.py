import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import zipfile

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

# ---------- 1. PatchScorer 정의 ----------
class PatchScorer(nn.Module):
    def __init__(self):
        super(PatchScorer, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(1)

# ---------- 2. SRCNN 정의 ----------
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# ---------- 3. 공통 함수 ----------
def img_to_tensor(img):
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1)) 
    return torch.tensor(img_np, dtype=torch.float32)

def split_into_patches(img_tensor, patch_size=32):
    _, H, W = img_tensor.shape
    patches, positions = [], []
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = img_tensor[:, i:i+patch_size, j:j+patch_size]
            if patch.shape[1] == patch_size and patch.shape[2] == patch_size:
                patches.append(patch)
                positions.append((i, j))
    return patches, positions

def region_selection(image_tensor, model, device, patch_size=32, top_k_each=16):
    patches, positions = split_into_patches(image_tensor, patch_size)
    model.eval()
    
    with torch.no_grad():
        input_batch = torch.stack(patches).to(device)
        scores = model(input_batch).cpu().numpy()

    sorted_indices = np.argsort(-scores)  # 중요도 높은 순서로 정렬
    region_levels = {}

    for idx, sorted_idx in enumerate(sorted_indices):
        pos = positions[sorted_idx]
        if idx < top_k_each:
            level = 512  # 가장 중요한 16개
        elif idx < 2 * top_k_each:
            level = 256  # 그 다음 16개
        else:
            level = 128  # 나머지는 전체 128x128 이미지에서 사용
        region_levels[pos] = level

    return region_levels


# ---------- 4. PSNR 및 파라미터 함수 ----------
def MSE(img1, img2):
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    if img1.shape[0] == 3:
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.shape[0] == 3:
        img2 = np.transpose(img2, (1, 2, 0))
    img1 = np.clip(img1 * 255.0, 0, 255)
    img2 = np.clip(img2 * 255.0, 0, 255)
    return np.mean((img1 - img2) ** 2)

def PSNR(img1, img2):
    mse = MSE(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / (mse ** 0.5))

# 파라미터 수 계산 함수
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    return total

# ---------- 5. 최종 통합 복원 함수 ----------
def reconstruct_with_patch_selection(img_path, patch_scorer, srcnn_model256, srcnn_model128, device, save_path=True, visualize_heatmap=True, compute_psnr=True):
    patch_size = 32
    img = Image.open(img_path).convert('RGB').resize((512, 512))
    img_tensor_512 = img_to_tensor(img)
    img_tensor_256 = img_to_tensor(img.resize((256, 256), Image.BICUBIC).resize((512, 512), Image.BICUBIC))
    img_tensor_128 = img_to_tensor(img.resize((128, 128), Image.BICUBIC).resize((512, 512), Image.BICUBIC))

    region_levels = region_selection(img_tensor_512, patch_scorer, device)
    _, positions = split_into_patches(img_tensor_512, patch_size)

    output_canvas = torch.zeros((3, 512, 512), dtype=torch.float32)
    count_canvas = torch.zeros((3, 512, 512), dtype=torch.float32)
    heatmap = np.zeros((512, 512), dtype=np.uint8)

    srcnn_model256.eval()
    srcnn_model128.eval()
    for pos in positions:
        i, j = pos
        level = region_levels.get(pos, 128)

        source_img = {
            512: img_tensor_512,
            256: img_tensor_256,
            128: img_tensor_128
        }[level]

        patch = source_img[:, i:i+patch_size, j:j+patch_size].unsqueeze(0).to(device)

        with torch.no_grad():
          if level == 512:
            out_patch = patch.squeeze(0).cpu()
          elif level == 256:
            out_patch = srcnn_model256(patch).squeeze(0).cpu()
          elif level == 128:
            out_patch = srcnn_model128(patch).squeeze(0).cpu()

        output_canvas[:, i:i+patch_size, j:j+patch_size] += out_patch
        count_canvas[:, i:i+patch_size, j:j+patch_size] += 1

        heat_value = {
            512: 255,
            256: 160,
            128: 80
        }[level]
        heatmap[i:i+patch_size, j:j+patch_size] = heat_value

    final_output = output_canvas / torch.clamp(count_canvas, min=1e-8)
    final_image = final_output.permute(1, 2, 0).numpy()
    final_image = np.clip(final_image, 0, 1)

    # PSNR 계산
    if compute_psnr:
        psnr = PSNR(img_tensor_512, final_output)
        print(f"PSNR: {psnr:.2f} dB")

    # 복원 이미지 저장
    if save_path:
        save_img = Image.fromarray((final_image * 255).astype(np.uint8))
        save_img.save('/content/drive/MyDrive/result.png')
        print(f"복원 이미지 저장됨: {save_path}")

    # heatmap 시각화
    if visualize_heatmap:
        plt.figure(figsize=(6, 6))
        plt.imshow(heatmap, cmap='viridis')
        plt.title("Patch Importance Level Map (512=Yellow, 256=Green, 128=Dark)")
        plt.axis("off")
        plt.colorbar()
        plt.show()

    # 최종 이미지 시각화
    plt.figure(figsize=(6, 6))
    plt.imshow(final_image)
    plt.title("Final Reconstructed Image (RGB)")
    plt.axis("off")
    plt.show()

    return final_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

patch_scorer = PatchScorer().to(device)
patch_scorer.load_state_dict(torch.load('/content/drive/MyDrive/train_model/train_model/patch_scorer_epoch65.pth', map_location=device))

srcnn_model_256 = SRCNN().to(device)
srcnn_model_256.load_state_dict(torch.load('/content/drive/MyDrive/srcnn/256/srcnn_model256_epoch45.pth', map_location=device))

srcnn_model_128 = SRCNN().to(device)
srcnn_model_128.load_state_dict(torch.load('/content/drive/MyDrive/srcnn/128/srcnn_model_epoch45.pth', map_location=device))

# 파라미터 수 확인
print("PatchScorer 파라미터 수")
count_parameters(patch_scorer)

print("\nSRCNN(256용) 파라미터 수")
count_parameters(srcnn_model_256)

print("\nSRCNN(128용) 파라미터 수")
count_parameters(srcnn_model_128)

reconstruct_with_patch_selection(
    img_path='/content/COCO/Test/4500.png',
    patch_scorer=patch_scorer,
    srcnn_model256=srcnn_model_256,
    srcnn_model128=srcnn_model_128,
    device=device,
)
