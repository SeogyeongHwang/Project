import numpy as np
import cv2
import func

# 1) Read the given images, and collect the dataset from random sub-blocks.
img = cv2.imread("image/canvas1/canvas1-a-p001.png", cv2.IMREAD_GRAYSCALE)
block_array = func.get_block(img, 3)
print(block_array.shape)

# 2) Let the image value at the center pixel as the ground-truth label (𝑦),
# and its adjacent pixel values as feature vector (𝐱). Here, you can consider any size of sub-blocks (e.g., 3×3 or 5×5).
X1, y1 = func.get_data(block_array, "causal")
print("causal: ")
print("X shape:", X1.shape)
print("y shape:", y1.shape)

X2, y2 = func.get_data(block_array, "non-causal")
print("non-causal:")
print("X shape:", X2.shape)
print("y shape:", y2.shape)

# 3) Generate the feature matrix and label vector as below. (Here, 𝐷 = 𝐾/2 or 𝐷 = 𝐾.)

# data를 담을 빈 array 생성
X = np.zeros((1, 4), dtype=int)
y = np.zeros(1, dtype=int)

# 다양한 image로부터 정보 추출
for i in range(1, 41):
    img = cv2.imread("image/canvas1/canvas1-a-p" + str(i).zfill(3) + ".png", cv2.IMREAD_GRAYSCALE)
    block_array = func.get_block(img, 3)
    append_X, append_y = func.get_data(block_array, "causal")
    X = np.append(X, append_X, axis=0)
    y = np.append(y, append_y, axis=0)

# 처음 선언한 array 삭제
X = np.delete(X, [0, 0], axis=0)
y = np.delete(y, [0], axis=0)

# 넘파이 파일로 저장
np.save('canvas_causal_X.npy', X)
np.save('canvas_causal_y.npy', y)
