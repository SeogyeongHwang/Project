import numpy as np


def get_block(image, k):
    kh = k // 2
    random = np.random.rand(image.shape[0], image.shape[1])
    block_array = np.zeros(k**2)
    for i in range(kh, image.shape[0] - kh):
        for j in range(kh, image.shape[1] - kh):
            if random[i, j] < 0.01:
                block = image[i - kh:i + kh + 1, j - kh:j + kh + 1]
                block = block.flatten()
                block_array = np.vstack((block_array, block))
    block_array = np.delete(block_array, [0, 0], axis=0)
    return block_array


def get_data(block_array, type):
    size = block_array.shape[1] // 2
    X = block_array[:, :size]
    y = block_array[:, size]
    if type == "causal":
        return X, y
    elif type == "non-causal":
        X = np.append(X, block_array[:, size+1:], axis=1)
        return X, y
    else:
        print("Wrong type")
        return 0

