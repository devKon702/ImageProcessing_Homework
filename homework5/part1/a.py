# MSSV: N20DCCN031
# Nguyễn Nhật Kha
# Lớp: D20DCCNPM01-N
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def linear_average_filter(image):
    padded_image = cv2.copyMakeBorder(image, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=0)

    result = np.zeros((262, 262))
    # Convolution
    for row in range(3, 259):
        for col in range(3, 259):
            result[row, col] = np.sum(padded_image[row-3:row+4, col-3:col+4]) / 49
    # Drop padding
    result = result[3:259, 3:259]
    return result

def create_full_contrast(input):
    min_val = np.min(input)
    max_val = np.max(input)
    if min_val == max_val:
        return np.zeros_like(input, dtype=np.uint8)
    scale = 255/(max_val - min_val)
    normalized = np.round(((input - min_val) * scale))
    return normalized

# Load image
parent_directory = os.path.dirname(os.path.abspath(__file__))
image = np.fromfile(f"{parent_directory}/salesman.bin", dtype=np.uint8).reshape(256,256)

# Apply linear average filter
filtered_image = linear_average_filter(image)

# Show images
plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
plt.imshow(create_full_contrast(image),cmap="gray")
plt.axis("off")
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(create_full_contrast(filtered_image),cmap="gray")
plt.axis("off")
plt.title("Filtered Image")
plt.show()
