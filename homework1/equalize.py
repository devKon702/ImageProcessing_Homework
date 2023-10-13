# Nguyễn Nhật Kha
# D20CQCNPM01-N
# N20DCCN031
import numpy as np
import matplotlib.pyplot as plt
import cv2
img = cv2.imread("./moon.jpg",cv2.IMREAD_GRAYSCALE)
equalized_image = cv2.equalizeHist(img)

img_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
equal_hist = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

plt.figure(figsize=(8, 6))

# Ảnh gốc
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

# Histogram của ảnh gốc
plt.subplot(2, 2, 2)
# plt.plot(img_hist, color='k')
plt.bar(range(256), img_hist[:,0])
plt.title('Histogram of Original Image')

# Ảnh đã cân bằng histogram
plt.subplot(2, 2, 3)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')

# Histogram của ảnh đã cân bằng histogram
plt.subplot(2, 2, 4)
# plt.plot(equal_hist, color='k')
plt.bar(range(256), equal_hist[:,0])
plt.title('Histogram of Equalized Image')

# Hiển thị các hình ảnh
plt.tight_layout()
plt.show()
