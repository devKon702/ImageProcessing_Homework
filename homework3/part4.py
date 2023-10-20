# Nguyễn Nhật Kha
# D20CQCNPM01-N
# N20DCCN031
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Đọc hình ảnh (8bits, 256x256)
johnny_image = np.fromfile("homework3/johnny.bin", dtype=np.uint8).reshape(256, 256)

# Show ảnh gốc
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.imshow(johnny_image, cmap='gray')
plt.title('Hình ảnh gốc')

plt.subplot(2, 2, 2)
plt.hist(johnny_image.ravel(), bins=256, range=(0, 256), density=True, color='gray', alpha=0.7)
plt.title('Histogram hình ảnh gốc')

# Show ảnh Equalize
equalize_img = cv2.equalizeHist(johnny_image)
plt.subplot(2, 2, 3)
plt.imshow(equalize_img, cmap='gray')
plt.title('Hình ảnh sau khi cân bằng')

plt.subplot(2, 2, 4)
plt.hist(equalize_img.ravel(), bins=256, range=(0, 256), density=True, color='gray', alpha=0.7)
plt.title('Histogram sau khi cân bằng')

plt.show()
