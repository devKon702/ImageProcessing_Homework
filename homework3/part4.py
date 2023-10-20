import numpy as np
import matplotlib.pyplot as plt

# (1) Đọc hình ảnh "johnny.bin" (256x256 pixel, 8 bit/pixel)
johnny_image = np.fromfile("homework3/johnny.bin", dtype=np.uint8).reshape(256, 256)

# (2) Tạo biểu đồ histogram cho hình ảnh gốc
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(johnny_image, cmap='gray')
plt.title('Hình ảnh gốc')

plt.subplot(1, 2, 2)
plt.hist(johnny_image.ravel(), bins=256, range=(0, 256), density=True, color='gray', alpha=0.7)
plt.title('Histogram hình ảnh gốc')

# (3) Thực hiện histogram equalization
hist, bins = np.histogram(johnny_image.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf = (cdf / cdf[-1]) * 255
equalized_image = cdf[johnny_image]

# (4) Tạo biểu đồ histogram cho hình ảnh đã cân bằng
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(equalized_image, cmap='gray')
plt.title('Hình ảnh sau khi cân bằng')

plt.subplot(1, 2, 2)
plt.hist(equalized_image.ravel(), bins=256, range=(0, 256), density=True, color='gray', alpha=0.7)
plt.title('Histogram sau khi cân bằng')

plt.show()
