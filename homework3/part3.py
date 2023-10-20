# Nguyễn Nhật Kha
# D20CQCNPM01-N
# N20DCCN031
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Đọc hình ảnh (8bits, 256x256)
actont_image = np.fromfile("homework3/actontBin.bin", dtype=np.uint8).reshape(256, 256)

# Mẫu T
template = np.zeros((50, 30), dtype=np.uint8)
template[5:15, :] = 1
template[15:45, 11:19] = 1
plt.figure(figsize=(8,4))
plt.subplot(1,1,1)
plt.imshow(template,cmap='gray')

J1 = cv2.matchTemplate(actont_image, template, cv2.TM_CCOEFF_NORMED)
print(J1)
J2 = np.where(J1 > 0.5, 255, 0)

plt.figure(figsize=(8, 4))
plt.subplot(1, 3, 1)
plt.imshow(actont_image, cmap='gray')
plt.title('Hình ảnh gốc')

plt.subplot(1, 3, 2)
plt.imshow(J1, cmap='gray')
plt.title('J1')

plt.subplot(1, 3, 3)
plt.imshow(J2, cmap='gray')
plt.title('J2')

plt.show()
