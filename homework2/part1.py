# Nguyễn Nhật Kha
# D20CQCNPM01-N
# N20DCCN031
import numpy as np
import matplotlib.pyplot as plt

# Đọc hình ảnh Lena và Peppers
lena_img = np.fromfile("homework2/lena.bin", dtype=np.uint8).reshape(256, 256)
peppers_img = np.fromfile("homework2/peppers.bin", dtype=np.uint8).reshape(256, 256)

plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.imshow(lena_img, cmap='gray')
plt.title('Lena Image')

plt.subplot(2, 2, 2)
plt.imshow(peppers_img, cmap='gray')
plt.title('Peppers Image')

# Tạo hình ảnh mới J
J = np.zeros((256, 256), dtype=np.uint8)
J[:, :128] = lena_img[:, :128]
J[:, 128:] = peppers_img[:, 128:]

# Tạo hình ảnh mới K
K = J.copy()
K[:, :128] = J[:, 128:]
K[:, 128:] = J[:, :128]

# Hiển thị hình ảnh J và K
plt.subplot(2, 2, 3)
plt.imshow(J, cmap='gray')
plt.title('J Image')

plt.subplot(2, 2, 4)
plt.imshow(K, cmap='gray')
plt.title('K Image')

plt.show()
