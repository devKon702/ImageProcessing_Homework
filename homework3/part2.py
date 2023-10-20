# Nguyễn Nhật Kha
# D20CQCNPM01-N
# N20DCCN031
import numpy as np
import matplotlib.pyplot as plt
# Đọc hình ảnh (8bits, 256x256)
lady_image = np.fromfile("homework3/lady.bin", dtype=np.uint8).reshape(256, 256)

# Show hình ảnh gốc
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.imshow(lady_image, cmap='gray')
plt.title('Hình ảnh gốc')

plt.subplot(2, 2, 2)
plt.hist(lady_image.ravel(), bins=256, range=(0, 256), density=True, color='gray', alpha=0.7)
plt.title('Histogram hình ảnh gốc')
# Hàm chuyển đôi full-scale contrast stretch
def full_scale_constrast_stretch(image):
    min_val = np.min(lady_image)
    max_val = np.max(lady_image)
    res = image.copy()
    MP=2**8-1
    R=max_val-min_val
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            res[i,j]=round((res[i,j]-min_val)/R*MP)
    return res

stretched_image = full_scale_constrast_stretch(lady_image)
# Show hình ảnh đã cân bằng
plt.subplot(2, 2, 3)
plt.imshow(stretched_image, cmap='gray')
plt.title('Hình ảnh sau khi cân bằng')

plt.subplot(2, 2, 4)
plt.hist(stretched_image.ravel(), bins=256, range=(0, 256), density=True, color='gray', alpha=0.7)
plt.title('Histogram sau khi cân bằng')

plt.show()
