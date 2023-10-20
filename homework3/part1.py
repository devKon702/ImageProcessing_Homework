# Nguyễn Nhật Kha
# D20CQCNPM01-N
# N20DCCN031
import numpy as np
import matplotlib.pyplot as plt
# Đọc hình ảnh (8bits, 256x256)
mammogram = np.fromfile("homework3/Mammogram1.bin", dtype=np.uint8).reshape(256, 256)
# Chọn ngưỡng threshold và biến đổi ảnh nhị phân
threshold = 128
binary_image = np.where(mammogram > threshold,255,0)
# Hàm tạo hình ảnh biên nhị phân
def approximate_contour_image(binary_image):
    contour_image = np.zeros_like(binary_image)
    for i in range(1, binary_image.shape[0] - 1):
        for j in range(1, binary_image.shape[1] - 1):
            if binary_image[i, j] == 255 and (
                binary_image[i - 1, j] == 0
                or binary_image[i + 1, j] == 0
                or binary_image[i, j - 1] == 0
                or binary_image[i, j + 1] == 0
            ):
                contour_image[i, j] = 255
    return contour_image

contour_image = approximate_contour_image(binary_image)

# Hiển thị hình ảnh
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(mammogram, cmap='gray')
plt.title('Hình ảnh gốc')

plt.subplot(1, 3, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('Hình ảnh nhị phân')

plt.subplot(1, 3, 3)
plt.imshow(contour_image, cmap='gray')
plt.title('Hình ảnh biên nhị phân')

plt.show()
