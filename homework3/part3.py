import numpy as np
import matplotlib.pyplot as plt
import cv2

# (1) Đọc hình ảnh "actontBin.bin" (256x256 pixel, 8 bit/pixel)
actont_image = np.fromfile("homework3/actontBin.bin", dtype=np.uint8).reshape(256, 256)

# (2) Thiết kế mẫu "T" dựa trên phân tích hình ảnh
# Ví dụ: Thiết kế mẫu "T" với băng ngang ở giữa và thanh dọc ở trung tâm

# (3) Thực hiện Binary Template Matching
template = np.zeros((7, 7), dtype=np.uint8)
template[3, :] = 1
template[:, 3] = 1

match_measure = cv2.matchTemplate(actont_image, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.7  # Ngưỡng tùy chọn
binary_image = (match_measure >= threshold).astype(np.uint8) * 255

# (4) Hiển thị hình ảnh binary
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(actont_image, cmap='gray')
plt.title('Hình ảnh gốc')

plt.subplot(1, 2, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('Hình ảnh binary sau Template Matching')

plt.show()
