import cv2
import matplotlib.pyplot as plt
import numpy as np

# Đọc hình ảnh
image = cv2.imread('homework_inclass/manly.png')

# Chuyển đổi hình ảnh sang không gian màu HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Xác định phạm vi màu
lower_bound = np.array([240, 100, 5]) # Giới hạn dưới cho màu xanh
upper_bound = np.array([250, 120, 20]) # Giới hạn trên cho màu xanh

# Tạo mặt nạ nhị phân bằng threshold
mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
# mask[mask == 0] = 255

print(mask)

# Hiển thị hình ảnh đã phân đoạn
plt.figure(4, figsize=(10,6))
plt.subplot(1,2,1)
plt.imshow(hsv_image)

plt.subplot(1,2,2)
plt.imshow(mask)
plt.show()


