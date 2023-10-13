# Nguyễn Nhật Kha
# D20CQCNPM01-N
# N20DCCN031
import cv2
import matplotlib.pyplot as plt

J1 = cv2.imread("homework2/lenagray.jpg", cv2.IMREAD_GRAYSCALE)
J2 = 255 - J1

plt.figure(figsize=(8, 4))
plt.imshow(J2, cmap='gray')
plt.title("J2 Image")
plt.show()

cv2.imwrite("homework2/lenagray_negative.jpg", J2)
