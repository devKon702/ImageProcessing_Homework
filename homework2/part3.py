# Nguyễn Nhật Kha
# D20CQCNPM01-N
# N20DCCN031
import cv2
import matplotlib.pyplot as plt

J1 = cv2.imread("homework2/lena512color.jpg")

J2 = J1.copy()
J2[:, :, 0] = J1[:, :, 2]  
J2[:, :, 1] = J1[:, :, 0]
J2[:, :, 2] = J1[:, :, 1]

plt.figure(figsize=(8, 4))
plt.imshow(J2)
plt.title("J2 Image")
plt.show()

cv2.imwrite("homework2/lena512color_swapped.jpg", J2)
