# Nguyễn Nhật Kha
# D20CQCNPM01-N
# N20DCCN031
import matplotlib.pyplot as plt
import cv2
images = ["dental.jpg","parrot.jpg","skull.jpg"]
for i in range(len(images)):
    original_img = cv2.imread("homework1/" + images[i],cv2.IMREAD_GRAYSCALE)
    ahe_8 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ahe_16 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    res_8 = ahe_8.apply(original_img)
    res_16 = ahe_16.apply(original_img)
    plt.figure(4, figsize=(8,12))
    plt.subplot(3,3,3*i+1)
    plt.title("Original") 
    plt.axis("off")
    plt.imshow(original_img,cmap='gray')

    plt.subplot(3,3,3*i+2)
    plt.title("AHE 8x8")
    plt.axis("off")
    plt.imshow(res_8,cmap='gray')

    plt.subplot(3,3,3*i+3)
    plt.title("AHE 16x16")
    plt.axis("off")
    plt.imshow(res_16,cmap='gray')
plt.show()
