# MSSV: N20DCCN031
# Nguyễn Nhật Kha
# Lớp: D20DCCNPM01-N
import numpy as np
import os
from matplotlib import pyplot as plt

def get_file(name):
    current_file_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(current_file_path)
    return np.fromfile(f"{parent_directory}/{name}", dtype=np.uint8)

def show_group_result(image_title, original, median, opened, closed):
    plt.figure(figsize=(10,8))
    plt.subplot(2,2,1)
    plt.title(image_title)
    plt.imshow(original, cmap="gray")
    plt.axis("off")

    plt.subplot(2,2,2)
    plt.title("Median Filter")
    plt.imshow(median, cmap="gray")
    plt.axis("off")

    plt.subplot(2,2,3)
    plt.title("Morphological Opening")
    plt.imshow(opened, cmap="gray")
    plt.axis("off")

    plt.subplot(2,2,4)
    plt.title("Morphological Closing")
    plt.imshow(closed, cmap="gray")
    plt.axis("off")

    plt.show()

def compute(image):
    wsize = 3
    wsizeo2 = wsize // 2

    filter_window = np.zeros((3, 3))
    size = 256
    median_img = np.zeros((size, size))
    eroded_img = np.zeros((size, size))
    dilated_img = np.zeros((size, size))
    opened_img = np.zeros((size, size))
    closed_img = np.zeros((size, size))

    # Apply median, erode, and dilate
    for row in range(wsizeo2, size - wsizeo2):
        for col in range(wsizeo2, size - wsizeo2):
            filter_window = image[row - wsizeo2:row + wsizeo2 + 1, col - wsizeo2:col + wsizeo2 + 1]

            median_img[row, col] = np.median(filter_window)
            eroded_img[row, col] = np.min(filter_window)
            dilated_img[row, col] = np.max(filter_window)

    # Apply dilate to eroded_img and erode to dilated_img
    for row in range(wsizeo2 + 1, size - wsizeo2 - 1):
        for col in range(wsizeo2 + 1, size - wsizeo2 - 1):
            filter_window = eroded_img[row - wsizeo2:row + wsizeo2 + 1, col - wsizeo2:col + wsizeo2 + 1]
            opened_img[row, col] = np.max(filter_window)

            filter_window = dilated_img[row - wsizeo2:row + wsizeo2 + 1, col - wsizeo2:col + wsizeo2 + 1]
            closed_img[row, col] = np.min(filter_window)

    return [median_img, opened_img, closed_img]

camera9 = get_file("camera9.bin").reshape(256,256)
camera99 = get_file("camera99.bin").reshape(256,256)

[median9,opened9,closed9] = compute(camera9)
show_group_result("Camera9.bin", camera9, median9, opened9, closed9)

[median99,opened99,closed99] = compute(camera99)
show_group_result("Camera99.bin", camera99, median99, opened99, closed99)
