# MSSV: N20DCCN031
# Nguyễn Nhật Kha
# Lớp: D20DCCNPM01-N
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def create_full_contrast(input):
    min_val = np.min(input)
    max_val = np.max(input)
    if(min_val == max_val):
        return np.zeros(input.shape, dtype=np.uint8)
    scale = 255/(max_val - min_val)
    return np.round(((input - min_val) * scale))

# Load image
parent_directory = os.path.dirname(os.path.abspath(__file__))
original_image = np.fromfile(f"{parent_directory}/salesman.bin", dtype=np.uint8).reshape(256,256)

# Create impulse response H
H = np.zeros((128, 128), dtype=np.float32)
H[62:69, 62:69] = 1/49

# Zero-pad the original image
padded_original = cv2.copyMakeBorder(original_image, 0, 128, 0, 128, cv2.BORDER_CONSTANT, value=0)

# Zero-pad the impulse response H
zero_padded_H = cv2.copyMakeBorder(H, 0, 256, 0, 256, cv2.BORDER_CONSTANT, value=0)

# Compute the DFT
dft_original = np.fft.fft2(padded_original)
dft_H = np.fft.fft2(zero_padded_H, s=dft_original.shape)

# Compute the centered log-manitude
centered_original = np.log(1 + np.abs(dft_original))
centered_H = np.log(1 + np.abs(dft_H))

# Compute convolution by the pointwise multiplication
convo_output =  dft_original * dft_H

# Compute the IDFT of the result
padded_output = np.abs(np.fft.ifft2(convo_output))

# Crop padding
final_output = padded_output[65:256+65, 65:256+65]

plt.figure(figsize=(10,8))
plt.subplot(2,4,1)
plt.imshow(create_full_contrast(original_image), cmap="gray")
plt.axis("off")
plt.title("Original")

plt.subplot(2,4,2)
plt.imshow(create_full_contrast(padded_original), cmap="gray")
plt.axis("off")
plt.title("Zero Padded Image")

plt.subplot(2,4,3)
plt.imshow(create_full_contrast(zero_padded_H), cmap="gray")
plt.axis("off")
plt.title("Zero Padded Impulse Resp")

plt.subplot(2,4,4)
plt.imshow(np.fft.fftshift(centered_original), cmap="gray")
plt.axis("off")
plt.title("Log spectrum of padded original")

plt.subplot(2,4,5)
plt.imshow(np.fft.fftshift(centered_H), cmap="gray")
plt.axis("off")
plt.title("Log spectrum of padded H")

plt.subplot(2,4,6)
plt.imshow(np.fft.fftshift(np.log(1 + np.abs(convo_output))), cmap="gray")
plt.axis("off")
plt.title("Log spectrum of padded result")

plt.subplot(2,4,7)
plt.imshow(create_full_contrast(padded_output), cmap="gray")
plt.axis("off")
plt.title("Zero Padded Result")

plt.subplot(2,4,8)
plt.imshow(create_full_contrast(final_output), cmap="gray")
plt.axis("off")
plt.title("Final Filtered Image")

plt.show()


def linear_average_filter(image):
    padded_image = cv2.copyMakeBorder(image, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=0)

    result = np.zeros((262, 262))
    # Convolution
    for row in range(3, 259):
        for col in range(3, 259):
            result[row, col] = np.sum(padded_image[row-3:row+4, col-3:col+4]) / 49
    # Drop padding
    result = result[3:259, 3:259]
    return result

# Get result of part a
A = linear_average_filter(original_image)

print("Max difference from part a:",end=' ') 
print(np.max(np.abs(create_full_contrast(final_output) - create_full_contrast(A))))