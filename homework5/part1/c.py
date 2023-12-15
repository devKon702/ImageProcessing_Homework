# MSSV: N20DCCN031
# Nguyễn Nhật Kha
# Lớp: D20DCCNPM01-N
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

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

# Create the zero-padded original image
padded_original = np.zeros((512, 512))
padded_original[:256,:256] = original_image

# Make the 256x256 impulse response image H
H = np.zeros((256, 256), dtype=np.float32)
H[125:132, 125:132] = 1/49
# Get the true zero-phase impulse response image
H2 = np.fft.fftshift(H)

# Create the zero-padded impulse response image
padded_H = np.zeros((512, 512))
padded_H[:128, :128] = H2[:128, :128]
padded_H[:128, 384:512] = H2[:128, 128:256]
padded_H[384:512, :128] = H2[128:256, :128]
padded_H[384:512, 384:512] = H2[128:256, 128:256]

# Compute the filtered result by pointwise multiplication of DFT's
final_output = np.fft.ifft2(np.fft.fft2(padded_original) * np.fft.fft2(padded_H))
# Drop padding and take the real part
final_output = final_output[:256, :256].real

plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
plt.title("Original Image")
plt.imshow(original_image, cmap="gray")
plt.axis("off")

plt.subplot(2,2,2)
plt.title("256x256 H")
plt.imshow(H2, cmap='gray')
plt.axis('off')

plt.subplot(2,2,3)
plt.title("512x512 zero-padded H")
plt.imshow(padded_H, cmap='gray')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(final_output, cmap='gray')
plt.title('Final Filtered Image')
plt.axis('off')

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