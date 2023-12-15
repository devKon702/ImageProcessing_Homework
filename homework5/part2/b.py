# MSSV: N20DCCN031
# Nguyễn Nhật Kha
# Lớp: D20DCCNPM01-N
import numpy as np
import os
import matplotlib.pyplot as plt

# Load image
def get_file(name):
    current_file_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(current_file_path)
    return np.fromfile(f"{parent_directory}/{name}", dtype=np.uint8)

original = get_file("girl2.bin").reshape(256,256)
Noise = get_file("girl2Noise32.bin").reshape(256,256)
HiNoise = get_file("girl2Noise32Hi.bin").reshape(256,256)

U_cutoff = 64
U, V = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
HLtildeCenter = np.double(np.sqrt(U**2 + V**2) <= U_cutoff)
HLtilde = np.fft.fftshift(HLtildeCenter)

# Apply ideal low-pass filter to the original girl2 image
LPF_original = np.fft.ifft2(np.fft.fft2(original) * HLtilde).real
MSE_LPF_original = np.mean((LPF_original - original)**2)
print(f'MSE: ideal LPF on Original: {MSE_LPF_original}')

# Hi Pass Noise Image: Apply filter, compute MSE, ISNR
LPF_HiNoise = np.fft.ifft2(np.fft.fft2(HiNoise) * HLtilde).real
MSE_LPF_HiNoise = np.mean((LPF_HiNoise - original)**2)
ISNR_LPF_HiNoise = 10 * np.log10(np.mean((HiNoise - original)**2) / MSE_LPF_HiNoise)
print(f'MSE: ideal LPF on Noise32Hi: {MSE_LPF_HiNoise}')
print(f'ISNR: ideal LPF on Noise32Hi: {ISNR_LPF_HiNoise} dB')

# Noise Image: Apply filter, compute MSE, ISNR
LPF_Noise = np.fft.ifft2(np.fft.fft2(Noise) * HLtilde).real
MSE_LPF_Noise = np.mean((LPF_Noise - original)**2)
ISNR_LPF_Noise = 10 * np.log10(np.mean((Noise - original)**2) / MSE_LPF_Noise)
print(f'MSE: ideal LPF on Noise32: {MSE_LPF_Noise}')
print(f'ISNR: ideal LPF on Noise32: {ISNR_LPF_Noise} dB')

def create_full_contrast(input):
    min_val = np.min(input)
    max_val = np.max(input)
    if(min_val == max_val):
        return np.zeros(input.shape, dtype=np.uint8)
    scale = 255/(max_val - min_val)
    return np.round(((input - min_val) * scale))

# Display images
plt.figure(figsize=(10,8))
plt.subplot(1,3,1)
plt.title('LPF on Original')
plt.imshow(create_full_contrast(LPF_original), cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title('LPF on Girl2Noise32Hi')
plt.imshow(create_full_contrast(LPF_HiNoise), cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title('LPF on Girl2Noise32')
plt.imshow(create_full_contrast(LPF_Noise), cmap='gray')
plt.axis('off')

plt.show()
