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

U_cutoff_H = 64
SigmaH = 0.19 * 256 / U_cutoff_H
U, V = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
HtildeCenter = np.exp((-2 * np.pi**2 * SigmaH**2) / (256**2) * (U**2 + V**2))
Htilde = np.fft.fftshift(HtildeCenter)
H = np.fft.ifft2(Htilde).real
H2 = np.fft.fftshift(H)
ZPH2 = np.zeros((512, 512))
ZPH2[:256, :256] = H2

# Original: Apply Gaussian low-pass filter and compute MSE
ZP_original = np.zeros((512, 512))
ZP_original[:256, :256] = original
LPF_original = np.fft.ifft2(np.fft.fft2(ZP_original) * np.fft.fft2(ZPH2)).real
LPF_original = LPF_original[128:384, 128:384]
MSE_LPF_original = np.mean((LPF_original - original)**2)
print(f'MSE: Gaussian LPF on Original: {MSE_LPF_original}')

# Hi Pass Noise: Apply Gaussian low-pass filter, compute MSE and ISNR
ZP_HiNoise = np.zeros((512, 512))
ZP_HiNoise[:256, :256] = HiNoise
LPF_HiNoise = np.fft.ifft2(np.fft.fft2(ZP_HiNoise) * np.fft.fft2(ZPH2)).real
LPF_HiNoise = LPF_HiNoise[128:384, 128:384]
MSE_LPF_HiNoise = np.mean((LPF_HiNoise - original)**2)
print(f'MSE: Gaussian LPF on Noise32Hi: {MSE_LPF_HiNoise}')
ISNR_LPF_HiNoise = 10 * np.log10(np.mean((HiNoise - original)**2) / MSE_LPF_HiNoise)
print(f'ISNR: Gaussian LPF on Noise32Hi: {ISNR_LPF_HiNoise} dB')

# Noise: Apply Gaussian low-pass filter, compute MSE and ISNR
ZP_Noise = np.zeros((512, 512))
ZP_Noise[:256, :256] = Noise
LPF_Noise = np.fft.ifft2(np.fft.fft2(ZP_Noise) * np.fft.fft2(ZPH2)).real
LPF_Noise = LPF_Noise[128:384, 128:384]
MSE_LPF_Noise = np.mean((LPF_Noise - original)**2)
print(f'MSE: Gaussian LPF on Noise32: {MSE_LPF_Noise}')
ISNR_LPF_Noise = 10 * np.log10(np.mean((Noise - original)**2) / MSE_LPF_Noise)
print(f'ISNR: Gaussian LPF on Noise32: {ISNR_LPF_Noise} dB')

# Display images
plt.figure(figsize=(8,10))
plt.subplot(1,3,1)
plt.title('Gauss on Original')
plt.imshow(LPF_original, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title('Gauss on Noise32Hi')
plt.imshow(LPF_HiNoise, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title('Gauss on Noise32')
plt.imshow(LPF_Noise, cmap='gray')
plt.axis('off')

plt.show()
