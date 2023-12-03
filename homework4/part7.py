# MSSV: N20DCCN031
# Nguyễn Nhật Kha
# Lớp: D20DCCNPM01-N
import numpy as np
import matplotlib.pyplot as plt

def get_img(path):
    return np.fromfile(path,dtype=np.uint8).reshape(256,256)
def create_real(input):
    return np.round(np.real(input) * (10**4)) * (10**(-4))
def create_full_contrast(input):
    min_val = np.min(input)
    max_val = np.max(input)
    if(min_val == max_val):
        return np.zeros(input.shape)
    scale = 255/(max_val - min_val)
    return np.round((input - min_val) * scale)
def show2img(img1, title1, img2, title2):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title(title1)
    plt.imshow(img1,cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title(title2)
    plt.imshow(img2,cmap='gray')
    plt.show()

I6 = get_img("camera.bin")
DFT_I6 = np.fft.fft2(I6)
mag = np.abs(DFT_I6)
phase = np.angle(DFT_I6)
J1 = create_real(np.fft.ifft2(mag))
J2 = create_real(np.fft.ifft2(np.exp(1j * phase)))
JJ1 = np.log(J1)
show2img(create_full_contrast(JJ1), "JJ1", create_full_contrast(J2), "J2")