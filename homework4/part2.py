# MSSV: N20DCCN031
# Nguyễn Nhật Kha
# Lớp: D20DCCNPM01-N
import numpy as np
import matplotlib.pyplot as plt

arr=np.array(list(float(y) for y in range(0,8)))
[COLUMN ,ROW] = np.meshgrid(arr,arr)
u0 = 2.0
v0 = 2.0

def create_imaginary(input):
    return np.round(np.imag(input) * (10**4)) * (10**(-4))
def create_real(input):
    return np.round(np.real(input) * (10**4)) * (10**(-4))
def create_center(input):
    return np.fft.fftshift(np.fft.fft2(input))
def create_full_contrast(input):
    min_val = np.min(input)
    max_val = np.max(input)
    if(min_val == max_val):
        return np.zeros(input.shape)
    scale = 255/(max_val - min_val)
    return np.round((input - min_val) * scale)
def format_print(name, data):
    print(f'{name}: ')
    print(data)
def show2img(img1, title1, img2, title2):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title(title1)
    plt.imshow(img1,cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title(title2)
    plt.imshow(img2,cmap='gray')
    plt.show()
def create_I2(m,n):
    I2=0.5 * np.exp((-1)*2j*np.pi/8*(u0*m + v0*n))
    return I2

I2 = create_I2(COLUMN,ROW)
I2_real = create_real(I2)
I2_imag = create_imaginary(I2)
show2img(create_full_contrast(I2_real),"I2 Real", create_full_contrast(I2_imag),"I2 Imaginary")

Itilde2 = create_center(I2)
Itilde2_real = create_real(Itilde2)
Itilde2_imag = create_imaginary(Itilde2)
format_print("Re[DFT(I2)]", Itilde2_real)
format_print("Im[DFT(I2)]", Itilde2_imag)