# MSSV: N20DCCN031
# Nguyễn Nhật Kha
# Lớp: D20DCCNPM01-N
import numpy as np
import matplotlib.pyplot as plt

arr=np.array(list(float(y) for y in range(0,8)))
[COLUMN ,ROW] = np.meshgrid(arr,arr)
u0 = 2.0
v0 = 2.0

def format_print(name, data):
    print(f'{name}: ')
    print(data)
def create_real(input):
    return np.round(np.real(input) * (10**4)) * (10**(-4))
def create_imaginary(input):
    return np.round(np.imag(input) * (10**4)) * (10**(-4))
def show2img(img1, title1, img2, title2):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title(title1)
    plt.imshow(img1,cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title(title2)
    plt.imshow(img2,cmap='gray')
    plt.show()
def create_full_contrast(input):
    min_val = np.min(input)
    max_val = np.max(input)
    if(min_val == max_val):
        return np.zeros(input.shape)
    scale = 255/(max_val - min_val)
    return np.round((input - min_val) * scale)
def create_center(input):
    return np.fft.fftshift(np.fft.fft2(input))
def create_I1(m,n):
    I1=0.5 * np.exp(2j*np.pi/8*(u0*m + v0*n))
    return I1

I1 = create_I1(COLUMN,ROW)
I1_real = create_real(I1)
I1_imag = create_imaginary(I1)
format_print("I1 Real", I1_real)
format_print("I1 Imaginary", I1_imag)
show2img(create_full_contrast(I1_real),"I1 Real",create_full_contrast(I1_imag),"I1 Imaginary")

Itilde1 = create_center(I1)
Itilde1_real = create_real(Itilde1)
Itilde1_imag = create_imaginary(Itilde1)
format_print("Re[DFT(I1)]", Itilde1_real)
format_print("Im[DFT(I1)]", Itilde1_imag)