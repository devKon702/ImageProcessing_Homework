# MSSV: N20DCCN031
# Nguyễn Nhật Kha
# Lớp: D20DCCNPM01-N
import numpy as np
import matplotlib.pyplot as plt

arr=np.array(list(float(y) for y in range(0,8)))
[COLUMN ,ROW] = np.meshgrid(arr,arr)
u1 = 1.5
v1 = 1.5

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
def create_I5(m,n):
    I5=np.cos((2*np.pi/8)*(u1*m+v1*n))
    return I5

I5 = create_I5(COLUMN,ROW)
plt.figure(figsize=(10,10))
plt.title("I5")
plt.imshow(create_full_contrast(I5),cmap='gray')
plt.show()

I5_real = create_real(I5)
I5_imag = create_imaginary(I5)
format_print("Re[(I5)]", I5_real)
format_print("Im[(I5)]", I5_imag)
