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
def create_I3(m,n):
    I3=np.cos((2*np.pi/8)*(u0*m+v0*n))
    return I3

I3 = create_I3(COLUMN,ROW)
plt.figure(figsize=(10,10))
plt.title("I3")
plt.imshow(create_full_contrast(I3), cmap='gray')
plt.show()

Itilde3 = create_center(I3)
Itilde3_real = create_real(Itilde3)
Itilde3_imag = create_imaginary(Itilde3)
format_print("Re[DFT(I3)]", Itilde3_real)
format_print("Im[DFT(I3)]", Itilde3_imag)