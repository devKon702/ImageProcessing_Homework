# MSSV: N20DCCN031
# Nguyễn Nhật Kha
# Lớp: D20DCCNPM01-N
import numpy as np
import matplotlib.pyplot as plt

def create_real(input):
    return np.round(np.real(input) * (10**4)) * (10**(-4))
def create_imaginary(input):
    return np.round(np.imag(input) * (10**4)) * (10**(-4))
def create_center(input):
    return np.fft.fftshift(np.fft.fft2(input))
def create_full_contrast(input):
    min_val = np.min(input)
    max_val = np.max(input)
    if(min_val == max_val):
        return np.zeros(input.shape)
    scale = 255/(max_val - min_val)
    return np.round((input - min_val) * scale)
def get_img(path):
    return np.fromfile(path,dtype=np.uint8).reshape(256,256)
def show_multi_image(img,row):
    row-=1
    plt.subplot(4, 5, row*5+1)
    if(row==0):
        plt.title("Origin")
    plt.imshow(create_full_contrast(img),cmap='gray')

    center_img = create_center(img)
    real_img = create_real(center_img)
    imag_img = create_imaginary(center_img)
    log_magnitude_spectrum = np.log(np.abs(center_img)+1)
    phase_spectrum = np.angle(center_img)

    plt.subplot(4, 5, row*5+2)
    if(row==0):
        plt.title("Real")
    plt.imshow(create_full_contrast(real_img),cmap='gray')

    plt.subplot(4, 5, row*5+3)
    if(row==0):
        plt.title("Imaginary")
    plt.imshow(create_full_contrast(imag_img),cmap='gray')

    plt.subplot(4, 5, row*5+4)
    if(row==0):
        plt.title("Log-magnitude")
    plt.imshow(create_full_contrast(log_magnitude_spectrum),cmap='gray')

    plt.subplot(4, 5, row*5+5)
    if(row==0):
        plt.title("Phase")
    plt.imshow(create_full_contrast(phase_spectrum),cmap='gray')

cam_img = get_img("camera.bin")
head_img = get_img("head.bin")
eyeR_img = get_img("eyeR.bin")
saleman_img = get_img("salesman.bin")

plt.figure(figsize=(14,12))
show_multi_image(cam_img,1)
show_multi_image(head_img,2)
show_multi_image(eyeR_img,3)
show_multi_image(saleman_img,4)
plt.show()