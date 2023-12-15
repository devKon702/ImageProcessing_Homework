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

girl = get_file("girl2.bin").reshape(256,256)
girl_noise = get_file("girl2Noise32.bin").reshape(256,256)
girl_hi_noise = get_file("girl2Noise32Hi.bin").reshape(256,256)

# Compute MSE
MSE_hi_noise = np.mean((girl_hi_noise - girl) ** 2)
print(f'MSE of girl2Noise32Hi: {MSE_hi_noise}')

MSE_noise = np.mean((girl_noise - girl) ** 2)
print(f'MSE of girl2Noise32: {MSE_noise}')

plt.figure(figsize=(10,8))
plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(girl, cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Girl2Noise32")
plt.imshow(girl_noise, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Girl2Noise32Hi")
plt.imshow(girl_hi_noise, cmap="gray")
plt.axis("off")
plt.show()
