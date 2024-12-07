import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from scipy.ndimage import convolve

# Fungsi untuk deteksi tepi menggunakan Roberts
def roberts_edge_detection(image):
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])
    gx = convolve(image, kernel_x)
    gy = convolve(image, kernel_y)
    return np.sqrt(gx**2 + gy**2)

# Fungsi untuk deteksi tepi menggunakan Sobel
def sobel_edge_detection(image):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gx = convolve(image, kernel_x)
    gy = convolve(image, kernel_y)
    return np.sqrt(gx**2 + gy**2)

# Load gambar grayscale
image = imread("C:\\Tugas_PCD\\Tugas_Sesi9\\th (1).jpg", mode="F")

# Normalisasi gambar
image = image / 255.0

# Deteksi tepi menggunakan Roberts dan Sobel
edges_roberts = roberts_edge_detection(image)
edges_sobel = sobel_edge_detection(image)

# Plot hasil
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Roberts Edge Detection')
plt.imshow(edges_roberts, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Sobel Edge Detection')
plt.imshow(edges_sobel, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
