import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===========================================================
# DIGITAL IMAGE PROCESSING: Representation, Sampling, Quantization
# and Mathematics of Image Formation
# Author: Adnan Majeed
# ===========================================================

# --- Step 1: Image Representation ---
# Read a grayscale image (you can replace with your file)
img_path = r"E:\env\messaging_app\opencv\flower.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("❌ Image not found! Check your path.")
    exit()

# Display basic info
print("Image Shape (Rows, Columns):", img.shape)
print("Data Type:", img.dtype)
print("Max pixel value:", np.max(img))
print("Min pixel value:", np.min(img))

plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Grayscale Image")
plt.axis('off')

# --- Step 2: Sampling (Reduce Resolution) ---
# Sampling = reducing number of pixels (spatial resolution)
sampled_img = img[::4, ::4]  # take every 4th pixel
plt.subplot(2, 3, 2)
plt.imshow(sampled_img, cmap='gray')
plt.title("Sampled Image (1/4 Resolution)")
plt.axis('off')

# --- Step 3: Quantization (Reduce Gray Levels) ---
# Quantization = reducing intensity levels
levels = [256, 128, 64, 32, 16, 8]
quantized_images = []

for level in levels:
    quantized = np.floor(img / (256 / level)) * (256 / level)
    quantized = np.uint8(quantized)
    quantized_images.append(quantized)

# Display one example
plt.subplot(2, 3, 3)
plt.imshow(quantized_images[3], cmap='gray')
plt.title("Quantized Image (32 Gray Levels)")
plt.axis('off')

# --- Step 4: Pixel Intensity Histogram ---
plt.subplot(2, 3, 4)
plt.hist(img.ravel(), bins=256, color='black')
plt.title("Histogram of Original Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

# --- Step 5: Mathematics of Image Formation ---
# Simple model:  f(x, y) = i(x, y) * r(x, y)
# where i(x, y) = illumination, r(x, y) = reflectance

rows, cols = img.shape
x = np.linspace(0, 1, cols)
y = np.linspace(0, 1, rows)
X, Y = np.meshgrid(x, y)

# Simulate illumination (bright center, dark corners)
illumination = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) * 8)
# Simulate reflectance (sinusoidal pattern)
reflectance = 0.5 + 0.5 * np.sin(10 * np.pi * X)

# Image formation = multiplication
formed_image = illumination * reflectance
formed_image_scaled = (formed_image * 255).astype(np.uint8)

plt.subplot(2, 3, 5)
plt.imshow(illumination, cmap='gray')
plt.title("Illumination Component i(x, y)")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(formed_image_scaled, cmap='gray')
plt.title("Formed Image f(x, y) = i(x, y)*r(x, y)")
plt.axis('off')

plt.tight_layout()
plt.show()
