import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Read Image ---
img_path = r'E:\env\messaging_app\opencv\flower.jpg'  # <-- update path
img = cv2.imread(img_path)

if img is None:
    print("❌ Image not found. Check the file path.")
    exit()

# Convert BGR (OpenCV default) to RGB
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- Step 2: Convert to Grayscale ---
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Step 3: Convert to Binary (Global Thresholding) ---
# Pixels > 127 -> 255 (white), else -> 0 (black)
_, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

# --- Step 4: Convert to Black & White (Adaptive Thresholding) ---
bw_img = cv2.adaptiveThreshold(
    gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
)

# --- Step 5: Edge Detection (Canny) ---
edges_canny = cv2.Canny(gray_img, 100, 200)

# --- Step 6: Edge Detection (Sobel Operator) ---
sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobelx, sobely)

# --- Step 7: Display All Results ---
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(rgb_img)
plt.title('Original RGB Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(gray_img, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(binary_img, cmap='gray')
plt.title('Binary Image (Threshold=127)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(bw_img, cmap='gray')
plt.title('Black & White (Adaptive)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(edges_canny, cmap='gray')
plt.title('Edge Detection (Canny)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(sobel_combined, cmap='gray')
plt.title('Edge Detection (Sobel)')
plt.axis('off')

plt.tight_layout()
plt.show()
