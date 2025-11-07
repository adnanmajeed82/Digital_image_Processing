import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

print("Current Working Directory:", os.getcwd())

# Step 1: Read image safely
img_path = r'E:\env\messaging_app\opencv\flower.jpg'  # Update path
img = cv2.imread(img_path)

if img is None:
    print("❌ Error: Image not found at", img_path)
    exit()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Step 2: Reshape for K-means
pixel_vals = img.reshape((-1, 3))
pixel_vals = np.float32(pixel_vals)

# Step 3: K-means clustering
k = 3
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(pixel_vals)
centers = np.uint8(kmeans.cluster_centers_)

segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape(img.shape)

# Step 4: Display
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(segmented_image)
plt.title(f'Segmented Image (k={k})')
plt.axis('off')
plt.show()
