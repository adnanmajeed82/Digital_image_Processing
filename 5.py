import cv2
import numpy as np

# Step 1: Load Image
img = cv2.imread(r"E:\env\messaging_app\opencv\flower.jpg", 0)


if img is None:
    print("Error: Image not found. Check the file path.")
    exit()

# Step 2: Apply Negative Transformation
negative_img = 255 - img

# Step 3: Brightness Adjustment
bright_img = cv2.add(img, 50)  # Increase intensity by 50

# Step 4: Gamma Correction
gamma = 1.5
gamma_img = np.array(255*(img/255)**gamma, dtype='uint8')

# Step 5: Display Images
cv2.imshow('Original', img)
cv2.imshow('Negative', negative_img)
cv2.imshow('Bright', bright_img)
cv2.imshow('Gamma Corrected', gamma_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
