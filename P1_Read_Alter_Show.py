import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1)Read image
silverhand = cv2.imread("johnny.jpg")

# 2)Resize
resized = cv2.resize(silverhand, (750, 500))

# 3)Convert to grayscale
grayTest = cv2.cvtColor(silverhand, cv2.COLOR_BGR2GRAY)

# 4)Edge detection
edgeTest = cv2.Canny(silverhand, 50, 50)

# 5)Resize variations
silverhandSmall = cv2.resize(silverhand, (0, 0), fx=0.05, fy=0.05)
grayTest = cv2.resize(grayTest, (200, 400))
edgeTest = cv2.resize(edgeTest, (780, 540))

# 6)Blurred
blurred = cv2.GaussianBlur(silverhand, (15, 15), 0)  
# 7)Sharpening
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened = cv2.filter2D(silverhand, -1, sharpen_kernel)  
# 8)Thresholding
_, thresholded = cv2.threshold(grayTest, 127, 255, cv2.THRESH_BINARY)  

# 9)Invert
negative = cv2.bitwise_not(silverhand)  

# 10)Dilated
dilated = cv2.dilate(edgeTest, np.ones((3, 3), np.uint8), iterations=1)  


titles = [
    "Original", "Small Resize", "Resized", "Grayscale", "Edged", 
    "Blurred", "Sharpened", "Thresholded", "Negative", "Dilated"
]
images = [
    silverhand, silverhandSmall, resized, grayTest, edgeTest, 
    blurred, sharpened, thresholded, negative, dilated
]

# 11)Plot all in one
plt.figure(figsize=(12, 8))
for i in range(len(images)):
    plt.subplot(2, 5, i + 1)
    plt.title(titles[i])
    plt.imshow(images[i], cmap="gray" if len(images[i].shape) == 2 else None)
    plt.axis("off")

plt.show()
