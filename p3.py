import cv2
import numpy as np

image = cv2.imread('sekiro.jpg') 
cv2.imshow("Original Image", image)

M = np.float32([[1, 0, 100], [0, 1, 50]])
translated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

center = (image.shape[1] // 2, image.shape[0] // 2)
M = cv2.getRotationMatrix2D(center, 45, 1)
rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

upscaled_image = cv2.resize(image, (0,0), fx=1.5, fy=1.5)

cv2.imshow("Translated Image", translated_image)
cv2.imshow("Rotated Image", rotated_image)
cv2.imshow("Upscaled Image", upscaled_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
