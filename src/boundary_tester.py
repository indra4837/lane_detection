import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("images/SGRoad.jpg")
plt.imshow(image)
print(image.shape)
plt.show()