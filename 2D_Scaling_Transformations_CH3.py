import os
os.getcwd()
os.chdir(r'C:\\Users\\dusti\\Documents\\CV_Course')

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

grayImage = r'C:\\Users\\dusti\\Documents\\CV_Course\\albert-einstein_gray.jpg'
colorImage = r'C:\\Users\\dusti\\Documents\\CV_Course\\tulips.jpg'

I_gray = cv2.imread(grayImage, cv2.IMREAD_GRAYSCALE)
I_BGR = cv2.imread(colorImage)

plt.imshow(I_gray, cmap = 'gray')
plt.show()

plt.imshow(I_BGR[:,:,::-1]) #[::-1] reverses the order of the color channels
plt.show()

#resize the image using cv2.resize
I_gray_resized = cv2.resize(src = I_gray, fx = 2, fy = 0.5, dsize = None) #fx and fy are the scaling factors
plt.imshow(I_gray_resized, cmap = 'gray')
plt.xticks([]) #remove the axis ticks
plt.yticks([]) #remove the axis ticks
plt.show()

I_gray_resized.shape
I_gray.shape

I_BGR_resized = cv2.resize(src = I_BGR, fx = 0.5, fy = 2, dsize = None)
plt.imshow(I_BGR_resized[:,:,::-1])
plt.xticks([])
plt.yticks([])
plt.show()

