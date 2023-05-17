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

#Linear Transformation for 2D points
p = np.array([2, 4])
Sx, Sy = 3, 0.5
S = np.array([[Sx, 0], [0, Sy]])
S

p_dash = S.dot(p) #dot product of S and p / matrix multiplication
p_dash #results are a floating point number

#for 3D points
p = np.array([2, 4, 6])
Sx, Sy, Sz = 3, 0.5, 2
S = np.array([[Sx, 0, 0], [0, Sy, 0], [0, 0, Sz]])
p_dash = S.dot(p)
p_dash

#Image Copy and Flipping Vertically
I_gray_copy = I_gray.copy()
plt.imshow(I_gray_copy, cmap = 'gray')
plt.show()
numRows = I_gray_copy.shape[0]
numCols = I_gray_copy.shape[1]
print(numRows, numCols)
I_gray2 = np.zeros((numRows, numCols), dtype = 'uint8') #create a new image of all zeros (black)
plt.imshow(I_gray2, cmap = 'gray')
plt.show()

for i in range(numRows):
    for j in range(numCols):
        I_gray2[i, j] = I_gray[i, j] #copy the image pixel by pixel
plt.imshow(I_gray2, cmap = 'gray')
plt.show()

#flip the image vertically
for i in range(numRows):
    for j in range(numCols):
        I_gray2[numRows - i - 1, j] = I_gray[i, j]
plt.imshow(I_gray2, cmap = 'gray')
plt.show()

#flip the image horizontally
for i in range(numRows):
    for j in range(numCols):
        I_gray2[i, numCols - j - 1] = I_gray[i, j]
plt.imshow(I_gray2, cmap = 'gray')
plt.show()

#crop the image
I_gray2 = I_gray[0:475,]
plt.imshow(I_gray2, cmap = 'gray')
plt.show()

#also
for i in range(numRows/2):
    for j in range(numCols):
        I_gray2[numRows - i - 1, j] = I_gray[i, j]
plt.imshow(I_gray2, cmap = 'gray')
plt.show()

#image doubling and holes
s = np.array([[2, 0], [0, 2]]) #scaling matrix
I2 = np.zeros((2*numRows, 2*numCols), dtype = 'uint8') #create a new image of all zeros (black)
for i in range(numRows):
    for j in range(numCols):
        p = np.array([i, j])
        p_dash = s.dot(p)
        new_i = p_dash[0]
        new_j = p_dash[1]
        I2[new_i, new_j] = I_gray[i, j] #copy the image pixel by pixel
plt.imshow(I2, cmap = 'gray')
plt.show()

#create function to show image in actual size
print(I2.shape) #shape has doubled
def displayImageInActualSize(I):
    dpi = plt.rcParams['figure.dpi']
    h, w = I.shape
    figSize = w/float(dpi) , h/float(dpi)
    fig = plt.figure(figsize = figSize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(I, cmap = 'gray')
    plt.show()

displayImageInActualSize(I2)