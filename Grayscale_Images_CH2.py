import os
os.getcwd()

import numpy as np
import matplotlib.pyplot as plt

#start with using numpy
im = np.arange(256) #create an array
im

im.shape

im = im[np.newaxis, :] #convert shape to array of 32 elements
im.shape

im = np.repeat(im, 100, axis = 0) #enlarge array to 100 rows
im.shape

im #shows the array in the terminal

plt.imshow(im, cmap = "gray") #build as an image
plt.show() #render the image


#processing grayscale images

#don't really need this chunk, just check the path and files available
#cwd = os.getcwd()  # Get the current working directory (cwd)
#files = os.listdir(cwd)  # Get all the files in that directory
#print("Files in %r: %s" % (cwd, files))

#imread isn't used anymore, use PIL.Image.open
#im = plt.imread(r'C:/Users/dusti/anaconda3/envs/CV_Course/albert-einstein_gray.jpg')

from PIL import Image
im2 = Image.open(r'albert-einstein_gray.jpg')
im2.show()

type(im2)
im2 = np.array(im2) #convert to numpy array, the way the course shows doesn't work with PIL
im2.shape
im2.dtype

plt.imshow(im2, cmap = 'gray')
plt.show() #rendered image should show on a graph as it's an array now

im2 #to show the matrix of values

im2[23, 300] #get the color code value for this position in the matrix

im2[23:100, 40:100] = 255 #replace these positions with pure white
plt.imshow(im2, cmap = 'gray')
plt.show()

im2[300:400, 40:100] = 0 #replace these positions with pure black
plt.imshow(im2, cmap = 'gray')
plt.show()

plt.imsave(r'albert-einstein_gray_modified.jpg', im2, cmap = 'gray') #save the image as a jpeg in the path


#running the same process with OpenCV
#install opencv-contrib-python in the conda env with pip not conda

#Grayscale Images in OpenCV
import cv2
#you also need matplotlib for OpenCV, but we already have it installed

img = cv2.imread(r'albert-einstein_gray.jpg', cv2.IMREAD_GRAYSCALE) #imread still works with OpenCV
type(img)
img.dtype
img.shape
img[23, 100] = 200
plt.imshow(img, cmap = 'gray') #render the image
plt.show()

cv2.imshow('Gray', img) #use this chunk to render the image in OpenCV
cv2.waitKey(0)
cv2.destroyAllWindows()

img[500:700, 500:600] = 255 #replace part of the image with white
cv2.imshow('Gray', img) #use this chunk to render the image in OpenCV
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(r'albert-einstein_gray_opencv.jpg', img)


#RGB Images 
#using Matplotlib
cim = plt.imread(r'tulips.jpg') #imread is working now.  not sure what's going on with the first iamge above
plt.imshow(cim)
plt.show()

cim.shape #output as (row, col, channels)
type(cim)
cim.dtype

#diplay image by channels with subplots
R = cim[:, :, 0]
G = cim[:, :, 1]
B = cim[:, :, 2]

plt.figure(1)
plt.subplot(231) #2x3 grid, 1st plot
plt.imshow(cim)
plt.xticks([])
plt.yticks([])

plt.subplot(232) #2x3 grid, 2nd plot
plt.imshow(cim)
plt.xticks([])
plt.yticks([])

plt.subplot(233) #2x3 grid, 3rd plot
plt.imshow(cim)
plt.xticks([])
plt.yticks([])

plt.subplot(234) #2x3 grid, 4th plot, brightness in image is contribution from that channel
plt.imshow(R, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.title('Red')

plt.subplot(235) #2x3 grid, 5th plot
plt.imshow(G, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.title('Green')

plt.subplot(236) #2x3 grid, 6th plot
plt.imshow(B, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.title('Blue')
plt.show()


#Using OpenCV which translates RGB images as BGR
cim = cv2.imread(r'tulips.jpg') #imread is working now.  not sure what's going on with the first iamge above
plt.imshow(cim)
plt.show()

cim.shape #output as (row, col, channels)
type(cim)
cim.dtype

#diplay image by channels with subplots
cim = cim[:, :, :: -1] #reversing the order of the channels due to OpenCV being backwards
R = cim[:, :, 0]
G = cim[:, :, 1]
B = cim[:, :, 2]

plt.figure(1)
plt.subplot(231) #2x3 grid, 1st plot
plt.imshow(cim)
plt.xticks([])
plt.yticks([])

plt.subplot(232) #2x3 grid, 2nd plot
plt.imshow(cim)
plt.xticks([])
plt.yticks([])

plt.subplot(233) #2x3 grid, 3rd plot
plt.imshow(cim)
plt.xticks([])
plt.yticks([])

plt.subplot(234) #2x3 grid, 4th plot, brightness in image is contribution from that channel
plt.imshow(R, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.title('Red')

plt.subplot(235) #2x3 grid, 5th plot
plt.imshow(G, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.title('Green')

plt.subplot(236) #2x3 grid, 6th plot
plt.imshow(B, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.title('Blue')
plt.show()

#resetting the values for part of the image to only red
R[100:400, 100:400] = 255
G[100:400, 100:400] = 0
B[100:400, 100:400] = 0
cim[:, :, 0] = R
cim[:, :, 1] = G
cim[:, :, 2] = B

plt.figure(1)
plt.subplot(231) #2x3 grid, 1st plot
plt.imshow(cim)
plt.xticks([])
plt.yticks([])

plt.subplot(232) #2x3 grid, 2nd plot
plt.imshow(cim)
plt.xticks([])
plt.yticks([])

plt.subplot(233) #2x3 grid, 3rd plot
plt.imshow(cim)
plt.xticks([])
plt.yticks([])

plt.subplot(234) #2x3 grid, 4th plot, brightness in image is contribution from that channel
plt.imshow(R, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.title('Red')

plt.subplot(235) #2x3 grid, 5th plot
plt.imshow(G, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.title('Green')

plt.subplot(236) #2x3 grid, 6th plot
plt.imshow(B, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.title('Blue')
plt.show()

#quiz - making a new image, then manipulating it to be RGB in Matplotlib
rgb = np.zeros((100,150,3), dtype='uint8')
rgb[:,0:50,0] = 255 #red
rgb[:,50:100,1] = 255 #green
rgb[:,100:150,2] = 255 #blue
 
plt.imshow(rgb)
plt.show()

#OpenCV - notice that the colors are backwards
cv2.imshow('RGB', rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()