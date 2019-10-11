# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:51:22 2019

@author: PateauTech_2
"""
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL.Image import *
#imagePath = "C:/Users/PateauTech_2/Documents/Photos_Macaques_queue_de_lion/rhesus-macaques-960x540.jpg"
imagePath = "C:/Users/PateauTech_2/Desktop/Positive_Image/Face/onlycenter33.jpg"

img =open(imagePath)
x, y = img.size

img.crop((0, 0, x/2, y/3)).save('C:/Users/PateauTech_2/Desktop/Positive_Image/image1_eye_right.png')
eyeRPath = "C:/Users/PateauTech_2/Desktop/Positive_Image/image1_eye_right.png"

img.crop((x/2, 0, x, y/3)).save('C:/Users/PateauTech_2/Desktop/Positive_Image/image1_eye_left.png')
eyeLPath = "C:/Users/PateauTech_2/Desktop/Positive_Image/image1_eye_left.png"

img.crop((x/3, y/3, 2*x/3, 2*y/3)).save('C:/Users/PateauTech_2/Desktop/Positive_Image/image1_nose.png')
nosePath = "C:/Users/PateauTech_2/Desktop/Positive_Image/image1_nose.png"

img.crop((0, 2*y/3, x, y)).save('C:/Users/PateauTech_2/Desktop/Positive_Image/image1_mouth.png')
mouthPath = "C:/Users/PateauTech_2/Desktop/Positive_Image/image1_mouth.png"

# Detecteur de Canny; contours

img = cv2.imread(imagePath,0)
edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()



# Detecteur de Harris; contours

img = cv2.imread(eyeRPath)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv2.imwrite('C:/Users/PateauTech_2/Desktop/Positive_Image/image.png',cv2.imshow('dst',img)) 

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
    
    
    
# Sobel Derivatives
    
def main(argv):
    
    window_name = ('Sobel Demo - Simple Edge Detector')
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    
    

    if len(argv) < 1:
        print ('Not enough parameters')
        print ('Usage:\nmorph_lines_detection.py < path_to_image >')
        return -1
    # Load the image
    src = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image: ' + argv[0])
        return -1
    
    
    src = cv2.GaussianBlur(src, (3, 3), 0)
    
    
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    
#    cv2.imshow(window_name, grad)
#    cv2.waitKey(0)
    
    return grad

gray = main(imagePath)

#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
lines = cv2.HoughLines(edges,1,np.pi/180,200)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite('houghlines3.jpg',img)

