# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:12:45 2019

@author: PateauTech_2
"""

# load libraries
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL.Image import *
import scipy
from scipy.stats import wasserstein_distance
from operator import itemgetter
#import dlib

class picture_treatment():
    
    # variable
    
    def __init__(self):
        #self.imagePath = np.loadtxt('C:/Users/PateauTech_2/.spyder-py3/positive_link.txt',dtype='str')
        self.imagePath = ["C:/Users/PateauTech_2/Desktop/Positive_Image/Face/29.jpg"]
    # function

    # import image and gray them
    def import_images(self,n):
        img = cv2.imread(n,0)
        return img
    
    # crop image
    def crop_images(self,n):
        img =open(n)
        x, y = img.size

        img.crop((0, 0, x/2, y/3)).save('C:/Users/PateauTech_2/Desktop/Positive_Image/image1_eye_right.png')
        eyeRPath = "C:/Users/PateauTech_2/Desktop/Positive_Image/image1_eye_right.png"
        
        img.crop((x/2, 0, x, y/3)).save('C:/Users/PateauTech_2/Desktop/Positive_Image/image1_eye_left.png')
        eyeLPath = "C:/Users/PateauTech_2/Desktop/Positive_Image/image1_eye_left.png"
        
        img.crop((x/3, y/3, 2*x/3, 2*y/3)).save('C:/Users/PateauTech_2/Desktop/Positive_Image/image1_nose.png')
        nosePath = "C:/Users/PateauTech_2/Desktop/Positive_Image/image1_nose.png"
        
        img.crop((0, 2*y/3, x, y)).save('C:/Users/PateauTech_2/Desktop/Positive_Image/image1_mouth.png')
        mouthPath = "C:/Users/PateauTech_2/Desktop/Positive_Image/image1_mouth.png"
        
        return [eyeRPath,eyeLPath,nosePath,mouthPath]

    # Sobel Derivatives
    
    def Sobel(self,src): # take a grey image
        window_name = ('Sobel')
        scale = 1
        delta = 0
        ddepth = cv2.CV_16S
        gray = src
        grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        cv2.imwrite('C:/Users/PateauTech_2/Desktop/Positive_Image/image.png',grad) 
        ##cv2.imshow(window_name, grad)
        #cv2.waitKey(0)
        return grad
    
    # Detecteur de Harris
    def Harris(self,gray): 
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,2,3,0.04)
        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)
        # Threshold for an optimal value, it may vary depending on the image.
        gray[dst>0.01*dst.max()]=[0,255,0]
        cv2.imshow('dst',gray)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
            
            
            
    # Detect the axes of symmetry in a image
    def find_symmetry(self,newpart): #take one white-and-black image in entry
        #initialize
        hor = len(newpart)
        ver = np.size(newpart)/hor
        
        #calcul the error of symmetry
        position = []
        error    = []
        for start in range(0,int(ver)-1):
            errorInter = 0
            startL = 1 + start
            endL   = -int(ver) + start
            for interation in range(0,int((ver-1)/2)):
                startI = startL + interation
                endI   = endL   - interation
                if endI < -int(ver)+1:
                    endI = int(ver)+endI
                if startI>int(ver)-1:
                    startI = -int(ver)+(startI-int(ver))
                errorInter += scipy.stats.wasserstein_distance(newpart[:,int(startI)],newpart[:,int(endI)])
                #errorInter += scipy.stats.wasserstein_distance(TotalHist[start][0], TotalHist[int(ver)-start-1][0],Weight_TotalHist[start],Weight_TotalHist[int(ver)-1-start])
            error.append(errorInter)
            if startL<int(ver)/2:
                positioninter = startL+int(ver)/2
            else:
                positioninter = startL-int(ver)/2
            position.append(positioninter)
        return min(error),int(position[min(enumerate(error), key=itemgetter(1))[0]])

    def ApplySym(self,positionMinError,newpart,AngleMinError):
        newpart = self.rotation(AngleMinError,newpart)
        hor = len(newpart)
        ver = np.size(newpart)/hor
        # sum the two parts of the images
        New_Image = np.zeros((int(hor),int((ver-1)/2)))
        start = positionMinError
        for i in range(0,int((ver-1)/2)):
            starti = int(start-int((ver-1)/2)+i)
            endi   = -int(ver)+(int(start+int((ver-1)/2)+i)-int(ver))
            New_Image[:,i] = newpart[:,starti]+newpart[:,int(endi)]
        return New_Image
    
    # rotation of the image
    def rotation(self,angle,newpart):
        colorImage = fromarray(newpart)
        rotated     = colorImage.rotate(angle)
        return np.asarray(rotated)
    
            
    # main script
    def main(self,crop,findsym):
        for n in self.imagePath:
            print(n)
            if crop == True:
                Face_parts = self.crop_images(n)
                gray = []
                for parts in Face_parts:
                    gray.append(self.import_images(parts))
                    newpart = self.Sobel(gray)
            else:
                newpart = self.Sobel(self.import_images(n))
            for u1 in newpart:
                for u2 in range(len(u1)):
                    if u1[u2]<100:
                        u1[u2] = 0

            if findsym == True:
                errorT = []
                positionT = []
                angleT = []
                for angle in range(0,90):
                    angleT.append(angle-45)
                    imageR = self.rotation(angle-45,newpart)
                    error,position = self.find_symmetry(imageR)
                    positionT.append(position)
                    errorT.append(error)
#                plt.figure(1)
#                plt.plot(angleT,errorT)
#                plt.figure(2)
#                plt.plot(angleT,positionT)
                newSymImage    = self.ApplySym(int(positionT[min(enumerate(errorT), key=itemgetter(1))[0]]),newpart,int(angleT[min(enumerate(errorT), key=itemgetter(1))[0]]))
                newSymImage2    = self.ApplySym(int(positionT[min(enumerate(errorT), key=itemgetter(1))[0]]),np.asarray(self.import_images(n)),int(angleT[min(enumerate(errorT), key=itemgetter(1))[0]]))

                plt.title('Reduction de l\'erreur de symetrie et recadrage',fontsize=20) 
                plt.subplot(2,3,1)
                plt.imshow(newpart)
                plt.subplot(2,3,2)
                plt.imshow(self.rotation(int(angleT[min(enumerate(errorT), key=itemgetter(1))[0]]),newpart))
                plt.subplot(2,3,3)
                plt.imshow(newSymImage)
                plt.subplot(2,3,4)
                plt.imshow(self.import_images(n))
                plt.subplot(2,3,5)
                plt.imshow(self.rotation(int(angleT[min(enumerate(errorT), key=itemgetter(1))[0]]),self.import_images(n)))
                plt.subplot(2,3,6)
                plt.imshow(newSymImage2)
                plt.savefig('C:/Users/PateauTech_2/Documents/SymetrieDetection_Recadrage.png')
            return newpart
                
            
        
#Face_parts = picture_treatment.crop_images(imagePath)
m = picture_treatment()
crop = False
sym  = True 
newpart = m.main(crop,sym)




for u1 in newpart:
    for u2 in range(len(u1)):
        if u1[u2]<100:
            u1[u2] = 0




#plt.imshow(New_Image)    
#
#
##plot the error on the image
#fig = plt.figure(figsize=(20,10))
#plt.title('Reduction de l\'erreur de symetrie',fontsize=20) 
#plt.subplot(2,1,1)
#plt.plot(position,error,'o')    
#plt.xlabel('Position Horizontale du pixel du milieu', fontsize=20)
#plt.ylabel('Erreur', fontsize=20)
#fig.tight_layout()
#plt.title('Reduction de l\'erreur de symetrie')
#plt.subplot(2,1,2)
#plt.imshow(newpart)
#plt.xlabel('Position Horizontale du pixel du milieu', fontsize=20)
#plt.ylabel('Erreur -> minimisation', fontsize=20)
#plt.savefig('C:/Users/PateauTech_2/Documents/SymetrieDetection.png')

