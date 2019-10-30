# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:30:24 2019

@author: PateauTech_2
"""

# import libraries
from scipy import ndimage
import cv2
from PIL import Image 
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import glob


# Class construction

class AutomaticDetectionFeatures(object):
    
    # variable
    
    def __init__(self,PathI,PathC,PathF,PathE,PathN,nbPointNose,resizeN):
        
        # path to images
        self.imagesPath    = PathI
        # path and activation of cascade already trained
        self.cascCriniere  = cv2.CascadeClassifier(PathC)
        self.cascFace      = cv2.CascadeClassifier(PathF)
        self.cascNose      = cv2.CascadeClassifier(PathN)
        self.cascEye       = cv2.CascadeClassifier(PathE)
        # number of detected points in the nose for Kmeans
        self.NoseDetect    = nbPointNose
        # simplification if any
        self.resize_image  = resizeN
        # simplification level after sobel extraction
        self.sensitivity   = 20
        # pixel around the center points
        self.diametre      = 50
        
    def gaussian(self,x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
    def cutImage(self,gray,cut):
        Yend = cut[3]
        Ystart = cut[2]
        Xend = cut[5]
        Xstart = cut[4]
        xS = int((np.size(gray)/len(gray))/cut[0])
        yS = int(len(gray)/cut[1])
        gray = gray[int(yS*Ystart):int(yS*Yend),int(xS*Xstart):int(xS*Xend)]
        return gray,int(yS*Ystart),int(xS*Xstart)

    
    def useCascade(self,cascade,image,Precise,*cut):    
        # detect objects in the image 
        if Precise[0] == False:
            objects = cascade.detectMultiScale(image)
        else:
            x = Precise[1]
            y = Precise[2]
            w = Precise[3]
            h = Precise[4]
            firstImg  = image[int(y):int(y)+int(h),int(x):int(x)+int(w)]
            finaleImg,xS,yS = self.cutImage(firstImg,cut[0])
            objects = cascade.detectMultiScale(finaleImg)
            for values in objects:
                values[0]+=(x+yS)
                values[1]+=(y+xS)
        return objects
    
    def simplifyImage(self,image):
        foo = Image.fromarray(image)
        foo_size = foo.size
        foo = foo.resize((int(foo_size[0]/self.resize_image),int(foo_size[1]/self.resize_image)),Image.ANTIALIAS)
        return np.array(foo)
    
    def somethingInside(self,vectorInside,vectorOutside):
        if vectorOutside[0]<vectorInside[0] and vectorOutside[1]<vectorInside[1] and vectorOutside[2]+vectorOutside[0]>vectorInside[2]+vectorInside[0] and vectorOutside[3]+vectorOutside[1]>vectorInside[3]+vectorInside[1]:
            return True
        else:
            return False
    
    def detectFaces(self,image):   
        CoorFaces    = self.useCascade(self.cascFace,self.simplifyImage(image),[False])
        CoorCriniere = self.useCascade(self.cascCriniere,self.simplifyImage(image),[False])
        # condition: the face must be inside the criniere!
        FaceCoordonnes = []
        for vectorF in CoorFaces:
            for vectorC in CoorCriniere:
                if self.somethingInside(vectorF,vectorC):
                    FaceCoordonnes.append([True,vectorF[0]*self.resize_image,vectorF[1]*self.resize_image,vectorF[2]*self.resize_image,vectorF[3]*self.resize_image])
                    
        return FaceCoordonnes
    
    def Sobel(self,src,coor): # take a grey image
        scale = 1
        delta = 0
        ddepth = cv2.CV_16S
        gray = src[int(coor[1]):int(coor[1])+int(coor[3]), int(coor[0]):int(coor[0])+int(coor[2])]
        grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        return grad # return the Sobel image (contours)
    
    def higherSensitivity(self,image):
        ind = 0
        for u1 in image:
            ind+=1
            for u2 in range(len(u1)):
                if u1[u2]<self.sensitivity:
                    u1[u2] = 0
                else:
                    u1[u2] = u1[u2]*(self.gaussian(ind,len(image)/2,len(image)/3))
        return image
    
    def ImageToCoor(self,img):
        X = []
        Y = []
        for i in range(0,len(img)):
            for j in range(0,int(np.size(img)/len(img))):
                if img[i,j]!=0:
                    for value in range(int(img[i,j])-1):
                        X.append(j)
                        Y.append(i)
        return X,Y
        
    def kmeans(self,image,Coor,imageTrue):
        img = self.higherSensitivity(image)
        X,Y = self.ImageToCoor(img)
        Data = {'x': X,'y': Y}
        df = DataFrame(Data,columns=['x','y'])
        kmeans = KMeans(n_clusters=self.NoseDetect).fit(df)
        centroids = kmeans.cluster_centers_
        position = np.zeros((2,2))
        indP = 0
        for points in centroids:
            position[indP,0] = points[0]+Coor[0]
            position[indP,1] = points[1]+Coor[1]
            indP+=1
#        plt.figure()
#        plt.imshow(imageTrue)
#        plt.scatter(position[:, 0], position[:, 1], c='red', s=50)
        return position
    
    def centerMass(self,image,Coor):
        X = Coor[0]+int(ndimage.measurements.center_of_mass(image)[0])
        Y = Coor[1]+int(ndimage.measurements.center_of_mass(image)[1])
        return [X,Y]
    
    def makeRound(self,x,y,level,image):
        for i in range(int(self.diametre)):
            for j in range(int(self.diametre)):
                image[y-i+int(self.diametre/2),x-j+int(self.diametre/2)] = level
        return image

    
    def TreatmentImage(self,image):
        #give the coordinate of face in images
        FacePosition = self.detectFaces(image)
        for FaceCoordonnes in FacePosition:
            #Nose
            cutNose = [8,8,3,6,1,7]
            NoseCoordonnes = self.useCascade(self.cascNose,image,FaceCoordonnes,cutNose)
            CentroidsNose = []
            #Eye
            cutEye = [8,8,1,4,1.5,6.5]
            EyeCoordonnes  = self.useCascade(self.cascEye,image,FaceCoordonnes,cutEye)
            CenterMassEye = []
            # Return Position of the Centroid of the nose
            for CoorN in NoseCoordonnes:
                CentroidsNose.append(self.kmeans(self.Sobel(image,CoorN),CoorN,image))
            # Return Position of the eye via mass center
            for CoorE in EyeCoordonnes:
                CenterMassEye.append(self.centerMass(self.Sobel(image,CoorE),CoorE))
        if FacePosition == []:
            return False,False
        else:
            return CentroidsNose,CenterMassEye
        
    def PlotPointsImage(self,image,CentroidsNose,CenterMassEye,indF):
        plt.figure()
        plt.imshow(image)  
        # modify the image to plot the result
        for coor in CenterMassEye:
            plt.scatter(coor[0], coor[1], c='red', s=50)
            #image = self.makeRound(int(coor[0]),int(coor[1]), 255,image)
            
        for nose in CentroidsNose:
            for coor in nose:
                #image = self.makeRound(int(coor[0]),int(coor[1]), 255,image)
                plt.scatter(coor[0], coor[1], c='red', s=50)
        plt.savefig('C:/Users/PateauTech_2/Documents/Test/imageextraite'+str(indF)+'.png')
    
    def main(self):
        indF = 1
        PositionIMG = []
        for imagePath in self.imagesPath:  
             # extraction of images
            image = cv2.imread(imagePath,0)
            # treatment of the image
            CentroidsNose,CenterMassEye = self.TreatmentImage(image)
            if type(CentroidsNose)!=bool:
                #self.PlotPointsImage(image,CentroidsNose,CenterMassEye,indF)
                indF+=1
                listePointsEye = []
                for i in CenterMassEye:
                    listePointsEye.append(i)
                listePointsNose = []
                for i in CentroidsNose:
                    for j in i:
                        listePointsNose.append(j)
                PositionIMG.append([imagePath,listePointsEye,listePointsNose])
        return PositionIMG
    
    
    
    
    
#detect distance between regions of interests        
class AutomaticLinkFeatures(object):
    
    def __init__(self,PositionFeatures):
        # path to images
        self.Path   = PositionFeatures[0]
        # points of interests
        self.Points = [] 
        for typeP in PositionFeatures[1:]:
            for coor in typeP:
                self.Points.append(coor)
            
    def distanceBeweenPoints(self,Points1,Points2):
        distance = 0
        for i in range(len(Points1)):
            distance+= (Points1[i]-Points2[i])**2
        return np.sqrt(distance)
    
    def findDistance(self,ListePoints):
        TableDistance = np.zeros((len(ListePoints),len(ListePoints)))
        ind1 = 0
        for points1 in ListePoints:
            ind2 = 0
            for points2 in ListePoints:
                TableDistance[ind1,ind2] = self.distanceBeweenPoints(points1,points2)
                ind2+=1
            ind1+=1
        return TableDistance
    
    def main(self):
        # extraction of images
        image         = cv2.imread(self.Path,0)
        # treatment points
        TableDistance = self.findDistance(self.Points)
        # Plot
        plt.figure()
        plt.imshow(image)
        for i in range(0, len(self.Points)):
            for j in range(0,len(self.Points)):
                plt.plot([self.Points[i][0],self.Points[j][0]], [self.Points[i][1],self.Points[j][1]], 'ro-')
        plt.show()
        return TableDistance
        
            

# Path to the cascades
cascPathCriniere  = "C:/Users/PateauTech_2/Documents/XMLFiles/cascade_criniere.xml"#C:/Users/PateauTech_2/Downloads/cascadecriniere.xml"
cascPathFace  = "C:/Users/PateauTech_2/Documents/XMLFiles/cascade_face_20.xml"
cascPathnose  = "C:/Users/PateauTech_2/Documents/XMLFiles/cascade_nose.xml"#"C:/Users/PateauTech_2/Downloads/cascadenose.xml"
cascPatheye  = "C:/Users/PateauTech_2/Documents/XMLFiles/cascade_eye.xml"#C:/Users/PateauTech_2/Downloads/cascadeeyeFalse.xml"

# Path to the pictures to analyze
Path = glob.glob('C:/Users/PateauTech_2/Documents/Photos_Macaques_queue_de_lion/Pictures from movie/*.JPG')

# Parameters
nbPointsNose = 2
reSizeImage  = 20

# Activate the algorithm to get picture
silenusDetection = AutomaticDetectionFeatures(Path,cascPathCriniere,cascPathFace,cascPatheye,cascPathnose,nbPointsNose,reSizeImage)    
PositionIMG      = silenusDetection.main()

for position in PositionIMG:
    silenusLink      = AutomaticLinkFeatures(position)
    Distance         = silenusLink.main()



            
                
            
            
        
