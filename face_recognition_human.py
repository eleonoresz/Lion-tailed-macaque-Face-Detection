# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:04:59 2019

@author: PateauTech_2
"""

import cv2
from PIL import Image 

#imagePath = "C:/Users/PateauTech_2/Documents/Photos_Macaques_queue_de_lion/singeaqueuedelion.jpg"
#imagePath = "C:/Users/PateauTech_2/Documents/Photos_Macaques_queue_de_lion/jour1-2019-10-04-10h57m05s968.png"
#imagePath = "C:/Users/PateauTech_2/Documents/Photos_Macaques_queue_de_lion/singeaqueuedelion.jpg"
#imagePath = "C:/Users/PateauTech_2/Documents/VideosMacaques/35.jpg"
#imagePath = "C:/Users/PateauTech_2/Documents/Photos_Macaques_queue_de_lion/jour2-2019-10-10-14h12m54s706.jpg"
#cascPath  = "C:/Users/PateauTech_2/Documents/opencv-master/data/haarcascades/haarcascade_frontalface_alt2.xml"
#cascPath  = "C:/Users/PateauTech_2/Documents/XMLFiles/MacaqueFrontalFaceModel.xml"
cascPath  = "C:/Users/PateauTech_2/Downloads/cascade.xml"

#cascPath  = "C:/Users/PateauTech_2/Documents/XMLFiles/MacaqueSingleEyeModel.xml"
#cascPath  = "C:/Users/PateauTech_2/Documents/XMLFiles/MacaqueEyePairModel.xml"

faceCascade = cv2.CascadeClassifier(cascPath)

## factor resize the image
resize_image = 30.0

for i in range(10,42):
    print(i)
    imagePath = 'C:/Users/PateauTech_2/Desktop/10_10_2019/DSC_00'+str(i)+'.jpg'
    foo = Image.open(imagePath)
    foo_size = foo.size
    foo = foo.resize((int(foo_size[0]/resize_image),int(foo_size[1]/resize_image)),Image.ANTIALIAS)
    foo.save("C:/Users/PateauTech_2/Desktop/10_10_2019/output.jpg",quality=95)
# read and convert image
    imagePath = "C:/Users/PateauTech_2/Desktop/10_10_2019/output.jpg"
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the image
    faces = faceCascade.detectMultiScale(gray)
#,
#    scaleFactor=1.1,
#    minNeighbors=5,
#    minSize=(30, 30),
#    #    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
#    )
    print("Found {0} faces!".format(len(faces)))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imwrite('C:/Users/PateauTech_2/Desktop/10_10_2019/image'+str(i)+'.png',image) 




#imagePath = "C:/Users/PateauTech_2/Desktop/Positive_Image/Face/onlycenter33.jpg"
## create landmark detector and load lbf model:
#facemark = cv2.face.createFacemarkLBF()
#facemark.loadModel("C:/Users/PateauTech_2/Downloads/lbfmodel.yaml")
#
## run landmark detector:
#ok, landmarks = facemark.fit(image, imagePath)
#
## print results:
#print ("landmarks LBF",ok, landmarks)