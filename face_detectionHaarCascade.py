# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:04:59 2019

@author: PateauTech_2
"""

import cv2


#imagePath = "C:/Users/PateauTech_2/Documents/Photos_Macaques_queue_de_lion/singeaqueuedelion.jpg"
#imagePath = "C:/Users/PateauTech_2/Documents/Photos_Macaques_queue_de_lion/jour1-2019-10-04-10h57m05s968.png"
imagePath = "C:/Users/PateauTech_2/Documents/Photos_Macaques_queue_de_lion/rhesus-macaques-960x540.jpg"
#cascPath  = "C:/Users/PateauTech_2/Documents/opencv-master/data/haarcascades/haarcascade_frontalface_alt2.xml"
cascPath  = "C:/Users/PateauTech_2/Documents/XMLFiles/MacaqueFrontalFaceModel.xml"
#cascPath  = "C:/Users/PateauTech_2/Documents/XMLFiles/MacaqueSingleEyeModel.xml"
#cascPath  = "C:/Users/PateauTech_2/Documents/XMLFiles/MacaqueEyePairModel.xml"

faceCascade = cv2.CascadeClassifier(cascPath)

# read and convert image
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

# show face detections
#for (x, y, w, h) in faces:
#    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
#
#cv2.imshow("Faces found", image)
#cv2.waitKey(0)