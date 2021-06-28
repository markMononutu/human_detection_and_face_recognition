# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 19:26:07 2021

@author: Mark Mononutu
"""

import cv2
import numpy as np
import os 
from datetime import datetime



# Set Current Working Directory
work_dir = "C:/Users/USER/FinalProject";

# Take all facial samples on dataset directory, returning 2 arrays
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(work_dir+'/trainer/trainer.yml')
#iniciate id counter
id = 0

# Name related to ids
names = ['Unknown', 'Rosita Lala (Klm 21)', 'Mark Mononutu (Klm 21)', 'Jesse Mononutu (Klm 21)', 'James Mononutu (Klm 21)'] 


face_cascade=cv2.CascadeClassifier(work_dir+"/haarcascade/haarcascade_frontalface_default.xml")
fullbody_cascade=cv2.CascadeClassifier(work_dir+"/haarcascade/haarcascade_fullbody.xml")
upperbody_cascade=cv2.CascadeClassifier(work_dir+"/haarcascade/haarcascade_upperbody.xml")
# ds_factor=0.6
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
#address = "http://192.168.100.19:8080/video"
#cam.open(address)
count =0

cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)



class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture('humanDetect.mp4')

    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        while True:
            ret, image = self.video.read()
            image = cv2.flip(image, 1) # Video Rotation
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            face_rects=face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5, minSize = (int(minW), int(minH)),)
            upperbody=upperbody_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 3, minSize = (int(minW), int(minH)),)
            fullbody=fullbody_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 3, minSize = (int(minW), int(minH)),)
            for (x,y,w,h) in face_rects:
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
                id,confidence = recognizer.predict(gray[y:y+h,x:x+w])

                #If condfidence is less than 1== ==> "0" : perfect match
                if (confidence < 100):
                    id = names[id]
                    confidence = " {0}%".format(round(100 - confidence))
                    cv2.putText(image, str(id), (x+5,y-5),font,1,(255,255,255),2)
                   
                elif (confidence > 100) :
                    id = "unknown"
                    confidence = " {0}%".format(round(100 - confidence))
                    cv2.putText(image, str(id), (x+5,y-5),font,1,(0,0,255),2)
                    
            # Full body detection        
            for (x,y,w,h) in fullbody:
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(image, str('full_body'), (x+5,y-5),font,1,(0,0,255),2)
            # Upper body detection      
            for (x,y,w,h) in upperbody:
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(image, str('upper_body'), (x+5,y-5),font,1,(0,0,255),2)
           
                
    
            ret, jpeg = cv2.imencode(' .jpg', image)
            return jpeg.tobytes()