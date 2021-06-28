# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 01:04:36 2021

@author: Mark Mononutu
"""

from tkinter import *
import tkinter as tk

import cv2
import os
from PIL import Image
import numpy as np
#import time


window=tk.Tk()
window.title("Face recognition system")
window.geometry("480x720")
#window.config(background="lime")

# Define image
bg = PhotoImage(file="background.png")

#Create a label
label = Label(window, image=bg)
label.place(x=0, y=0, relwidth=1, relheight=1)

l2=tk.Label(window,text="SECURE CAM",font=("Times",30), fg="black")
l2.place(relx=0.5,rely=0.1, anchor=CENTER, width=480)

def generate_dataset():
    # Define Video Capture Object
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height
    
    # Set Current Working Directory
    work_dir = "C:/Users/USER/FinalProject/";
    
    # Define Video Capture Object
    face_detector = cv2.CascadeClassifier('{0}/haarcascade/haarcascade_frontalface_default.xml'.format(work_dir))
    
    # For each person, enter one numeric face id
    face_id = input('Enter user id and press <enter> ==>  ')
    
    messagebox.showinfo('info','Initializing face capture. Look the camera, Click OK, and wait ...')
    # Initialize individual sampling face count
    count = 0
    
    while(True):
    
        ret, img = cam.read()
        img = cv2.flip(img, 1) # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
        for (x,y,w,h) in faces:
    
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
    
            # Save the captured image into the datasets folder
            cv2.imwrite(work_dir+"/dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
    
            cv2.imshow('image', img)
    
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 200: # Take 200 face sample and stop video
             break
    
    # Do a bit of cleanup
    messagebox.showinfo('info','Face data stored.')
    cam.release()
    cv2.destroyAllWindows()

def train():
    # Set Current Working Directory
    work_dir = "C:/Users/USER/FinalProject";
    
    # Path for face image database
    path = work_dir+'/dataset'
    
    # Take all facial samples on dataset directory, returning 2 arrays
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("{0}/haarcascade/haarcascade_frontalface_default.xml".format(work_dir))
    
    # Function to get the images and label data
    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []
    
        for imagePath in imagePaths:
    
            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')
    
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
    
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
    
        return faceSamples,ids
    
    messagebox.showinfo ('info','Training faces. It will take a few seconds, click OK and Wait until next message ...')
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    
    # Save the model into trainer/trainer.yml
    recognizer.write(work_dir+'/trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi
    
    # Print the numer of faces trained and end program
    messagebox.showinfo('info', '{0} faces trained. Exiting Program'.format(len(np.unique(ids))))

def launch():
    messagebox.showinfo('info', 'Server is starting, refresh your browser after few seconds')
    os.system('python main.py')
    

def start():
    labelInfo=tk.Label(window,text="Open this url",font=("Times",10), fg="black")
    labelInfo.place(relx=0.5,rely=0.75, anchor=CENTER, width=480)
    labelInfo=tk.Label(window,text="in your browser after few seconds: http://127.0.0.1:5000/ ",font=("Times",10), fg="black")
    labelInfo.place(relx=0.5,rely=0.80, anchor=CENTER, width=480)
    buttonLaunch = tk.Button(window, text="Start server", font=("Times", 20), bg='green', fg='black', command=launch)
    buttonLaunch.place(relx=0.5, rely=0.90, anchor=CENTER)
   
    

# Menu Buttons
b1=tk.Button(window,text="Generate dataset",font=("Times",25),bg='#8A2BE2',fg='black', command=generate_dataset)
b1.place(relx=0.5, rely=0.25, anchor=CENTER)

b2=tk.Button(window,text="Training Face Data",font=("Times",25),bg='#8A2BE2',fg='black', command=train)
b2.place(relx=0.5, rely=0.45, anchor=CENTER)

b2=tk.Button(window,text="Start System",font=("Times",25),bg='green',fg='black', command=start)
b2.place(relx=0.5, rely=0.65, anchor=CENTER)



window.mainloop()