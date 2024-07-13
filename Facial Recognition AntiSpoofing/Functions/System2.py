# Dependencies
import cv2
import face_recognition as fr
import numpy as np
import mediapipe as mp
import os
from tkinter import *
from PIL import Image, ImageTk
import imutils
import math

def BiometricSign():
    global screen2, count, blink, img_info, step, capt, VideoLabel, UserReg
    # Check Capt
    if capt is not None:
        ret, frame = capt.read()

        # Resize
        frame = imutils.resize(frame, width=1280)

        # Conv. Video
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)

        # Show Video
        VideoLabel.configure(image=img)
        VideoLabel.image = img
        VideoLabel.after(10, BiometricSign)

    else:
        capt.release()


def SignUp():
    global NameReg, UserReg, PWReg, InputNameReg, InputUserReg, InputPassReg, capt, VideoLabel, screen2

    NameReg, UserReg, PWReg = InputNameReg.get(), InputUserReg.get(), InputPassReg.get()
    # Incomplete Form
    if len(NameReg) == 0 or len(UserReg) == 0 or len(PWReg) == 0:
        # Print Error
        print('PLEASE FILL OUT ALL THE INFORMATION REQUIRED TO CREATE YOUR ACCOUNT ')

    # Complete Form
    else:
        # Extract: Name / User / Password
        # Check Users
        UserFiles = os.listdir(PathUserCheck)

        # Usernames
        UserNames = []

        # Track User List
        for file in UserFiles:
            # Extract User
            User = file.split('.')[0]
            UserNames.append(User)

        # Check User
        if UserReg in UserNames:
            # Registered
            print('YOU HAVE ALREADY REGISTERED')

        else:
            # Not Registered
            # Save Info
            info.append(list[NameReg, UserReg, PWReg])

            # Export Info
            f = open(f"{OutFolderPathUser}/{UserReg}.txt","w")
            f.write(NameReg + ',' +UserReg + ',' +PWReg + '\n')
            f.close()

            # Clean
            InputNameReg.delete(0, END)
            InputUserReg.delete(0, END)
            InputPassReg.delete(0, END)


            # Create a New screen
            screen2 = Toplevel(screen)
            screen2.title('Biometric Sign Up')
            screen2.geometry('1280x720')

            # Video Label
            VideoLabel = Label(screen2)
            VideoLabel.place(x=0, y=0)

            # Video Capture
            capt = cv2.VideoCapture(0)
            capt.set(3,1280)
            capt.set(4,720)
            BiometricSign()

def SignIn():
    print('hols')





# Path
OutFolderPathUser = '/home/oscar-ahumada/Documents/Machine Learning/Facial Recognition Project/Facial Recognition AntiSpoofing/DataBase/Users'
PathUserCheck = '/home/oscar-ahumada/Documents/Machine Learning/Facial Recognition Project/Facial Recognition AntiSpoofing/DataBase/Users'
OutFolderPathFace = '/home/oscar-ahumada/Documents/Machine Learning/Facial Recognition Project/Facial Recognition AntiSpoofing/DataBase/FaceImages'


# Read Images
#img_info = cv2.imread('/home/oscar-ahumada/Documents/Machine Learning/Facial Recognition Project/Facial Recognition AntiSpoofing/SetUp/BtSign.png')
img_check = cv2.imread('/home/oscar-ahumada/Documents/Machine Learning/Facial Recognition Project/Facial Recognition AntiSpoofing/SetUp/check.png')
img_Step0 = cv2.imread('/home/oscar-ahumada/Documents/Machine Learning/Facial Recognition Project/Facial Recognition AntiSpoofing/SetUp/Step0.png')
img_Step1 = cv2.imread('/home/oscar-ahumada/Documents/Machine Learning/Facial Recognition Project/Facial Recognition AntiSpoofing/SetUp/Step1.png')
img_Step2 = cv2.imread('/home/oscar-ahumada/Documents/Machine Learning/Facial Recognition Project/Facial Recognition AntiSpoofing/SetUp/Step2.png')
img_LvCheck = cv2.imread('/home/oscar-ahumada/Documents/Machine Learning/Facial Recognition Project/Facial Recognition AntiSpoofing/SetUp/LivenessCheck.png')


# Variables
blink = False
count = 0
step = 0
sample = 0

# Offset
offsety= 30
offsetx = 20

# Threshold
ConfThreshold = 0.5

# Tool Draw
mpDraw = mp.solutions.drawing_utils
ConfigDraw = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

# Object Face Mesh
FaceMeshObject = mp.solutions.face_mesh
FaceMesh = FaceMeshObject.FaceMesh(max_num_faces=1)

# Face Detector Object
Detector = mp.solutions.face_detection
FaceDetector = Detector.FaceDetection(min_detection_confidence=0.5, model_selection=1)



# Info List
info = []

# Main Screen
screen = Tk()
screen.title('Facial Recognition System')
screen.geometry('1280x720')

# Background
BG_Imagen = PhotoImage(file='/home/oscar-ahumada/Documents/Machine Learning/Facial Recognition Project/Facial Recognition AntiSpoofing/SetUp/Inicio.png')
background = Label(image = BG_Imagen, text = "Start")
background.place(x = 0, y = 0, relwidth = 1, relheight = 1)


# Input User Information for Sign Up

# 1. Name
InputNameReg = Entry(screen)
InputNameReg.place(x=110, y=320)


# 2. Username
InputUserReg = Entry(screen)
InputUserReg.place(x=110, y=430)

# 3. Password
InputPassReg = Entry(screen)
InputPassReg.place(x=110, y=540)

# Input User Information for Logging

# 1. Username
InputUserLog = Entry(screen)
InputUserLog.place(x=650, y=380)

# 3. Password
InputPassLog = Entry(screen)
InputPassLog.place(x=650, y=500)

# Buttons
# 1. Sign up

imageBS = PhotoImage(file='/home/oscar-ahumada/Documents/Machine Learning/Facial Recognition Project/Facial Recognition AntiSpoofing/SetUp/BtSign.png')
BtSign = Button(screen, text='Sign up',image=imageBS, height='40', width='200', command=SignUp)
BtSign.place(x=300, y=580)


# 2. Sign In
imageBL = PhotoImage(file='/home/oscar-ahumada/Documents/Machine Learning/Facial Recognition Project/Facial Recognition AntiSpoofing/SetUp/BtLogin.png')
BtLogin = Button(screen, text='Login In',image=imageBL, height='40', width='200', command=SignIn)
BtLogin.place(x=900, y=580)


screen.mainloop()


