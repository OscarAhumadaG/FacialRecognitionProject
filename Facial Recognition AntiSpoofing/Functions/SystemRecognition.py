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

# Function to code faces
def Code_Face(images):
    # List
    CodeList = []

    for img in images:
        # Set the color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Converting images to code
        cod = fr.face_encodings(img)[0]
        # Save codes in list
        CodeList.append(cod)
    return CodeList


def CloseWindow():
    global step, count
    # Reset
    count = 0
    step = 0
    screen2.destroy()

def CloseWindow2():
    global step, count
    # Reset
    count = 0
    step = 0
    screen3.destroy()

# Show profile
def Profile():
    global step, count, UserName, OutFolderPathUser
    # Variables reset
    step = 0
    count = 0

    # Creating a new screen
    screen4 = Toplevel(screen)
    screen4.title('PROFILE')
    screen4.geometry('1280x720')

    # Background
    background2 = Label(screen4,image=BG_Image2, text="Start")
    background2.place(x=0, y=0, relwidth=1, relheight=1)

    # File
    UserFile = open(f"{OutFolderPathUser}/{UserName}.txt", 'r')
    InfoUser = UserFile.read().split(",")
    Name = InfoUser[0]
    User = InfoUser[1]
    Pass = InfoUser[2]
    UserFile.close()

    # Check User
    if User in classes:
        # Interfaz
        texto1 = Label(screen4, text=f"Bienvenido {Name}")
        texto1.place(x=580, y=50)

        # Label Img
        labelimageUser = Label(screen4)
        labelimageUser.place(x=490,y=80)

        # Imagen
        PosUserImg = classes.index(User)
        UserImg = images[PosUserImg]

        ImgUser = Image.fromarray(UserImg)

        ImgUser = cv2.imread(f"{OutFolderPathFace}/{User}.png")

        ImgUser = cv2.cvtColor(ImgUser, cv2.COLOR_RGB2BGR)
        ImgUser = Image.fromarray(ImgUser)
        ImgUser = ImgUser.resize((400, 400))

        IMG = ImageTk.PhotoImage(image=ImgUser)

        labelimageUser.configure(image=IMG)
        labelimageUser.image = IMG



# User Registration
def BiometricSign():
    global screen, screen2, count, blink, img_info, step, capt, VideoLabel, UserReg

    # Check Capt
    if capt is not None:
        ret, frame = capt.read()

        if ret:

            frameSave = frame.copy()

            # Resize
            frame = imutils.resize(frame, width=1280)

            # Create another frame
            FrameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


            # Frame Show
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Inference Face Mesh
            MeshResult = FaceMesh.process(FrameRGB)

            # Mesh Result Lists
            px = []
            py = []
            ListResult = []

            if MeshResult.multi_face_landmarks:
                # Extract Face Mesh
                for face in MeshResult.multi_face_landmarks:
                    # Draw
                    mpDraw.draw_landmarks(frame, face, FaceMeshObject.FACEMESH_CONTOURS, ConfigDraw, ConfigDraw)

                    # Extract KeyPoints
                    for id, point in enumerate(face.landmark):
                        # Image Info
                        width = frame.shape[1]
                        height = frame.shape[0]

                        x, y = int(point.x * width), int(point.y * height)
                        px.append(x)
                        py.append(y)
                        ListResult.append([id, x, y])

                        # 468 KeyPoints
                        if len(ListResult) == 468:
                            # Right Eye
                            x1, y1 = ListResult[145][1:]
                            x2, y2 = ListResult[159][1:]
                            Lenght_1 = math.hypot(x2 - x1, y2 - y1)

                            # Left Eye
                            x3, y3 = ListResult[374][1:]
                            x4, y4 = ListResult[386][1:]
                            Lenght_2 = math.hypot(x4 - x3, y4 - y3)

                            # Right Parietal
                            x5, y5 = ListResult[139][1:]
                            # Left Parietal
                            x6, y6 = ListResult[368][1:]


                            # Right Eyebrow
                            x7, y7 = ListResult[70][1:]
                            x8, y8 = ListResult[300][1:]

                            # Face Detection
                            faces = FaceDetector.process(FrameRGB)

                            if faces.detections is not None:
                                for face in faces.detections:
                                    score = face.score[0]
                                    bbox = face.location_data.relative_bounding_box

                                    # Confidence Umbral
                                    if score > ConfThreshold:
                                        # Calculating the rectangle coordinates
                                        xi, yi = bbox.xmin * width, bbox.ymin * height
                                        width2, height2 = bbox.width * width, bbox.height * height

                                        # Applying Offset proportionally
                                        offsetWidth = (offsetx / 100) * width2
                                        offsetHeight = (offsety / 100) * height2

                                        xi = int(xi - offsetWidth / 2)
                                        width2 = int(width2 + offsetWidth)
                                        xf = xi + width2

                                        # Adjusting the height
                                        extra_height = int(height2 * 0.2)  # Adding a 20%  more of height
                                        yi = int(yi - offsetHeight)
                                        height2 = int(height2 + offsetHeight + extra_height)
                                        yf = yi + height2



                                        # Error Handling
                                        if xi < 0 : xi = 0
                                        if yi < 0 : yi = 0
                                        if width2 < 0: width2 = 0
                                        if height2 < 0: height2 = 0

                                        # Steps
                                        if step == 0:

                                            # Drawing the rectangle in the frame
                                            cv2.rectangle(frame, (xi, yi), (xi + width2, yi + height2), (255, 255, 255),2)

                                            # Image Step0
                                            h_s0, w_s0, c = img_Step0.shape
                                            frame[50:50 + h_s0, 50:50 + w_s0] = img_Step0

                                            # Image Step1
                                            h_s1, w_s1, c = img_Step1.shape
                                            frame[50:50 + h_s1, 1030:1030 + w_s1] = img_Step1

                                            # Image Step2
                                            h_s2, w_s2, c = img_Step2.shape
                                            frame[270:270 + h_s2, 1030:1030 + w_s2] = img_Step2

                                            # Face centered
                                            if x7 > x5 and x8 < x6:
                                                # IMG Check
                                                h_check, w_check, c = img_check.shape
                                                frame[165:165 + h_check, 1105:1105 + w_check] = img_check

                                                # Blinking count
                                                if Lenght_1 <= 10 and Lenght_2 <= 10 and blink == False:
                                                    count += 1
                                                    blink = True

                                                elif Lenght_1 > 10 and Lenght_2 > 10 and blink == True:
                                                    blink = False

                                                cv2.putText(frame, f'Blinking: {int(count)}', (1070, 375), cv2.FONT_HERSHEY_COMPLEX,0.5, (255, 255, 255),1)

                                                if count >= 3:
                                                    h_check, w_check, c = img_check.shape
                                                    frame[385:385 + h_check, 1105:1105 + w_check] = img_check

                                                    # Open eyes
                                                    if Lenght_1 > 16 and Lenght_2 > 16:
                                                        # Cut
                                                        cut = frameSave[yi:yf, xi:xf]

                                                        # Save Face
                                                        cv2.imwrite(f"{OutFolderPathFace}/{UserReg}.png", cut)

                                                        # Step 1
                                                        step = 1
                                            else:
                                                count = 0

                                        if step == 1:
                                            # Draw
                                            cv2.rectangle(frame, (xi, yi, width, height), (0, 255, 0), 2)
                                            # IMG Check Liveness
                                            h_lvCheck, w_lvCheck, c = img_LvCheck.shape
                                            frame[50:50 + h_lvCheck, 50:50 + w_lvCheck] = img_LvCheck

                                # Close
                                close = screen2.protocol("WM_DELETE_WINDOW", CloseWindow)


                            # Circle
                            # cv2.circle(frame,(x7,y7),2, (255,0,0), cv2.FILLED)
                            # cv2.circle(frame, (x8, y8), 2, (255, 0, 0), cv2.FILLED)


            # Conv. Video
            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)

            # Show Video
            VideoLabel.configure(image=img)
            VideoLabel.image = img
            VideoLabel.after(10, BiometricSign)

    else:
        # Release VideoCapture when capt is None
        if capt is not None:
            capt.release()

# User Log In
def BiometricLog():
    global screen, screen3, LogUser, LogPass, step, OutFolderPath, capt, VideoLabel, FaceCode, classes, images, blink, count,UserReg, UserName
    # Check Capt
    if capt is not None:
        ret, frame = capt.read()

        if ret:

            frameSave = frame.copy()

            # Resize
            frame = imutils.resize(frame, width=1280)

            # Create another frame
            FrameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Frame Show
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Inference Face Mesh
            MeshResult = FaceMesh.process(FrameRGB)

            # Mesh Result Lists
            px = []
            py = []
            ListResult = []

            if MeshResult.multi_face_landmarks:
                # Extract Face Mesh
                for face in MeshResult.multi_face_landmarks:
                    # Draw
                    mpDraw.draw_landmarks(frame, face, FaceMeshObject.FACEMESH_CONTOURS, ConfigDraw, ConfigDraw)

                    # Extract KeyPoints
                    for id, point in enumerate(face.landmark):
                        # Image Info
                        width = frame.shape[1]
                        height = frame.shape[0]

                        x, y = int(point.x * width), int(point.y * height)
                        px.append(x)
                        py.append(y)
                        ListResult.append([id, x, y])

                        # 468 KeyPoints
                        if len(ListResult) == 468:
                            # Right Eye
                            x1, y1 = ListResult[145][1:]
                            x2, y2 = ListResult[159][1:]
                            Lenght_1 = math.hypot(x2 - x1, y2 - y1)

                            # Left Eye
                            x3, y3 = ListResult[374][1:]
                            x4, y4 = ListResult[386][1:]
                            Lenght_2 = math.hypot(x4 - x3, y4 - y3)

                            # Right Parietal
                            x5, y5 = ListResult[139][1:]
                            # Left Parietal
                            x6, y6 = ListResult[368][1:]

                            # Right Eyebrow
                            x7, y7 = ListResult[70][1:]
                            x8, y8 = ListResult[300][1:]

                            # Face Detection
                            faces = FaceDetector.process(FrameRGB)

                            if faces.detections is not None:
                                for face in faces.detections:
                                    score = face.score[0]
                                    bbox = face.location_data.relative_bounding_box

                                    # Confidence Umbral
                                    if score > ConfThreshold:
                                        # Calculating the rectangle coordinates
                                        xi, yi = bbox.xmin * width, bbox.ymin * height
                                        width2, height2 = bbox.width * width, bbox.height * height

                                        # Applyig Offset proportionally
                                        offsetWidth = (offsetx / 100) * width2
                                        offsetHeight = (offsety / 100) * height2

                                        xi = int(xi - offsetWidth / 2)
                                        width2 = int(width2 + offsetWidth)
                                        xf = xi + width2

                                        # Adjusting the height
                                        extra_height = int(height2 * 0.2)  # Adding a 20%  more of height
                                        yi = int(yi - offsetHeight)
                                        height2 = int(height2 + offsetHeight + extra_height)
                                        yf = yi + height2

                                        # Error Handling
                                        if xi < 0: xi = 0
                                        if yi < 0: yi = 0
                                        if width2 < 0: width2 = 0
                                        if height2 < 0: height2 = 0

                                        # Steps
                                        if step == 0:

                                            # Drawing the rectangle in the frame
                                            cv2.rectangle(frame, (xi, yi), (xi + width2, yi + height2), (255, 255, 255),
                                                          2)

                                            # Image Step0
                                            h_s0, w_s0, c = img_Step0.shape
                                            frame[50:50 + h_s0, 50:50 + w_s0] = img_Step0

                                            # Image Step1
                                            h_s1, w_s1, c = img_Step1.shape
                                            frame[50:50 + h_s1, 1030:1030 + w_s1] = img_Step1

                                            # Image Step2
                                            h_s2, w_s2, c = img_Step2.shape
                                            frame[270:270 + h_s2, 1030:1030 + w_s2] = img_Step2

                                            # Face centered
                                            if x7 > x5 and x8 < x6:
                                                # IMG Check
                                                h_check, w_check, c = img_check.shape
                                                frame[165:165 + h_check, 1105:1105 + w_check] = img_check

                                                # Blinking count
                                                if Lenght_1 <= 10 and Lenght_2 <= 10 and blink == False:
                                                    count += 1
                                                    blink = True

                                                elif Lenght_1 > 10 and Lenght_2 > 10 and blink == True:
                                                    blink = False

                                                cv2.putText(frame, f'Blinking: {int(count)}', (1070, 375),
                                                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

                                                if count >= 3:
                                                    h_check, w_check, c = img_check.shape
                                                    frame[385:385 + h_check, 1105:1105 + w_check] = img_check

                                                    # Open eyes
                                                    if Lenght_1 > 16 and Lenght_2 > 16:

                                                        # Step 1
                                                        step = 1
                                            else:
                                                count = 0

                                        if step == 1:
                                            # Draw
                                            cv2.rectangle(frame, (xi, yi, width, height), (0, 255, 0), 2)
                                            # IMG Check Liveness
                                            h_lvCheck, w_lvCheck, c = img_LvCheck.shape
                                            frame[50:50 + h_lvCheck, 50:50 + w_lvCheck] = img_LvCheck

                                            # Find faces
                                            facesloc = fr.face_locations(FrameRGB)
                                            facescod = fr.face_encodings(FrameRGB, facesloc)

                                            for facecod, faceloc in zip(facescod, facesloc):
                                                # Matching
                                                Match = fr.compare_faces(FaceCode, facecod)

                                                # Similarities
                                                Similarity = fr.face_distance(FaceCode, facecod)

                                                # Minimum value
                                                min = np.argmin(Similarity)

                                                if Match[min]:
                                                    # Extract Username
                                                    UserName = classes[min].upper()

                                                    Profile()


                                # Close
                                close = screen3.protocol("WM_DELETE_WINDOW", CloseWindow2)

                            # Circle
                            # cv2.circle(frame,(x7,y7),2, (255,0,0), cv2.FILLED)
                            # cv2.circle(frame, (x8, y8), 2, (255, 0, 0), cv2.FILLED)

            # Conv. Video
            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)

            # Show Video
            VideoLabel.configure(image=img)
            VideoLabel.image = img
            VideoLabel.after(10, BiometricLog)
        else:
            capt.release()
    else:
        # Release VideoCapture when capt is None
        if capt is not None:
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
        UserName = []

        # Track User List
        for file in UserFiles:
            # Extract User
            User = file.split('.')[0]
            UserName.append(User)

        # Check User
        if UserReg in UserName:
            # Registered
            print('YOU HAVE ALREADY REGISTERED')

        else:
            # Not Registered
            # Save Info
            info.append(list[NameReg, UserReg, PWReg])

            # Export Info
            f = open(f"{OutFolderPathUser}/{UserReg.upper()}.txt", "w")
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
    global LogUser, LogPass, OutFolderPathFace, capt, VideoLabel, screen3, FaceCode, classes, images
    # Extract: Name, User, Password
    LogUser, LogPass = InputUserLog.get(), InputPassLog.get()

    # DB Faces
    images = []
    classes = []
    lista = os.listdir(OutFolderPathFace)

    # Read face images
    for lis in lista:
        # Read image
        imgdb = cv2.imread(f"{OutFolderPathFace}/{lis}")
        # Save Img DB
        images.append(imgdb)
        # Name Img
        classes.append(os.path.splitext(lis)[0])

    # Face code
    FaceCode = Code_Face(images)

    # Creating a new screen
    screen3 = Toplevel(screen)
    screen3.title('Biometric Sign In')
    screen3.geometry('1280x720')

    # Video Label
    VideoLabel = Label(screen3)
    VideoLabel.place(x=0, y=0)

    # Video Capture
    capt = cv2.VideoCapture(0,)
    capt.set(3, 1280)
    capt.set(4, 720)
    BiometricLog()


# Path
OutFolderPathUser = '../DataBase/Users'
PathUserCheck = '../DataBase/Users'
OutFolderPathFace = '../DataBase/FaceImages'


# Read Images
#img_info = cv2.imread('/home/oscar-ahumada/Documents/Machine Learning/Facial Recognition Project/Facial Recognition AntiSpoofing/SetUp/BtSign.png')
img_check = cv2.imread('../SetUp/check.png')
img_Step0 = cv2.imread('../SetUp/Step0.png')
img_Step1 = cv2.imread('../SetUp/Step1.png')
img_Step2 = cv2.imread('../SetUp/Step2.png')
img_LvCheck = cv2.imread('../SetUp/LivenessCheck.png')

img_Step0 = cv2.cvtColor(img_Step0, cv2.COLOR_RGB2BGR)
img_Step1 = cv2.cvtColor(img_Step1, cv2.COLOR_RGB2BGR)
img_Step2 = cv2.cvtColor(img_Step2, cv2.COLOR_RGB2BGR)
img_LvCheck = cv2.cvtColor(img_LvCheck, cv2.COLOR_RGB2BGR)


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
BG_Imagen = PhotoImage(file='../SetUp/Inicio.png')
background = Label(image = BG_Imagen, text = "Start")
background.place(x = 0, y = 0, relwidth = 1, relheight = 1)

# Profile
BG_Image2 = PhotoImage(file='../SetUp/Back2.png')

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
InputUserLog.place(x=890, y=500)

# 3. Password
InputPassLog = Entry(screen)
InputPassLog.place(x=890, y=540)

# Buttons
# 1. Sign up

imageBS = PhotoImage(file='../SetUp/BtSign.png')
BtSign = Button(screen, text='Sign up',image=imageBS, height='40', width='200', command=SignUp)
BtSign.place(x=110, y=580)


# 2. Sign In
imageBL = PhotoImage(file='../SetUp/BtLogin.png')
BtLogin = Button(screen, text='Login In',image=imageBL, height='40', width='200', command=SignIn)
BtLogin.place(x=860, y=580)


screen.mainloop()


