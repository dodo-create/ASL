import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np 
import math 

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1) #Maximum Amount of hands to be detected


offset = 20 #so the cropped image isnt tightbound, has some extra space
imgSize = 300 #defined image size




#Starts the Camera
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img) #Locates the hand, with skeleton
    #Cropping
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox'] #Gives us the values of the bounding box from the dictionary

        #Creating an Image by ourself so all gestures have the same boundary
        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255 #a square of 300x300, 3 is the colur information a white block
        imgCropShape = imgCrop.shape #matrix of 3 values height,width,channel

       

        aspectRatio = h/w #if value >1 height is greater, if value<1 width is greater if value=1 its a sqaure

        if aspectRatio > 1:
            k = imgSize/h  #stretching the height
            wCal = math.ceil(k*w) #rounds off to the higher integer
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            imgWhite[0:imgResizeShape[0],0:imgResizeShape[1]] = imgCrop #Opens the image on top of the white image
            




        imgCrop = img[y-offset:y+h+offset,x-offset:x+w+offset] #since it is a matrix, we have defined the ranges/dimensions of the crop 
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)





    cv2.imshow("Image",img)
    cv2.waitKey(1)


