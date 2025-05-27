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
        x, y, w, h = hand['bbox'] #Gives us the values of the bounding box from the dictionary

        # Crop the image first (fixed missing definition)
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        #Creating an Image by ourself so all gestures have the same boundary
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255 #a square of 300x300, 3 is the colur information a white block

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
