import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np 
import math 
import time



cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1) #Maximum Amount of hands to be detected

offset = 20 #so the cropped image isnt tightbound, has some extra space
imgSize = 300 #defined image size


folder = "Data/B"
counter = 0




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
        paddingRatio = 0.9  # Scale down final image to leave margin (10% margin inside the white square)


        if aspectRatio > 1:
            k = (imgSize * paddingRatio) / h  # scale down to leave margin
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, int(imgSize * paddingRatio)))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            hGap = math.ceil((imgSize - imgResizeShape[0]) / 2)
            imgWhite[hGap:hGap + imgResizeShape[0], wGap:wGap + wCal] = imgResize

        else:
            k = (imgSize * paddingRatio) / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (int(imgSize * paddingRatio), hCal))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - imgResizeShape[1]) / 2)
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, wGap:wGap + imgResizeShape[1]] = imgResize


        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)

