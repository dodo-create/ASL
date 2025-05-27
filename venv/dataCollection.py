import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1) #Maximum Amount of hands to be detected


offset = 20 #so the cropped image isnt tightbound, has some extra space
#Starts the Camera
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img) #Locates the hand, with skeleton
    #Cropping
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox'] #Gives us the values of the bounding box from the dictionary
        imgCrop = img[y-offset:y+h+offset,x-offset:x+w+offset] #since it is a matrix, we have defined the ranges/dimensions of the crop 
        cv2.imshow("ImageCrop", imgCrop)
    cv2.imshow("Image",img)
    cv2.waitKey(1)


