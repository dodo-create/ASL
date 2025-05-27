import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1) #Maximum Amount of hands to be detected

#Starts the Camera
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img) #Locates the hand, with skeleton
    #Cropping
    if hands:
        hand = hands[0]
    cv2.imshow("Image",img)
    cv2.waitKey(1)


