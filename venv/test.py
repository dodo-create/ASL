import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow as tf  # TensorFlow 2.19 compatible import


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)  # Maximum Amount of hands to be detected
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")  # Load pre-trained model

offset = 20  # so the cropped image isn't tightbound, has some extra space
imgSize = 300  # defined image size

folder = "Data/C"
counter = 0

labels = ["A", "B", "C"]  # Your custom labels


# Starts the Camera
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)  # Locates the hand, with skeleton

    # Cropping
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']  # Gets the bounding box from the hand dictionary

        # Crop the image
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Creating a white image to place the cropped hand with padding
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # 300x300 white image

        aspectRatio = h / w
        paddingRatio = 0.9  # Scale down final image to leave margin (10% margin inside the white square)

        if aspectRatio > 1:
            k = (imgSize * paddingRatio) / h  # scale down height
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, int(imgSize * paddingRatio)))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            hGap = math.ceil((imgSize - imgResizeShape[0]) / 2)
            imgWhite[hGap:hGap + imgResizeShape[0], wGap:wGap + wCal] = imgResize

        else:
            k = (imgSize * paddingRatio) / w  # scale down width
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (int(imgSize * paddingRatio), hCal))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - imgResizeShape[1]) / 2)
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, wGap:wGap + imgResizeShape[1]] = imgResize

        # Make prediction on the white-padded hand image
        prediction, index = classifier.getPrediction(imgWhite)
        print(prediction, index)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
