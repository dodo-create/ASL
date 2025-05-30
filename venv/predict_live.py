import cv2
import numpy as np
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import math

model = load_model("gesture_model.h5")
class_names = ['A', 'B', 'C']

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        aspectRatio = h / w
        paddingRatio = 0.9

        if aspectRatio > 1:
            k = (imgSize * paddingRatio) / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, int(imgSize * paddingRatio)))
            wGap = math.ceil((imgSize - wCal) / 2)
            hGap = math.ceil((imgSize - imgResize.shape[0]) / 2)
            imgWhite[hGap:hGap + imgResize.shape[0], wGap:wGap + wCal] = imgResize

        else:
            k = (imgSize * paddingRatio) / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (int(imgSize * paddingRatio), hCal))
            wGap = math.ceil((imgSize - imgResize.shape[1]) / 2)
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, wGap:wGap + imgResize.shape[1]] = imgResize

        # Normalize and predict
        img_input = np.expand_dims(imgWhite / 255.0, axis=0)
        prediction = model.predict(img_input)
        gesture = class_names[np.argmax(prediction)]

        # Show prediction
        cv2.putText(img, gesture, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 2)

    cv2.imshow("Gesture Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
