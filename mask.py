import cv2
import numpy as np
import os

dirPath = os.path.dirname(os.path.realpath(__file__))
faceImagePath = dirPath + "\\images\\face\\"

face_classifier = cv2.CascadeClassifier(dirPath + "\\data\\haarcascade_frontalface_default.xml")
face_mask = cv2.imread('images/mask_cropped.jpg')

h_mask, w_mask = face_mask.shape[:2]
scaling_factor = 0.5

for filename in os.listdir(faceImagePath):

    image = cv2.imread("images/face/" + filename)
    image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_rects = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in face_rects:
        try:
            if h > 0 and w > 0:
                h, w = int(0.65 * h), int(0.75 * w)
                y += 160
                x += 40


            roi = image[y:y + h, x:x + w]

            face_mask_small = cv2.resize(face_mask, (w, h), interpolation=cv2.INTER_AREA)
            gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)

            ret, mask = cv2.threshold(gray_mask, 250, 255, cv2.THRESH_BINARY_INV)
            mask_inv = cv2.bitwise_not(mask)
            masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask)

            masked_frame = cv2.bitwise_and(roi, roi, mask=mask_inv)
            image[y:y + h, x:x + w] = cv2.add(masked_face, masked_frame)

            cv2.imwrite("output/mask/" + filename, image)
            # cv2.imshow("Image", image)
        except:
            print(f'${filename} error')

cv2.waitKey(0)
