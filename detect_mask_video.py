# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import matplotlib.pyplot as plt
import numpy as np
import imutils
import time
import cv2
import os


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
while True:
    try:
        modelType = int(input(
            "What model would you like to use? "
            "\nEnter 1 for default "
            "\nEnter 2 for cascade "
            "\nEnter 3 for dlib with a blue mask "
            "\nEnter 4 for dlib with a black mask "
            "\n>> "))
        break
    except:
        print("That's not a valid option!\n")

modelTypeStr = ""

if modelType == 1:
    modelTypeStr = "default"
elif modelType == 2:
    modelTypeStr = "cascade"
elif modelType == 3:
    modelTypeStr = "dlibBlue"
elif modelType == 4:
    modelTypeStr = "dlibBlack"
else:
    print("\n[INFO] invalid selection: setting to default model\n")
    modelTypeStr = "default"

modelName = "mask_detector_" + modelTypeStr

modelNameFile = modelName + "_model"
modelName += ".model"

maskNet = load_model("models/" + modelName)
print("\n[INFO] loaded model: " + modelName + "\n")

while True:
    try:
        videoNum = int(input(
            "Which video would you like to analyze? "
            "\nEnter 1 for test video "
            "\nEnter 2 for Ontario government video"
            "\n>> "))
        break
    except:
        print("That's not a valid option!\n")

videoName = ""

print(videoNum)

if videoNum == 1:
    videoName = "test_video"
elif videoNum == 2:
    videoName = "ongov"
else:
    print("\n[INFO] invalid selection: setting to test video\n")
    videoName = "test_video"

cap = cv2.VideoCapture('dataset/videos/' + videoName + '.mp4')

if (cap.isOpened() == False):
    print("Error opening video stream or file")
    quit()
else:
    print("\n[INFO] loaded video: " + videoName + '.mp4' + "\n")

fps = round(cap.get(cv2.CAP_PROP_FPS))
print("[INFO] video fps: " + str(fps) + "\n")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter("output/videos/" + modelNameFile + "_" + videoName + ".avi",
                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

saveFrames = input(
    "Would you to save all individual video frames? "
    "\nEnter y for yes "
    "\nEnter n for no "
    "\n>> ")

if saveFrames[0].lower() == "y":
    saveFrames = True
    print("\n[INFO] saving video frames to: " + "output/frames/" + modelTypeStr + "/\n")
else:
    print("\n[INFO] not saving video frames\n")
    saveFrames = False

mask_on = 0
mask_off = 0
frameNum = 0
print("[INFO] Processing video...\n")

# loop over the frames from the video
while True:
    ret, frame = cap.read()

    if ret:
        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            if label == "Mask":
                mask_on += 1
            if label == "No Mask":
                mask_off += 1
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            cv2.putText(frame, modelName, (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # write frames to output video
        out.write(frame)

        # save some frames
        # if (frameNum % round(fps / 2)) == 0:
        #    cv2.imwrite("output/frames/" + modelTypeStr + "/%d.png" % frameNum, frame)

        if saveFrames:
            cv2.imwrite("output/frames/" + modelTypeStr + "/%d.png" % frameNum, frame)
            frameNum += 1

        # if the `q` key was pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

print("[INFO] video saved to " + "output/videos/" + modelNameFile + "_" + videoName + ".avi" + "\n")

# Plot a pie chart comparing the time spent with mask on vs mask off
print("[INFO] Generating report...")
maskOn_Time = round(mask_on / fps, 2)
maskOff_Time = round(mask_off / fps, 2)
print("Mask on: {0}".format(maskOn_Time))
print("Mask off: {0}".format(maskOff_Time))
y = np.array([maskOn_Time, maskOff_Time])
pieLabel = ["Time with mask on = {0}s".format(maskOn_Time), "Time with Mask off = {0}s".format(maskOff_Time)]
plt.pie(y, labels=pieLabel)
plt.savefig("output/graphs/{0}_pieChart.jpg".format(modelNameFile))

print("\n[INFO] report saved to " + "output/graphs/{0}_pieChart.jpg".format(modelNameFile))

# cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
