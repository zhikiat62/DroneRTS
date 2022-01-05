# simple example demonstrating how to control a Tello using your keyboard.
# For a more fully featured example see manual-control-pygame.py
#
# Use W, A, S, D for moving, E, Q for rotating and R, F for going up and down.
# When starting the script the Tello will takeoff, pressing ESC makes it land
#  and the script exit.

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import os
from drone import Tello
import numpy as np

# import time
# import math

FACE_DETECT_MODEL = "face_detect_model"
MASK_DETECT_MODEL = "mask_detect_model"
CONFIDENCE_LEVEL = 0.5


def detect_and_predict_mask(frame, faceNet, maskNet, confidenceLevel):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
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
        if confidence > confidenceLevel:
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
        preds = maskNet.predict(faces, batch_size=35)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


# connect to drone first
myDrone = Tello()
myDrone.connect()

# open the drone streaming for warm up
myDrone.streamon()
# get the frame on
frame_read = myDrone.get_frame_read()

# starting to take off the drone
myDrone.takeoff()

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.join(os.getcwd(), FACE_DETECT_MODEL, "deploy.prototxt")
weightsPath = os.path.join(os.getcwd(), FACE_DETECT_MODEL, "res10_300x300_ssd_iter_140000.caffemodel")
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskModelPath = os.path.join(os.getcwd(), MASK_DETECT_MODEL, "mask-detector-model.model")
maskNet = load_model(maskModelPath)

while True:
    # In reality you want to display frames in a seperate thread. Otherwise
    #  they will freeze while the drone moves.
    img = frame_read.frame
    cv2.imshow("drone", img)

    key = cv2.waitKey(1) & 0xff

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(img, faceNet, maskNet, CONFIDENCE_LEVEL)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (withMask, withMaskIncorrect, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if withMask > withoutMask and withMask > withMaskIncorrect else "Incorrect Mask" if withMaskIncorrect > withMask and withMaskIncorrect > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 255, 255) if label == "Incorrect Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(withMask, withoutMask, withMaskIncorrect) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(img, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)

    # key to control the drone
    if key == 27:  # ESC
        break
    elif key == ord('w'):
        myDrone.move_forward(30)
    elif key == ord('s'):
        myDrone.move_back(30)
    elif key == ord('a'):
        myDrone.move_left(30)
    elif key == ord('d'):
        myDrone.move_right(30)
    elif key == ord('e'):
        myDrone.rotate_clockwise(30)
    elif key == ord('q'):
        myDrone.rotate_counter_clockwise(30)
    elif key == ord('r'):
        myDrone.move_up(30)
    elif key == ord('f'):
        myDrone.move_down(30)

myDrone.streamoff()
cv2.destroyAllWindows()
myDrone.land()
