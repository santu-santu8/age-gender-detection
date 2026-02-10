import streamlit as st
import cv2
import numpy as np
import os
import urllib.request

st.set_page_config(page_title="Age & Gender Detection", layout="centered")
st.title("📸 Age & Gender Detection")

# ----------------- MODEL DOWNLOAD -----------------
MODEL_URLS = {
    "age_net.caffemodel": "https://huggingface.co/username/age_net.caffemodel/resolve/main/age_net.caffemodel",
    "age_deploy.prototxt": "https://huggingface.co/username/age_deploy.prototxt/resolve/main/age_deploy.prototxt",
    "gender_net.caffemodel": "https://huggingface.co/username/gender_net.caffemodel/resolve/main/gender_net.caffemodel",
    "gender_deploy.prototxt": "https://huggingface.co/username/gender_deploy.prototxt/resolve/main/gender_deploy.prototxt",
    "opencv_face_detector.pbtxt": "https://huggingface.co/username/opencv_face_detector.pbtxt/resolve/main/opencv_face_detector.pbtxt",
    "opencv_face_detector_uint8.pb": "https://huggingface.co/username/opencv_face_detector_uint8.pb/resolve/main/opencv_face_detector_uint8.pb"
}

for file, url in MODEL_URLS.items():
    if not os.path.exists(file):
        with st.spinner(f"Downloading {file} ..."):
            urllib.request.urlretrieve(url, file)

# ----------------- LOAD MODELS -----------------
faceNet = cv2.dnn.readNet("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
ageNet = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
genderNet = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-17)', '(18-25)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
padding = 20

# ----------------- STREAMLIT CAMERA -----------------
camera_input = st.camera_input("Take a photo")

if camera_input is not None:
    # Convert to OpenCV image
    file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    # ----------------- FACE DETECTION -----------------
    def faceBox(faceNet, frame):
        frameHeight, frameWidth = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (227,227), [104,117,123], swapRB=False)
        faceNet.setInput(blob)
        detection = faceNet.forward()
        bboxs = []
        for i in range(detection.shape[2]):
            confidence = detection[0,0,i,2]
            if confidence > 0.7:
                x1 = int(detection[0,0,i,3]*frameWidth)
                y1 = int(detection[0,0,i,4]*frameHeight)
                x2 = int(detection[0,0,i,5]*frameWidth)
                y2 = int(detection[0,0,i,6]*frameHeight)
                bboxs.append([x1,y1,x2,y2])
        return bboxs

    bboxs = faceBox(faceNet, frame)

    # ----------------- AGE & GENDER PREDICTION -----------------
    for bbox in bboxs:
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),
                     max(0,bbox[0]-padding):min(bbox[2]+padding,frame.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

        # Gender
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]

        # Age
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]

        # Draw box & label
        cv2.rectangle(frame, (bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0), -1)
        cv2.putText(frame, f"{gender}, {age}", (bbox[0], bbox[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Detected Faces")
