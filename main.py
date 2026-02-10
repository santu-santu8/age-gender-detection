import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Age & Gender Detector", layout="wide")
st.title("📸 Age & Gender Detector (Mobile Compatible)")

# --- Load models ---
@st.cache_resource  # cache models to speed up
def load_models():
    faceNet = cv2.dnn.readNet("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
    ageNet = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
    genderNet = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")
    return faceNet, ageNet, genderNet

faceNet, ageNet, genderNet = load_models()

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-17)', '(18-25)','(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
padding = 20

# --- Face detection function ---
def faceBox(faceNet, frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227,227), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    bboxs = []
    for i in range(detections.shape[2]):
        conf = detections[0,0,i,2]
        if conf > 0.7:
            x1 = int(detections[0,0,i,3]*w)
            y1 = int(detections[0,0,i,4]*h)
            x2 = int(detections[0,0,i,5]*w)
            y2 = int(detections[0,0,i,6]*h)
            bboxs.append([x1, y1, x2, y2])
    return bboxs

# --- Streamlit camera input ---
camera_input = st.camera_input("📷 Take a photo")

if camera_input:
    # Convert BytesIO to OpenCV
    img = Image.open(camera_input)
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    bboxs = faceBox(faceNet, frame)

    if len(bboxs) == 0:
        st.warning("No face detected. Try again!")
    else:
        for bbox in bboxs:
            # Crop face with padding
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),
                         max(0,bbox[0]-padding):min(bbox[2]+padding,frame.shape[1]-1)]
            
            blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            
            # Gender prediction
            genderNet.setInput(blob)
            genderPred = genderNet.forward()
            gender = genderList[genderPred[0].argmax()]

            # Age prediction
            ageNet.setInput(blob)
            agePred = ageNet.forward()
            age = ageList[agePred[0].argmax()]

            label = f"{gender}, {age}"

            # Draw rectangle and label
            cv2.rectangle(frame, (bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0), -1)
            cv2.putText(frame, label, (bbox[0], bbox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # Display the final image
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
