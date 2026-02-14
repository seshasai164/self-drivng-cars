import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

st.set_page_config(page_title="Self Driving Car Object Detection")

st.title("🚗 Object Detection for Self Driving Cars")

st.write("Upload an image to detect objects like Car, Person, Truck, Traffic Light")

# Load YOLOv8 model
@st.cache_resource
def load_model():
    model = YOLO("yolov8m.pt")
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    img = np.array(image)

    if st.button("Detect Objects"):

        results = model.predict(source=img, conf=0.2, iou=0.5)

        result_img = results[0].plot()

        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        st.image(result_img, caption="Detected Image")
