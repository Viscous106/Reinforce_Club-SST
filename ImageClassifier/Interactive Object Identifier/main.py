import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input,
    decode_predictions
)
from PIL import Image
import pyttsx3


# Load and cache the model
@st.cache_resource
def load_model():
    model = ResNet50(weights = "imagenet")
    return model

model = load_model()

# Preprocessing the image
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis = 0)
    return img

# Classify the image
def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top = 3)[0]
        return decoded_predictions
    except Exception as e:
        print(e)
        return None


# Text to speech
def read_predictions(preds):
    engine = pyttsx3.init()
    text = "I think this is "

    for i, (imagenet_id, label, confidence) in enumerate(preds):
        text += f"{label} with {confidence * 100:.0f} percent confidence"
        if i < len(preds)-1:
            text += ", or "
    engine.say(text)
    engine.runAndWait()


def main():
    
    st.title("🔍 Interactive Object Identifier")
    st.write("Upload an image or use your webcam, and let AI tell you what it sees.")


    # radio buttons for choosing upload or capture
    mode = st.radio("Select Mode", ["Upload Image", "Use Webcam"])

    # option for uploading image using webcam
    if mode == "Upload Image":
        uploaded = st.file_uploader("Choose an image...", type = ["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded Image", use_container_width = True)

            if st.button("🔍 Classify Uploaded Image"):
                with st.spinner("Analyzing ..."):
                    preds = classify_image(model, image)
                st.success("Here is what i found:")

                for i, (imagenet_id, label, confidence) in enumerate(preds):
                    st.write(f"**{i+1}. {label}** - {confidence*100:.2f}% confidence")
                read_predictions(preds) 

    # option for capturing image using webcam
    elif mode == "Use Webcam":
        cam_image = st.camera_input("Take a photo")
        if cam_image:
            image = Image.open(cam_image)
            st.image(image, caption="Captured Image", use_container_width = True)

            if st.button("🔍 Classify Captured Image"):
                with st.spinner("Analyzing ..."):
                    preds = classify_image(model,image)
                st.success("Here is what i found:")

                for i, (imagenet_id, label, confidence) in enumerate(preds):
                    st.write(f"**{i+1}. {label}** - {confidence*100:.2f}% confidence")
                read_predictions(preds) 



if __name__ == "__main__":
    main()