import cv2
import numpy as np
import streamlit as st
from keras.applications.xception import (
    Xception,
    preprocess_input,
    decode_predictions
)


@st.cache_resource
def load_model():
    model = Xception(weights="imagenet")
    return model

def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (299, 299))

    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(model, image):
    try:
        processed_img = preprocess_image(image)
        predictions = model.predict(processed_img)
        decoded_predictions = decode_predictions(predictions, top=10)[0]
        return decoded_predictions
    
    except Exception as e:
        st.error(f"Error Classifying Image: {e}")
        return None