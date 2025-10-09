import cv2
import numpy as np
from PIL import Image
import streamlit as st
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M, preprocess_input, decode_predictions
from meta_ai_api import MetaAI

def load_model():
    return EfficientNetV2M(weights="imagenet")

def preprocess_image(image):
    img = fill_transparent_with_white(image)
    img = np.array(img)
    img = cv2.resize(img, (480, 480))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=5)[0]

        return decoded_predictions
    except Exception as e:
        st.error(f"Error Classifying Image: {str(e)}")

def describe_image(ai, predictions):
    instructions = "The following 3 terms you will receive best describe an image. Your job is to write a very specific description (around 1-2 lines) of what you think the image is. Feel free to make the description as absurd as you like. Note that your response must contain nothing other than the description, no personal remarks. Here are the 3 terms:"
    response = ai.prompt(message=f"{instructions} {predictions[0][1].replace('_', ' ')}, {predictions[1][1].replace('_', ' ')}, {predictions[2][1].replace('_', ' ')}")
    return response["message"]

def fill_transparent_with_white(image):
    img_array = np.array(image)
    if img_array.shape[-1] == 4:
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)
        b, g, r, a = cv2.split(img_cv)
        img_bgr = cv2.merge([b, g, r])
        white = np.ones_like(img_bgr, dtype=img_bgr.dtype) * 255
        alpha_mask = a / 255.0
        blended = (img_bgr * alpha_mask[:, :, np.newaxis] + white * (1 - alpha_mask)[:, :, np.newaxis])
        return blended.astype(np.uint8)
    else:
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

def main():
    st.set_page_config(page_title="AI Image Describer")

    st.title("AI Image Describer")
    st.write("Upload an Image and let our stupid but creative AI come up with a Description...")

    @st.cache_resource
    def load_cached_model():
        return load_model()
    
    model = load_cached_model()
    meta_ai = MetaAI()

    uploaded_file = st.file_uploader("Choose an image: ", type=["png", "jpg", "jpeg", "webp"])

    if uploaded_file is not None:
        image = st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

        btn = st.button("Describe")

        if btn:
            with st.spinner("Generating Description..."):
                image = Image.open(uploaded_file)
                predictions = classify_image(model, image)

                if predictions:
                    st.write(describe_image(meta_ai, predictions))

if __name__ == "__main__":
    main()
