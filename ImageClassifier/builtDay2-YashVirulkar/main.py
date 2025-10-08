
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image
from model import get_model_and_processor, describe_image, get_detailed_model_and_processor, describe_image_in_detail

def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        
        return decoded_predictions
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="AI Image Tools")
    
    st.title("AI Image Tools")
    st.write("Upload an image and let AI do its magic!")
    
    @st.cache_resource
    def load_cached_model():
        return load_model()
    
    @st.cache_resource
    def load_cached_blip_model():
        return get_model_and_processor()

    @st.cache_resource
    def load_cached_detailed_blip_model():
        return get_detailed_model_and_processor()

    model = load_cached_model()
    blip_processor, blip_model = load_cached_blip_model()
    detailed_blip_processor, detailed_blip_model = load_cached_detailed_blip_model()
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        image = st.image(
            uploaded_file, caption="Uploaded Image", use_container_width=True
        )
        
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Classify Image"):
                with st.spinner("Analyzing Image for Classification..."):
                    pil_image = Image.open(uploaded_file)
                    predictions = classify_image(model, pil_image)
                    
                    if predictions:
                        st.subheader("Classification Predictions")
                        for _, label, score in predictions: 
                            st.write(f"{label}: {score:.2%}")

        with col2:
            if st.button("Describe Image"):
                with st.spinner("Analyzing Image for Description..."):
                    with open("temp_image.jpg", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    description = describe_image(blip_processor, blip_model, "temp_image.jpg")
                    
                    if description:
                        st.subheader("Image Description")
                        st.write(description)

        with col3:
            if st.button("Describe in Detail"):
                with st.spinner("Analyzing Image for Detailed Description..."):
                    with open("temp_image.jpg", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    detailed_description = describe_image_in_detail(detailed_blip_processor, detailed_blip_model, "temp_image.jpg")
                    
                    if detailed_description:
                        st.subheader("Detailed Image Description")
                        st.write(detailed_description)
                        
if __name__ == "__main__":
    main()
