import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import cv2

@st.cache_resource
def load_model():
    base_model = ResNet50(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)
    return model

def preprocess(img):
    img = img.convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def get_features(model, img):
    preprocessed = preprocess(img)
    features = model.predict(preprocessed)
    return features

def calculate_similarity(model, img1, img2):
    f1 = get_features(model, img1)
    f2 = get_features(model, img2)
    sim = cosine_similarity(f1, f2)[0][0]
    return sim

def main():
    st.set_page_config(page_title="AI Image Similarity Finder")
    st.title("AI Image Similarity Finder")
    st.write("Upload two images and the AI will tell you how visually similar they are!")

    model = load_model()

    col1, col2 = st.columns(2)

    with col1:
        img_file1 = st.file_uploader("Upload First Image", type=["jpg", "jpeg", "png"])

    with col2:
        img_file2 = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png"])

    if img_file1 and img_file2:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file1, caption="Image 1", use_container_width=True)
        with col2:
            st.image(img_file2, caption="Image 2", use_container_width=True)

        if st.button("Compare Images"):
            with st.spinner("Analyzing similarity..."):
                img1 = Image.open(img_file1)
                img2 = Image.open(img_file2)
                similarity = calculate_similarity(model, img1, img2)
                st.success(f"Similarity Score: {similarity:.2%} ")

                if similarity > 0.8:
                    st.info("The images are highly similar!")
                elif similarity > 0.5:
                    st.warning("The images are somewhat similar.")
                else:
                    st.error("The images are quite different.")

if __name__ == "__main__":
    main()

