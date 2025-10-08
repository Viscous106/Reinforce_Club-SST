import streamlit as st
import cv2
from PIL import Image
import model
import numpy as np


st.set_page_config(page_title="AI Classifier", layout="centered")
st.title("AI Video frame Classifier using Xception Model")
st.write("The app will capture a frame from your webcam and analyze it to 'somewhat' tell what it can find on the video feed. This is made just to have some laugh and is in no way really accurate")

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
ai_model = model.load_model()

# cap = cv2.VideoCapture(0)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = ai_model

    def recv(self, frame: av.VideoFrame):

        img = frame.to_ndarray(format="bgr24")

        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        predictions = model.classify_image(self.model, pil_image)


        if predictions:
            top_prediction = predictions[0]
            _, label, score = top_prediction
            text = f"{label.replace('_', ' ').title()}: {score:.2%}"

            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(img, (5, 5), (10 + text_width, 20 + text_height), (0,0,0), -1)
            cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="classifier",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)