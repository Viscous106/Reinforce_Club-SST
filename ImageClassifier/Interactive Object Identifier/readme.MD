# 🔍 Interactive Object Identifier

An **AI-powered image identifier** built with **Streamlit** and **TensorFlow’s ResNet50** model.  
It lets you **upload an image** or **capture one via webcam**, identifies what’s in it, and even **speaks out the results** using text-to-speech.

---

## 🚀 Features

- 🧠 **AI Object Detection** – Uses pre-trained **ResNet50 (ImageNet)** for accurate classification.
- 📸 **Two Input Modes** – Upload an image or take one with your webcam.
- 🔊 **Voice Output** – Speaks out top 3 predictions with confidence levels.
- ⚡ **Instant & Cached** – Loads model once with `@st.cache_resource()` for fast performance.

---

## 🧩 How It Works

1. Load the **ResNet50** model (cached for efficiency).
2. User uploads or captures an image.
3. Image is resized (224×224), preprocessed, and passed into the model.
4. The model predicts the top 3 object categories.
5. Results are displayed and read aloud using **pyttsx3**.

---

## 📁 File Structure

Interactive-Object-Identifier/
│
├── main.py # Main Streamlit application
├── requirements.txt # Python dependencies
└── README.md # Project documentation
