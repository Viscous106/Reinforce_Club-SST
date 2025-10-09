# AI Image Describer

A Streamlit application that uses the **EfficientNetV2M** deep learning model for image classification and the **Meta AI API** to generate funny, absurd descriptions (No API Key required!).

---

## How It Works

1.  **Classification:** The app loads a cached **EfficientNetV2M** model (pre-trained on ImageNet) to classify an uploaded image and get the top-ranked predictions.
2.  **Description Generation:** The top classifications are sent as a prompt to the **Meta AI API**, which generates a short, creative, and often absurd description based on the terms.
3.  **Interface:** A simple Streamlit web interface handles image upload and displays the final caption.

---

## Prerequisites

You need **Python 3.10.18+** and the necessary packages installed.

---

## Installation

1.  **Install Dependencies:**

    ```bash
    uv sync
    ```
    OR
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the App:**

    ```bash
    streamlit run main.py
    ```

---

## Usage

1.  Navigate to the app's local URL (e.g., `http://localhost:8501`).
2.  Use the file uploader to select an image.
3.  Click the **"Describe"** button and wait for the AI's funny caption to appear.