# AI-Powered Image Analysis Tool

This project is a Streamlit web application that provides a suite of AI-powered tools for image analysis. You can upload an image and perform the following actions:

- **Image Classification**: Identify the main subject of the image.
- **Image Description**: Generate a short, descriptive caption for the image.
- **Detailed Image Description**: Generate a more detailed and descriptive caption.

## Features

- **Image Classification**: Utilizes the MobileNetV2 model, pre-trained on the ImageNet dataset, to classify the uploaded image and display the top 3 predictions with their confidence scores.
- **Image Description (Base)**: Employs the Salesforce BLIP (Base) model to generate a concise, one-sentence description of the image content.
- **Image Description (Detailed)**: Uses the Salesforce BLIP (Large) model to produce a more elaborate and descriptive caption, offering richer details about the image.

## How to Run

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>/ImageClassifier/builtDay2-YashVirulkar
   ```

2. **Install the dependencies:**
   Make sure you have Python 3.x installed. Then, install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run main.py
   ```

4. **Upload an image:**
   Open your web browser and navigate to the local URL provided by Streamlit. Click on the "Browse files" button to upload an image and use the buttons to trigger the different AI functionalities.

## Models Used

- **Image Classification**: [MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2)
- **Image Captioning**: [Salesforce BLIP (Base and Large)](https://huggingface.co/Salesforce/blip-image-captioning-base)
