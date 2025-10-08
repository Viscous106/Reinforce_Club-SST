# Live AI Webcam Classifier

This is a web application that uses your computer's webcam to perform real-time object classification using the pre-trained Xception deep learning model. The application streams the video feed directly in the browser and overlays the top prediction from the model onto the video.

## Features

* **Real-Time Video Streaming**: Captures live video from your webcam directly in the browser.
* **Live AI Classification**: Each frame is analyzed by the Xception model to identify objects.
* **Dynamic Overlay**: The top prediction and its confidence score are drawn directly onto the video feed.
* **Interactive Web UI**: Built with Streamlit for a clean and easy-to-use interface.


## How It Works

The application is built with a few key libraries:

* **Streamlit**: The core framework for creating the interactive web application.
* **streamlit-webrtc**: A Streamlit component that handles the real-time communication (WebRTC) needed to safely stream video from the browser to the Python backend for processing.
* **OpenCV**: Used for basic image processing, such as converting color formats and drawing the prediction text onto each video frame.
* **TensorFlow/Keras**: Provides the pre-trained Xception model, which is a powerful deep neural network trained on the ImageNet dataset.
* **Pillow & NumPy**: Used for handling and manipulating image data.


## File Structure

  * `main.py`: This is the main script that runs the Streamlit application. It handles the UI, sets up the `webrtc_streamer` component, and uses a `VideoTransformer` class to process the video feed.
  * `model.py`: This file contains all the logic related to the machine learning model. It is responsible for loading the pre-trained Xception model and preprocessing the image frames before they are sent for prediction.
  * `requirements.txt`: Lists all the Python dependencies required for the project.
