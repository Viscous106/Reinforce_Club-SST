## AI Image Similarity Finder

This PR adds a new project folder for **AI Image Similarity Finder**. The application allows users to compare two images and find out how visually similar they are using AI.

### Features

- Upload two images in JPG, JPEG, or PNG format.
- Preprocesses images (resize to 224×224, normalize) for AI analysis.
- Extracts features using **ResNet50** pretrained on ImageNet.
- Calculates **cosine similarity** between images.
- Displays a similarity score with an easy-to-understand interpretation:
  - 💯 High similarity → images are very alike
  - ⚠️ Medium similarity → images share some features
  - ❌ Low similarity → images are quite different
- Interactive **Streamlit** interface with side-by-side image display.
- Cached model loading for faster performance.

### How it Works

1. Users upload two images via the web interface.
2. Images are preprocessed and passed through ResNet50 to extract feature vectors.
3. Cosine similarity is calculated between the two feature vectors.
4. The similarity score and interpretation are displayed on the app.

This PR introduces the full project structure including `app.py` and necessary dependencies for running the app locally.
