import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

# Function to download model if not found
def download_model():
    file_id = "1e2vT2gSG0wW1hZ2p-q8o4DIHygwJwTZ1"  # Replace with actual File ID
    output = "cat_dog_classifier.h5"
    
    if not os.path.exists(output):
        st.info(f"Downloading {output} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
        st.success(f"{output} download complete!")

# Download the model if not available
download_model()

# Load the trained model
@st.cache_data
def load_model():
    return tf.keras.models.load_model("cat_dog_classifier.h5")

model = load_model()

# Streamlit UI
st.title("🐱🐶 Cat vs Dog Image Classifier")
st.markdown("### Upload an image to classify whether it's a **cat or a dog**.")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    def preprocess_image(image):
        image = image.resize((224, 224))  # Resize to match model input size
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image

    image = preprocess_image(Image.open(uploaded_file))

    # Prediction
    if st.button("🔍 Classify Image"):
        prediction = model.predict(image)
        class_label = "Dog 🐶" if prediction[0][0] > 0.5 else "Cat 🐱"
        st.success(f"Predicted Class: **{class_label}**")

# Footer
st.markdown("---")
st.markdown("**Developed by YARAMALA SAIKUMAR** | Powered by Machine Learning & Streamlit 🚀")
