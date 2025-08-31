# app.py
import streamlit as st
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import gdown

# ------------------------------
# Download the model if it doesn't exist
# ------------------------------
MODEL_URL = "https://github.com/asparagusty/SuperVision/releases/download/v1.0.0/model.h5"
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading pre-trained model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ------------------------------
# Load the model
# ------------------------------
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")

# ------------------------------
# App title
# ------------------------------
st.title("Dog vs Cat Classifier 🐶🐱")

# ------------------------------
# Upload image
# ------------------------------
uploaded_file = st.file_uploader("Upload an image of a cat or dog", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).resize((150, 150))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    
    # Predict
    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        st.success("It's a Dog! 🐶")
    else:
        st.success("It's a Cat! 🐱")
