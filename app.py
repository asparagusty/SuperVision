import streamlit as st
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import gdown

# ------------------------------
# Download weights if they don't exist
# ------------------------------
WEIGHTS_URL = "https://github.com/asparagusty/SuperVision/releases/download/v1.0.0/model.h5"
WEIGHTS_PATH = "model.h5"

if not os.path.exists(WEIGHTS_PATH):
    st.info("Downloading pre-trained model weights...")
    gdown.download(WEIGHTS_URL, WEIGHTS_PATH, quiet=False)

# ------------------------------
# Define model architecture
# ------------------------------
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

# Load weights
try:
    model.load_weights(WEIGHTS_PATH)
except Exception as e:
    st.error(f"Failed to load weights: {e}")

# ------------------------------
# Streamlit app
# ------------------------------
st.title("Dog vs Cat Classifier 🐶🐱")

uploaded_file = st.file_uploader("Upload an image of a cat or dog", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = model.predict(img_array)[0]

    if np.argmax(prediction) == 0:
        st.success("It's a Cat! 🐱")
    else:
        st.success("It's a Dog! 🐶")

