import streamlit as st
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import gdown

# ------------------------------
# Download weights from GitHub release
# ------------------------------
WEIGHTS_URL = "https://github.com/asparagusty/SuperVision/releases/download/v1.0.0/model.weights.h5"
WEIGHTS_PATH = "model.weights.h5"

if not os.path.exists(WEIGHTS_PATH):
    st.info("Downloading pre-trained model weights...")
    gdown.download(WEIGHTS_URL, WEIGHTS_PATH, quiet=False)

# ------------------------------
# Define model architecture
# ------------------------------
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
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


