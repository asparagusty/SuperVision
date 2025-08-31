import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import requests

st.title("🐶🐱 Cat vs Dog Classifier (SuperVision)")

MODEL_PATH = "model.h5"
MODEL_URL = "https://github.com/asparagusty/SuperVision/releases/download/v1.0.0/model.h5"  # Replace with GitHub raw link or gdown link

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    r = requests.get(MODEL_URL, allow_redirects=True)
    open(MODEL_PATH, 'wb').write(r.content)
    st.success("Model downloaded successfully!")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
class_labels = ["Cat", "Dog"]

uploaded_file = st.file_uploader("Upload an image...", type=["jpg","jpeg","png"])
if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)/255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.markdown(f"### Prediction: **{class_labels[predicted_class]}** 🐾")
    st.write(f"Confidence: {confidence:.2f}")
