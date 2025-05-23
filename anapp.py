import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("optimized_animal_classifier.h5")

# Define class names (update according to your model's output)
class_names = ['Cat', 'Dog', 'Elephant', 'Lion', 'Tiger']  # example classes

# Streamlit UI
st.title("🐾 Animal Classifier")
st.write("Upload an image of an animal, and the model will classify it.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image
    img = img.resize((224, 224))  # adjust based on your model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # normalize if the model was trained with normalized inputs

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    # Show result
    st.write(f"### Prediction: {predicted_class}")
