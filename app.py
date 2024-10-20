import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import pickle

# Load the models

st.title('Fight Prediction Using ResNet152')
st.caption('-In Model-1 Use Resnet152')
st.caption('-In Model-2 Use Resnet152 with fine Tuning')
st.caption('-In Model-2 Use Efficientnetb7_Process')
with open('index_to_class.pkl', 'rb') as f:
    classes = pickle.load(f)

# Preprocessing functions
def Resnet152_Process(img_path, target_size=(224, 224)):
    img = Image.open(img_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return img_array    

def Efficientnetb7_Process(img_path, target_size=(128, 128)):
    img = Image.open(img_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return img_array    

# Sidebar options to choose the model
st.sidebar.title("Choose a Model")
model_option = st.sidebar.selectbox("Select Model", ("ResNet152 (Model 2)","ResNet152 (Model 1)", "EfficientNetB7 (Model 3)"))

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png",'webp'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Model selection and image processing
    if model_option == "ResNet152 (Model 1)":
        model1 = tf.keras.models.load_model('model1.h5')
        processed_img = Resnet152_Process(uploaded_file)
        processed_img = np.expand_dims(processed_img, axis=0)
        prediction = model1.predict(processed_img)
        predicted_class = classes[int(np.argmax(prediction, axis=1))]
    elif model_option == "ResNet152 (Model 2)":
        model2 = tf.keras.models.load_model('model2.h5')
        processed_img = Resnet152_Process(uploaded_file)
        processed_img = np.expand_dims(processed_img, axis=0)
        prediction = model2.predict(processed_img)
        predicted_class = classes[int(np.argmax(prediction, axis=1))]
    else:
        model3 = tf.keras.models.load_model('model3.h5')
        processed_img = Efficientnetb7_Process(uploaded_file)
        processed_img = np.expand_dims(processed_img, axis=0)
        prediction = model3.predict(processed_img)
        predicted_class = classes[int(np.argmax(prediction, axis=1))]

    st.success(f"Predicted Class: {predicted_class}")
