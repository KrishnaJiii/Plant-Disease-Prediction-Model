import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image

# Load model & classes
model = tf.keras.models.load_model("plant_disease_prediction_model.h5")
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

st.title("🌱 Plant Disease Prediction")
st.write("Upload a leaf image to predict the disease")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"✅ Predicted Disease: **{predicted_class}**")

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 10px;
        right: 10px;
        color: #888888;
        font-size: 14px;
        z-index: 100;
    }
    </style>

    <div class="footer">
        Developed by <b>Krishna Bedi</b>
    </div>
    """,
    unsafe_allow_html=True
)
