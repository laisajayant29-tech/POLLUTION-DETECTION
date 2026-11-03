
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Pollution Detection System",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Pollution Detection System")
st.write("Upload an image to detect the type of pollution.")

st.divider()

# Load trained model
@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "pollution_model.keras"
    )

    with open("class_names.txt", "r") as f:
        class_names = f.read().splitlines()

    return model, class_names


# Check model files
if not os.path.exists("pollution_model.keras") or not os.path.exists("class_names.txt"):
    st.error("Model files are missing. Please upload them to GitHub.")
    st.stop()

model, class_names = load_model()

st.success("CNN Model Loaded Successfully!")

# Image upload
st.subheader("📸 Upload Pollution Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Detect Pollution", use_container_width=True):

        # Preprocessing
        img = img.resize((64, 64))

        x = np.array(img, dtype=np.float32)

        x = np.expand_dims(x, axis=0)

        # Prediction
        prediction = model.predict(x, verbose=0)

        predicted_index = np.argmax(prediction)

        predicted_class = class_names[predicted_index]

        confidence = float(
            prediction[0][predicted_index] * 100
        )

        # Display result
        st.divider()

        st.subheader("📊 Prediction Result")

        st.success(
            f"Detected Pollution: {predicted_class}"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        # Display probabilities
        st.subheader("Class Prediction Probabilities")

        for i, name in enumerate(class_names):

            probability = float(prediction[0][i] * 100)

            st.write(f"{name}: {probability:.2f}%")

            st.progress(probability / 100)

st.divider()

st.caption("Pollution Detection System | CNN Deep Learning")
