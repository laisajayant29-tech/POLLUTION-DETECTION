
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(
    page_title="Pollution Detection System",
    page_icon="🌍",
    layout="centered"
)


st.title("🌍 Pollution Detection System")
st.write("Upload an image to detect the type of pollution using CNN.")

st.divider()


TRAIN_PATH = r"C:\Users\INTEL AI DATA LAB\Downloads\cap_3\jai.python\Country Flag.Dataset\POLLUTION DETECTION APP\Train"

@st.cache_resource
def train_model():

    data = tf.keras.utils.image_dataset_from_directory(
        TRAIN_PATH,
        image_size=(64, 64),
        batch_size=2,
        shuffle=True
    )

    class_names = data.class_names

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 3)),
        tf.keras.layers.Rescaling(1./255),

        tf.keras.layers.Conv2D(
            16, (3, 3), activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            len(class_names),
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(data, epochs=50)

    return model, class_names


with st.spinner("Loading and training CNN model..."):
    model, class_names = train_model()

st.success("CNN Model is ready!")

with st.expander("View Model Information"):
    st.write("Model Architecture")
    model.summary(print_fn=lambda x: st.text(x))

    st.write("Classes:", class_names)

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

        img = img.resize((64, 64))

        x = np.array(img)

        x = np.expand_dims(x, axis=0)

        prediction = model.predict(x, verbose=0)

        predicted_index = np.argmax(prediction)

        predicted_class = class_names[predicted_index]

        confidence = float(prediction[0][predicted_index]) * 100

    
        st.divider()

        st.subheader("📊 Prediction Result")

        st.success(f"Detected Pollution: {predicted_class}")

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2f}%"
        )

        st.subheader("Class Prediction Probabilities")

        for i, name in enumerate(class_names):

            probability = float(prediction[0][i]) * 100

            st.write(f"{name}: {probability:.2f}%")

            st.progress(probability / 100)

st.divider()

st.caption(
    "Pollution Detection System | CNN Deep Learning Project"
)
