import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from pathlib import Path

# =====================================
# 1. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Pollution Detection System",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Pollution Detection System")
st.write("Upload an image to detect the type of pollution.")

st.divider()

# =====================================
# 2. PROJECT FILE PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "pollution_model.keras"
CLASS_PATH = BASE_DIR / "classes.json"

IMG_SIZE = (64, 64)

# =====================================
# 3. LOAD TRAINED MODEL
# =====================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
    st.success("AI Model Loaded Successfully!")

except Exception as e:
    st.error(f"Unable to load model: {e}")

    st.info(
        "Make sure pollution_model.keras is present "
        "in the same folder as app.py."
    )

    st.stop()


# =====================================
# 4. LOAD CLASS NAMES
# =====================================

try:

    if not CLASS_PATH.is_file():
        raise FileNotFoundError(
            f"classes.json not found: {CLASS_PATH}"
        )

    with open(CLASS_PATH, "r") as file:
        class_data = json.load(file)

    if isinstance(class_data, list):
        class_names = class_data

    elif isinstance(class_data, dict):

        # Supports {"0": "Air Pollution", ...}
        # and {"Air Pollution": 0, ...}

        if all(str(k).isdigit() for k in class_data):
            class_names = [
                class_data[k]
                for k in sorted(class_data, key=int)
            ]

        else:
            class_names = [
                name for name, index in
                sorted(class_data.items(), key=lambda item: item[1])
            ]

    else:
        raise ValueError("Invalid classes.json format")

except Exception as e:

    st.error(f"Unable to load class names: {e}")
    st.stop()


# =====================================
# 5. IMAGE UPLOAD SECTION
# =====================================

st.subheader("📸 Upload Pollution Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)


# =====================================
# 6. IMAGE PREDICTION
# =====================================

if uploaded_file is not None:

    try:

        # Read uploaded image
        img = Image.open(uploaded_file).convert("RGB")

        # Display image
        st.image(
            img,
            caption="Uploaded Image",
            use_container_width=True
        )

        # Detect button
        if st.button("🔍 Detect Pollution", type="primary"):

            with st.spinner("Analyzing image..."):

                # Resize image
                resized_img = img.resize(IMG_SIZE)

                # Convert image into array
                x = np.array(
                    resized_img,
                    dtype=np.float32
                )

                # Add batch dimension
                x = np.expand_dims(x, axis=0)

                # IMPORTANT:
                # Uncomment ONLY if your training code
                # normalized images by dividing by 255.
                #
                # x = x / 255.0

                # Make prediction
                prediction = model.predict(x, verbose=0)

                # Validate prediction
                if prediction.ndim != 2:
                    raise ValueError(
                        "Unexpected model output shape."
                    )

                if prediction.shape[1] != len(class_names):
                    raise ValueError(
                        "Model output classes do not match "
                        "classes.json."
                    )

                # Get predicted class
                predicted_index = int(
                    np.argmax(prediction[0])
                )

                predicted_class = str(
                    class_names[predicted_index]
                )

                confidence = float(
                    prediction[0][predicted_index]
                ) * 100

            # =====================================
            # 7. DISPLAY DETECTION RESULT
            # =====================================

            st.success("Prediction Completed!")

            st.subheader("Detection Result")

            if predicted_class.lower() == "air pollution":

                st.error("🌫️ Air Pollution Detected")

            elif predicted_class.lower() == "water pollution":

                st.info("💧 Water Pollution Detected")

            elif predicted_class.lower() == "noise pollution":

                st.warning("🔊 Noise Pollution Detected")

            else:

                st.success(
                    f"Detected: {predicted_class}"
                )

            # Confidence
            st.metric(
                label="Prediction Confidence",
                value=f"{confidence:.2f}%"
            )

            # =====================================
            # 8. DISPLAY ALL PROBABILITIES
            # =====================================

            st.subheader("Prediction Probabilities")

            for i, class_name in enumerate(class_names):

                probability = float(
                    prediction[0][i]
                ) * 100

                st.write(
                    f"{class_name}: {probability:.2f}%"
                )

                st.progress(
                    min(max(probability / 100, 0), 1)
                )

    except Exception as e:

        st.error(f"Prediction Error: {e}")


# =====================================
# 9. FOOTER
# =====================================

st.divider()

st.caption(
    "🌍 Pollution Detection System | "
    "CNN Deep Learning Model"
)