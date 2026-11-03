
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="centered"
)

# Load CNN Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_cancer_cnn.keras")

model = load_model()

# App Title
st.title("🔬 Skin Cancer Detection")
st.subheader("CNN Based Image Classification")

st.write(
    "Upload a skin image to get a prediction "
    "from the trained CNN model."
)

# Upload Image
uploaded_file = st.file_uploader(
    "Upload Skin Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read Image
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Skin Image",
        width=300
    )

    if st.button("Predict"):

        with st.spinner("Analyzing image..."):

            # Preprocessing
            img = image.resize((128, 128))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            prediction = model.predict(img_array, verbose=0)[0][0]

            st.subheader("Prediction Result")

            if prediction >= 0.5:
                st.error("Model Prediction: Malignant")
            else:
                st.success("Model Prediction: Benign")

            st.write(
                f"Model output score: {float(prediction):.4f}"
            )

            st.warning(
                "This is an educational project only. "
                "It cannot diagnose skin cancer. "
                "Please consult a qualified dermatologist "
                "for medical assessment."
            )

# Footer
st.divider()
st.caption("Skin Cancer Detection | CNN Machine Learning Project")
