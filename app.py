from typing import BinaryIO, List, Tuple

import numpy as np
import streamlit as st
from keras.models import Model, load_model
from keras.preprocessing import image
from PIL import Image

# Load model and normalize values
model = load_model("my_model.keras")
mean = 120.76826477050781
std = 64.1497802734375
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def preprocess_image(uploaded_file: BinaryIO, mean: float, std: float) -> np.ndarray:
    """
    Preprocesses the uploaded image for prediction.
    Args:
        uploaded_file (file): Uploaded image file.
        mean (float): Mean value for normalization.
        std (float): Standard deviation value for normalization.
    Returns:
        np.ndarray: Preprocessed image ready for prediction.
    """
    resized_img = Image.open(uploaded_file).convert("RGB").resize((32, 32))
    resized_img_array = image.img_to_array(resized_img)
    normalized_img = (resized_img_array - mean) / (std + 1e-7)
    normalized_img = normalized_img.reshape((1, 32, 32, 3))

    return normalized_img


def predict_image(
    model: Model, image: np.ndarray, class_names: List[str]
) -> Tuple[str, float, np.ndarray]:
    """
    Predicts the class of a given image using the trained model.
    Args:
        model (keras.Model): Trained Keras model.
        image (np.ndarray): Preprocessed image.
        class_names (list): List of class names.
    Returns:
        Tuple[str, float, np.ndarray]: Predicted class name,
        confidence percentage and prediction output.
    """
    prediction = model.predict(image)
    predicted_idx = np.argmax(prediction)
    predicted_class = class_names[predicted_idx]
    confidence = np.max(prediction) * 100

    return predicted_class, confidence, prediction


# Streamlit app interface
st.set_page_config(page_title="CIFAR-10 Classifier")
st.title("🔍 CIFAR-10 Image Classifier")
st.markdown("📸 Upload an image and let the model predict!")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded image", width=600)
    normalized_image = preprocess_image(uploaded_file=uploaded_file, mean=mean, std=std)
    predicted_class, confidence, prediction = predict_image(
        model=model, image=normalized_image, class_names=class_names
    )
    confidence_threshold = 60.0
    if confidence >= confidence_threshold:
        st.markdown(f"📖 Prediction: {predicted_class}")
        st.markdown(f"😎 Confidence: {confidence:.2f} %")
    else:
        st.markdown("🤔 The model is not sure about its prediction...")
        st.markdown(
            "BUT!!! You can still see its top k predictions on your provided image!"
        )
        k = st.slider("Pick your k: ", min_value=1, max_value=10)
        prediction_idx_sorted = np.argsort(prediction)
        top_indinces = prediction_idx_sorted[0][-k:][::-1]
        st.markdown("🎯 Top-k Predictions:")
        for i, idx in enumerate(top_indinces):
            st.markdown(f"{i+1}.")
            st.markdown(f"📖 Prediction: {class_names[idx]}")
            st.markdown(f"😎 Confidence: {prediction[0][idx] * 100:.2f} %")
